import logging
import pdb
from functools import partial

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import normflows as nf
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.distributions import (
    Cauchy,
    Independent,
    LogNormal,
    LowRankMultivariateNormal,
    MultivariateNormal,
    Normal,
)

from sade.models.layers import PositionalEncoding3D, SpatialNorm3D
from sade.models.layerspp import FlowAttentionBlock, get_conv_layer

from . import registry


@torch.jit.script
def gaussian_logprob(z, ldj):
    _GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))
    return _GCONST_ - 0.5 * torch.sum(z**2, dim=-1) + ldj


class StandardDistribution(Independent):
    SUPPORTED_DISTRIBUTIONS = {
        "cauchy": Cauchy,
        "normal": Normal,
        # This will require a speecial trnasform at the end of the flow
        # An exponential will do
        # "lognormal": LogNormal
    }

    def __init__(
        self,
        base_distribution_name,
        *event_shape: int,
        device=None,
        dtype=None,
        validate_args=True,
    ):
        assert (
            base_distribution_name in self.SUPPORTED_DISTRIBUTIONS.keys()
        ), f"Base distribution {base_distribution_name} not supported.Supported distributions are {self.SUPPORTED_DISTRIBUTIONS.keys()}"

        loc = torch.tensor(0.0, device=device, dtype=dtype).repeat(event_shape)
        scale = torch.tensor(1.0, device=device, dtype=dtype).repeat(event_shape)
        self.base = self.SUPPORTED_DISTRIBUTIONS[base_distribution_name]

        super().__init__(
            self.base(loc, scale, validate_args=validate_args),
            len(event_shape),
            validate_args=validate_args,
        )


class LowRankMVN(nn.Module):
    def __init__(self, input_dims, low_rank_dims=256):
        super().__init__()

        self.cov_factor = nn.Parameter(
            torch.randn(input_dims, low_rank_dims), requires_grad=True
        )
        self.cov_diag = nn.Parameter(torch.ones(input_dims), requires_grad=True)
        self.mean = nn.Parameter(torch.zeros(input_dims), requires_grad=False)

    @property
    def diag(self):
        return torch.nn.functional.softplus(self.cov_diag) + 1e-5

    @property
    def covariance_matrix(self):
        return LowRankMultivariateNormal(
            loc=self.mean, cov_factor=self.cov_factor, cov_diag=self.diag
        ).covariance_matrix

    def forward(self, x):
        return LowRankMultivariateNormal(
            loc=self.mean, cov_factor=self.cov_factor, cov_diag=self.diag
        ).log_prob(x)

    def log_prob(self, x):
        return self(x)


class MVN(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.D = D = input_dims
        lower_tril_numel = D * (D + 1) // 2 - D
        self.cov_factor = nn.Parameter(torch.ones(lower_tril_numel), requires_grad=True)
        self.cov_diag = nn.Parameter(torch.ones(D), requires_grad=True)
        self.mean = nn.Parameter(torch.zeros(D), requires_grad=True)

        self.tril_idx = torch.tril_indices(D, D, offset=-1).tolist()

    @property
    def diag(self):
        return torch.nn.functional.softplus(self.cov_diag + 1e-5).diag()

    @property
    def scale_tril(self):
        tril = torch.zeros(self.D, self.D, requires_grad=False)
        tril[self.tril_idx] = self.cov_factor
        return self.diag + tril

    def forward(self, x):
        return MultivariateNormal(loc=self.mean, scale_tril=self.scale_tril).log_prob(x)

    def log_prob(self, x):
        return self(x)


def subnet_fc(c_in, c_out, ndim=256, act=nn.GELU(), input_norm=False):
    return nn.Sequential(
        nn.LayerNorm(c_in) if input_norm else nn.Identity(),
        nn.Linear(c_in, ndim),
        nn.LayerNorm(ndim),
        act,
        nn.Linear(ndim, ndim),
        nn.LayerNorm(ndim),
        act,
        nn.Linear(ndim, c_out),
    )


class PatchFlow(torch.nn.Module):
    """
    Contructs a conditional flow model that operates on patches of an image.
    Each patch is fed into the same flow model i.e. parameters are shared across patches.
    The flow models are conditioned on a positional encoding of the patch location.
    The resulting patch-densities can then be recombined into a full image density.
    """

    def __init__(
        self,
        config,
        input_size,
    ):
        super().__init__()

        param_keys = ["kernel_size", "stride", "padding"]
        for key in param_keys:
            assert key in config.flow.local_patch_config
            assert (
                key in config.flow.global_patch_config
                if config.flow.use_global_context
                else True
            )

        channels = input_size[0]
        self.device = config.device

        # Base distribution
        dist_name = config.flow.base_distribution
        if dist_name == "gaussian_mixture":
            self.base_distribution = nf.distributions.base.GaussianMixture(
                n_modes=10, dim=channels, trainable=True
            )
        elif dist_name == "normflow":
            self.base_distribution = nf.distributions.base.DiagGaussian(
                channels, trainable=False
            )
        elif dist_name == "multivariate_normal":
            self.base_distribution = MVN(channels)
        else:
            self.base_distribution = StandardDistribution(
                dist_name, channels, device=self.device
            )

        # Patch parameters
        self.local_patch_config = config.flow.local_patch_config
        self.global_patch_config = config.flow.global_patch_config

        self.kernel_size = self.local_patch_config.kernel_size
        self.stride = self.local_patch_config.stride
        self.padding = self.local_patch_config.padding

        # Used to chunk the input into in fast_forward (vectorized)
        self.patch_batch_size = config.flow.patch_batch_size

        # Params for neural nets within a flow block
        self.num_blocks = config.flow.num_blocks
        self.context_embedding_size = config.flow.context_embedding_size
        self.use_global_context = config.flow.use_global_context
        self.global_embedding_size = config.flow.global_embedding_size
        self.input_norm = config.flow.input_norm

        with torch.no_grad():
            # Pooling for local "patch" flow
            # Each patch-norm is input to the shared conditional flow model
            self.local_pooler = SpatialNorm3D(
                channels, **self.local_patch_config
            ).requires_grad_(False)

            # Compute the spatial resolution of the patches
            _, self.channels, h, w, d = self.local_pooler(torch.empty(1, *input_size)).shape
            self.spatial_res = (h, w, d)
            self.num_patches = h * w * d
            self.position_encoder = PositionalEncoding3D(self.context_embedding_size)
            logging.info(
                f"Generating {self.kernel_size}x{self.kernel_size}x{self.kernel_size} patches from input size: {input_size}"
            )
            logging.info(f"Pooled spatial resolution: {self.spatial_res}")
            logging.info(f"Number of flows / patches: {self.num_patches}")

        context_dims = self.context_embedding_size

        if self.use_global_context:
            # Pooling for global "low resolution" flow
            # self.norm_pooler = SpatialNorm3D(
            #     channels, **self.global_patch_config
            # ).requires_grad_(False)
            # self.conv_pooler = get_conv_layer(
            #     3, channels, channels, kernel_size=3, stride=2
            # )
            # self.global_pooler = nn.Sequential(
            #     self.norm_pooler,
            #     self.conv_pooler,
            # )
            c = 2
            self.conv_init = get_conv_layer(
                3,
                c,
                c,
                kernel_size=self.global_patch_config["kernel_size"],
                stride=self.global_patch_config["stride"],
            )
            self.conv_pooler = get_conv_layer(3, c, c, kernel_size=3, stride=2)
            self.global_pooler = nn.Sequential(
                self.conv_init,
                self.conv_pooler,
            )

            # Spatial resolution of the global context patches
            _, _, h, w, d = self.global_pooler(torch.empty(1, c, *input_size[1:])).shape
            logging.info(f"Global Context Shape: {(h, w, d)}")
            self.global_attention = FlowAttentionBlock(
                input_size=(c, h, w, d),
                embed_dim=self.global_embedding_size,
                outdim=self.context_embedding_size,
            )
            context_dims += self.context_embedding_size

        num_features = self.channels
        self.flow = self.build_flow_head(
            num_features,
            context_dims,
            # input_norm=self.input_norm,
            num_blocks=self.num_blocks,
        )

        self.init_weights()
        self.to(self.device)
        self.flow.to(self.device)

    def init_weights(self):
        # Initialize weights with Xavier
        linear_modules = list(
            filter(lambda m: isinstance(m, nn.Linear), self.flow.modules())
        )
        total = len(linear_modules)
        # pdb.set_trace()
        for idx, m in enumerate(linear_modules):
            # Last layer gets init w/ zeros
            if idx == total - 1:
                nn.init.zeros_(m.weight.data)
            else:
                nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        if self.use_global_context:
            for m in self.global_attention.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # print(m)
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)

    def build_cflow_head(self, input_dim, conditioning_dim, input_norm=False, num_blocks=2):
        coder = Ff.SequenceINN(input_dim)
        for k in range(num_blocks):
            # idx = int(k % 2 == 0)
            coder.append(
                Fm.AllInOneBlock,
                cond=0,
                cond_shape=(conditioning_dim,),
                subnet_constructor=partial(subnet_fc, input_norm=input_norm, act=nn.GELU()),
                global_affine_type="SOFTPLUS",
                permute_soft=True,
                affine_clamping=1.9,
            )

        return coder

    def build_flow_head(self, input_dim, conditioning_dim, num_blocks=2):
        num_res_blocks = 2
        hidden_units = 256
        flows = []

        flows.append(
            nf.flows.MaskedAffineAutoregressive(
                input_dim,
                hidden_features=hidden_units,
                num_blocks=2,
                context_features=conditioning_dim,
            )
        )

        for i in range(num_blocks):
            flows += [nf.flows.LULinearPermute(input_dim)]
            flows += [
                nf.flows.CoupledRationalQuadraticSpline(
                    input_dim,
                    num_res_blocks,
                    hidden_units,
                    num_context_channels=conditioning_dim,
                    num_bins=8
                    # activation=nn.LeakyReLU(0.2),
                )
            ]

        # Construct flow model
        flows = flows[::-1]  # Normflow expects flows in reverse order
        nfm = nf.ConditionalNormalizingFlow(q0=self.base_distribution, flows=flows)
        # nfm.forward_kld
        return nfm

    def forward(self, x, return_attn=False, fast=True):
        raise NotImplementedError
        B, C = x.shape[0], x.shape[1]
        x_norm = self.local_pooler(x)
        self.position_encoder = self.position_encoder.cpu()
        context = self.position_encoder(x_norm)

        if self.use_global_context:
            global_pooled_image = self.global_pooler(x)
            # Every patch gets the same global context
            global_context = self.global_attention(global_pooled_image)

        if fast and self.use_global_context:
            zs, log_jac_dets = self.fast_forward(x_norm, context, global_context)

        else:
            # Patches x batch x channels
            local_patches = rearrange(x_norm, "b c h w d -> (h w d) b c")
            context = rearrange(context, "b c h w d -> (h w d) b c")

            zs = []
            log_jac_dets = []

            for patch_feature, context_vector in zip(local_patches, context):
                c = context_vector.to(self.device)

                if self.use_global_context:
                    c = torch.cat([c, global_context], dim=1)

                z, ldj = self.flow.inverse_and_log_det(
                    patch_feature,
                    context=c,
                )
                zs.append(z)
                log_jac_dets.append(ldj)
                c = c.cpu()

        # Use einops to concatenate all patches
        zs = rearrange(zs, "n b c -> (n b) c")
        log_jac_dets = rearrange(log_jac_dets, "n b -> (n b)")

        # zs = torch.cat(zs, dim=0).reshape(self.num_patches, B, C)
        # log_jac_dets = torch.cat(log_jac_dets, dim=0).reshape(self.num_patches, B)

        return zs, log_jac_dets

    def fast_forward(self, x, local_ctx, global_ctx):
        # (Patches * batch) x channels
        local_ctx = rearrange(local_ctx, "b c h w d -> (h w d) b c")
        patches = rearrange(x, "b c h w d -> (h w d) b c")

        nchunks = self.num_patches // self.patch_batch_size
        nchunks += 1 if self.num_patches % self.patch_batch_size else 0

        patches = patches.chunk(nchunks, dim=0)
        ctx_chunks = local_ctx.chunk(nchunks, dim=0)
        zs, jacs = [], []
        gc = repeat(global_ctx, "b c -> (n b) c", n=self.patch_batch_size)

        for p, ctx in zip(patches, ctx_chunks):
            # Check that patch context is same for all batch elements
            #             assert torch.isclose(c[0, :32], c[B-1, :32]).all()
            #             assert torch.isclose(c[B+1, :32], c[(2*B)-1, :32]).all()
            ctx = ctx.to(self.device)
            ctx = rearrange(ctx, "n b c -> (n b) c")
            p = rearrange(p, "n b c -> (n b) c")

            c = torch.cat([ctx, gc[: ctx.shape[0]]], dim=1)
            z, ldj = self.flow.inverse_and_log_det(p, context=c)

            zs.append(z)
            jacs.append(ldj)

            del ctx, gc, p

        return zs, jacs

    def nll(self, zs, log_jac_dets):
        # return -torch.mean(gaussian_logprob(zs, log_jac_dets))
        # pdb.set_trace()
        return -torch.mean(self.base_distribution.log_prob(zs) + log_jac_dets)

    @torch.no_grad()
    def log_density(self, x, fast=True):
        self.eval()
        b = x.shape[0]
        h, w, d = self.spatial_res
        zs, jacs = self.forward(x, fast=fast)
        # pdb.set_trace()
        # logpx = gaussian_logprob(zs, jacs)
        logpx = self.base_distribution.log_prob(zs) + jacs
        logpx = rearrange(logpx, "(h w d b) -> b h w d", b=b, h=h, w=w, d=d)
        return logpx

    @staticmethod
    def stochastic_step(scores, x_batch, flow_model, opt=None, train=False, n_patches=1):
        if train:
            flow_model.train()
            opt.zero_grad(set_to_none=True)
        else:
            flow_model.eval()

        patches, context = PatchFlow.get_random_patches(
            scores, x_batch, flow_model, n_patches
        )

        patch_feature = patches.to(flow_model.device)
        context_vector = context.to(flow_model.device)
        patch_feature = rearrange(patch_feature, "n b c -> (n b) c")
        context_vector = rearrange(context_vector, "n b c -> (n b) c")

        global_pooled_image = flow_model.global_pooler(x_batch)
        global_context = flow_model.global_attention(global_pooled_image)
        gctx = repeat(global_context, "b c -> (n b) c", n=n_patches)

        # Concatenate global context to local context
        context_vector = torch.cat([context_vector, gctx], dim=1)

        z, ldj = flow_model.flow.inverse_and_log_det(
            patch_feature,
            context=context_vector,
        )

        loss = flow_model.nll(z, ldj) * n_patches

        if train:
            loss.backward()
            opt.step()

        return loss.item() / n_patches

    @staticmethod
    def get_random_patches(scores, x_batch, flow_model, n_patches):
        h = flow_model.local_pooler(scores).cpu()
        flow_model.position_encoder = flow_model.position_encoder.cpu()
        local_patches = rearrange(h, "b c h w d -> (h w d) b c")
        context = rearrange(flow_model.position_encoder(h), "b c h w d -> (h w d) b c")

        # Get brain masks
        mask = x_batch > x_batch.min()

        # Generous mask that will work for all samples
        mask = mask.sum(0).sum(0) > 0
        mask = mask.flatten().cpu()

        # Get indices for patches inside the brain mask
        masked_idxs = torch.arange(flow_model.num_patches)[mask]
        # Get random patches
        shuffled_idx = torch.randperm(len(masked_idxs))
        rand_idx = masked_idxs[shuffled_idx]
        rand_idx = rand_idx[:n_patches]

        return local_patches[rand_idx], context[rand_idx]
