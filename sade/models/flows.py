from . import registry
from functools import partial
import logging
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from sade.models.layers import PositionalEncoding3D, SpatialNorm3D
from sade.models.layerspp import get_conv_layer, FlowAttentionBlock


def gaussian_logprob(z, ldj):
    _GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))
    return _GCONST_ - 0.5 * torch.sum(z**2, dim=-1) + ldj


def subnet_fc(c_in, c_out, ndim=256, act=nn.LeakyReLU(), input_norm=True):
    return nn.Sequential(
        nn.LayerNorm(c_in) if input_norm else nn.Identity(),
        nn.Linear(c_in, ndim),
        act,
        nn.LayerNorm(ndim),
        nn.Linear(ndim, c_out),
        act,
        # nn.Linear(ndim, c_out),
    )


@registry.register_model(name="flow3d")
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
            self.norm_pooler = SpatialNorm3D(
                channels, **self.global_patch_config
            ).requires_grad_(False)
            self.conv_pooler = get_conv_layer(
                3, channels, channels, kernel_size=3, stride=2
            )
            self.global_pooler = nn.Sequential(
                self.norm_pooler,
                self.conv_pooler,
            )
            # Spatial resolution of the global context patches
            _, c, h, w, d = self.global_pooler(torch.empty(1, *input_size)).shape
            logging.info("Global Context Shape: ", (c, h, w))
            self.global_attention = FlowAttentionBlock(
                input_size=(c, h, w, d),
                embed_dim=self.global_embedding_size,
                outdim=self.context_embedding_size,
            )
            context_dims += self.context_embedding_size

        num_features = self.channels
        self.flow = self.build_cflow_head(num_features, context_dims, self.num_blocks)
        self.to(self.device)

    def init_weights(self):
        # Initialize weights with Xavier
        for m in self.flow.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print(m)
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

    def build_cflow_head(self, input_dim, conditioning_dim, num_blocks=2):
        coder = Ff.SequenceINN(input_dim)
        for k in range(num_blocks):
            # idx = int(k % 2 == 0)
            coder.append(
                Fm.AllInOneBlock,
                cond=0,
                cond_shape=(conditioning_dim,),
                subnet_constructor=partial(subnet_fc, act=nn.GELU()),
                global_affine_type="SOFTPLUS",
                permute_soft=True,
                affine_clamping=1.9,
            )

        return coder

    def forward(self, x, return_attn=False, fast=True):
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

                z, ldj = self.flow(
                    patch_feature,
                    c=[c],
                )
                zs.append(z)
                log_jac_dets.append(ldj)
                c = c.cpu()

        zs = torch.cat(zs, dim=0).reshape(self.num_patches, B, C)
        log_jac_dets = torch.cat(log_jac_dets, dim=0).reshape(self.num_patches, B)

        if return_attn:
            return zs, log_jac_dets

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

        for p, ctx in zip(patches, ctx_chunks):
            # Check that patch context is same for all batch elements
            #             assert torch.isclose(c[0, :32], c[B-1, :32]).all()
            #             assert torch.isclose(c[B+1, :32], c[(2*B)-1, :32]).all()
            ctx = ctx.to(self.device)
            gc = repeat(global_ctx, "b c -> (n b) c", n=ctx.shape[0])
            ctx = rearrange(ctx, "n b c -> (n b) c")
            p = rearrange(p, "n b c -> (n b) c")

            c = torch.cat([ctx, gc], dim=1)
            z, ldj = self.flow(p, c=[c])

            zs.append(z)
            jacs.append(ldj)

            ctx = ctx.cpu()

        return zs, jacs

    def logprob(self, zs, log_jac_dets):
        return gaussian_logprob(zs, log_jac_dets)

    def nll(self, zs, log_jac_dets):
        return -torch.mean(self.logprob(zs, log_jac_dets))

    @torch.no_grad()
    def log_density(self, x, fast=True):
        self.eval()
        b = x.shape[0]
        h, w, d = self.spatial_res
        zs, jacs = self.forward(x, fast=fast)
        logpx = self.logprob(zs, jacs)
        logpx = rearrange(logpx, "(h w d) b -> b h w d", b=b, h=h, w=w, d=d)

        return logpx

    @staticmethod
    def stochastic_train_step(flow, x, opt, n_patches=1):
        flow.train()
        B, C, _, _, _ = x.shape
        h = flow.local_pooler(x)
        local_patches = rearrange(h, "b c h w d -> (h w d) b c")

        flow.position_encoder = flow.position_encoder.cpu()
        context = rearrange(flow.position_encoder(h), "b c h w d -> (h w d) b c")

        rand_idx = torch.randperm(flow.num_patches)[:n_patches]
        local_loss = 0.0
        for idx in rand_idx:
            patch_feature, context_vector = (
                local_patches[idx],
                context[idx],
            )
            context_vector = context_vector.to(self.device)
            if flow.use_global_context:
                # Need separate loss for each patch
                global_pooled_image = flow.global_pooler(x)
                global_context = flow.global_attention(global_pooled_image)
                # Concatenate global context to local context
                context_vector = torch.cat([context_vector, global_context], dim=1)

            z, ldj = flow.flow(
                patch_feature,
                c=[context_vector],
            )
            if flow.gmm is not None:
                z = flow.gmm(z, context_vector[:, : flow.context_embedding_size])

            opt.zero_grad(set_to_none=True)
            loss = flow.nll(z, ldj)
            loss.backward()

            opt.step()
            local_loss += loss.item()

        return {"train_loss": local_loss / n_patches}
