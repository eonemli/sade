import os
import sys

import numpy as np
import torch
from captum.attr import GuidedBackprop, LayerGradCam
from captum.attr._utils.attribution import LayerAttribution
from tqdm import tqdm

from sade.configs.ve import biggan_config
from sade.datasets.loaders import get_dataloaders
from sade.models import registry
from sade.msma.flow_model import FlowModel, build_nd_flow
from sade.run.utils import restore_pretrained_weights


class ScoreFlow(torch.nn.Module):
    def __init__(self, net, flow):
        super().__init__()
        self.net = net
        self.flow = flow

    def forward(self, x, t):
        b = x.shape[0]
        out = self.net(x, t).view(1, b, -1)
        out = torch.linalg.norm(out, dim=-1, keepdim=False)
        out = -1 * self.flow(out)
        return out


def run(config, workdir):
    # Initialize main model.
    config.training.use_fp16 = False
    n_timesteps = config.msma.n_timesteps = 10
    score_model = registry.create_model(config, distributed=False)
    timesteps = registry.get_msma_sigmas(config)

    state = dict(model=score_model, step=0)
    pretrain_dir = os.path.join(config.training.pretrain_dir, "checkpoint.pth")
    state = restore_pretrained_weights(pretrain_dir, state, config.device)
    score_model = score_model.eval()

    flow_model = FlowModel(config.msma.n_timesteps, device=config.device)
    flow_model = flow_model.eval()
    flow_state = torch.load(f"{workdir}/msma/checkpoint_10.pth")
    flow_model.load_state_dict(flow_state["model_state_dict"], strict=True)
    scoreflow = ScoreFlow(score_model, flow_model)

    gb = GuidedBackprop(scoreflow)
    stop_layer = scoreflow.net.up_layers[-1]
    layer_gc = LayerGradCam(scoreflow, stop_layer)

    # Build data iterator
    config.eval.batch_size = 1
    _, datasets = get_dataloaders(
        config,
        evaluation=True,
        num_workers=1,
        infinite_sampler=False,
    )

    _, inlier_ds, ood_ds = datasets

    inlier_ds = torch.utils.data.Subset(inlier_ds, list(range(4)))
    ood_ds = torch.utils.data.Subset(ood_ds, list(range(4)))

    x_inlier_attrs = []
    x_ood_attrs = []

    for res_arr, ds in zip([x_inlier_attrs, x_ood_attrs], [inlier_ds, ood_ds]):
        ds_iter = tqdm(iter(ds))
        for x in ds_iter:
            x = x["image"].cuda().unsqueeze(0)
            x = torch.repeat_interleave(x, n_timesteps, dim=0)
            guided_backprop_attr = gb.attribute(
                x, target=0, additional_forward_args=timesteps
            )
            guided_backprop_attr = guided_backprop_attr.detach().cpu()
            x = x.detach()
            x_backprop = guided_backprop_attr.sum(0).sum(0)

            torch.cuda.empty_cache()
            grad_cam_attr, _ = layer_gc.attribute(
                (x), 0, additional_forward_args=timesteps, attribute_to_layer_input=True
            )
            x_grad_cam = grad_cam_attr.detach().cpu()
            x_grad_cam = LayerAttribution.interpolate(
                x_grad_cam,
                x.shape[2:],
            )
            x_grad_cam = x_grad_cam.sum(0)[0]
            torch.cuda.empty_cache()

            x_guided_grad_cam = (x_grad_cam * x_backprop).cpu()
            res_arr.append(x_guided_grad_cam)

    x_inlier_attrs = torch.stack(x_inlier_attrs)
    x_ood_attrs = torch.stack(x_ood_attrs)

    experiment_name = f"{experiment.inlier}_{experiment.ood}"
    np.savez_compressed(
        f"{workdir}/msma/{experiment_name}_results.npz",
        **{"ood": x_ood_attrs, "inliers": x_inlier_attrs},
    )


if __name__ == "__main__":
    from sade.configs.ve import biggan_config

    workdir = sys.argv[1]

    config = biggan_config.get_config()
    config.training.use_fp16 = True
    experiment = config.eval.experiment
    experiment.train = "abcd-val"  # The dataset used for training MSMA
    experiment.inlier = "abcd-test"
    experiment.ood = "lesion_load_20"

    run(config, workdir)
