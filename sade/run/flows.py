import copy
import functools
import glob
import os
import sys
import numpy as np

import torch
from torch.utils import tensorboard
from torchinfo import summary
from tqdm import tqdm
from sade.datasets.loaders import get_dataloaders

from sade.models.flows import PatchFlow
import models.registry as registry
from sade.models.ema import ExponentialMovingAverage
from sade.run.utils import get_flow_rundir, restore_checkpoint, restore_pretrained_weights

import logging


def flow_trainer(config, workdir):
    kimg = config.flow.training_kimg
    log_tensorboard = config.flow.log_tensorboard
    lr = config.flow.lr
    log_interval = config.training.log_freq
    device = config.device
    ema_halflife_kimg = config.flow.ema_halflife_kimg
    ema_rampup_ratio = config.flow.ema_rampup_ratio

    # Initialize score model
    score_model = registry.create_model(config, log_grads=False)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    # Get the score model checkpoint from pretrained run
    checkpoint_paths = os.path.join(config.training.pretrain_dir, "checkpoint.pth")
    # latest_checkpoint_path = max(checkpoint_paths, key=lambda x: int(x.split("_")[-1][1]))
    # state = restore_checkpoint(latest_checkpoint_path, state, config.device)
    state = restore_pretrained_weights(checkpoint_paths, state, config.device)
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(config, score_model, return_norm=False)

    # Initialize flow model
    flownet = registry.create_flow(config)

    summary(flownet, depth=1, verbose=2)

    # Build data iterators
    dataloaders, _ = get_dataloaders(
        config,
        evaluation=False,
        ood_eval=False,
        num_workers=2,
        infinite_sampler=True,
    )

    train_dl, eval_dl, _ = dataloaders
    train_iter = iter(train_dl)
    eval_iter = iter(eval_dl)

    losses = []
    batch_sz = config.training.batch_size
    total_iters = 10  # kimg * 1000 // batch_sz + 1
    progbar = tqdm(range(total_iters))

    run_dir = get_flow_rundir(config, workdir)
    os.makedirs(run_dir, exist_ok=True)

    flow_checkpoint_path = f"{run_dir}/checkpoint.pth"
    flow_checkpoint_meta_path = f"{run_dir}/checkpoint-meta.pth"

    # Set logger so that it outputs to both console and file
    gfile_stream = open(os.path.join(run_dir, "stdout.txt"), "w")
    file_handler = logging.StreamHandler(gfile_stream)
    stdout_handler = logging.StreamHandler(sys.stdout)

    # TODO: RESUME CHECKPPOINT

    # Override root handler
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
        handlers=[file_handler, stdout_handler],
    )

    if log_tensorboard:
        writer = tensorboard.SummaryWriter(log_dir=run_dir)

    flownet = flownet.to(device)
    # Main copy to be used for "fast" weight updates
    teacher_flow_model = copy.deepcopy(flownet).to(device)
    # Model will be updated with EMA weights
    flownet = flownet.eval().requires_grad_(False)

    # Defining optimization step
    opt = torch.optim.AdamW(teacher_flow_model.parameters(), lr=lr, weight_decay=1e-5)
    flow_train_step = functools.partial(
        PatchFlow.stochastic_train_step,
        flow_model=teacher_flow_model,
        opt=opt,
        n_patches=config.flow.patches_per_train_step,
    )

    niter = 0
    imgcount = 0
    best_val_loss = np.inf
    checkpoint_interval = 10

    for niter in progbar:
        x_batch = next(train_iter)["image"].to(device)
        scores = scorer(x_batch)

        loss_dict = flow_train_step(scores, x_batch)
        imgcount += x_batch.shape[0]

        # Ramp up EMA beta
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, imgcount * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_sz / max(ema_halflife_nimg, 1e-8))
        writer.add_scalar("ema_beta", ema_beta, niter)

        # Update original model with EMA weights
        for p_ema, p_net in zip(flownet.parameters(), teacher_flow_model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        if log_tensorboard:
            for loss_type in loss_dict:
                writer.add_scalar(f"train_loss/{loss_type}", loss_dict[loss_type], niter)

        if niter % log_interval == 0:
            torch.cuda.empty_cache()
            flownet.eval()

            with torch.no_grad():
                val_loss = 0.0
                x = next(eval_iter)["image"].to(device)
                x = scorer(x)
                z, log_jac_det = flownet(x)
                val_loss = flownet.nll(z, log_jac_det).item()

            progbar.set_description(f"Val Loss: {val_loss:.4f}")
            if log_tensorboard:
                writer.add_scalar("val_loss", val_loss, niter)
            losses.append(val_loss)

        progbar.set_postfix(batch=f"{imgcount}/{kimg}K")

        if niter % checkpoint_interval == 0 and val_loss < best_val_loss:
            # if the current validation loss is the best one
            best_val_loss = val_loss  # Update the best validation loss
            torch.save(
                {
                    "kimg": niter,
                    "model_state_dict": flownet.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss,
                },
                flow_checkpoint_meta_path,
            )

        # progbar.set_description(f"Loss: {loss:.4f}")
    if val_loss < best_val_loss:
        torch.save(
            {
                "epoch": -1,
                "kimg": niter,
                "model_state_dict": flownet.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_loss": val_loss,
            },
            flow_checkpoint_path,
        )
    else:  # Rename checkpoint_meta to checkpoint
        os.rename(flow_checkpoint_meta_path, flow_checkpoint_path)

    if log_tensorboard:
        writer.close()

    return losses
