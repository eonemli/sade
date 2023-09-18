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

    summary(flownet, depth=0, verbose=2)

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


    run_dir = get_flow_rundir(config, workdir)
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Saving checkpoints to {run_dir}")

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

    losses = []
    batch_sz = config.training.batch_size
    total_iters = 10  # kimg * 1000 // batch_sz + 1
    progbar = tqdm(range(total_iters))
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


def flow_evaluator(config, workdir):
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
    flownet = registry.create_flow(config).eval().requires_grad_(False)

    flow_path = get_flow_rundir(config, workdir)
    ckpt_path = f"{flow_path}/checkpoint.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{flow_path}/checkpoint-meta.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Could not find checkpoint or checkpoint-meta at {ckpt_path}"
        )
    else:
        logging.info(f"Found checkpoint at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    _ = state_dict["model_state_dict"].pop("position_encoder.cached_penc", None)
    flownet.load_state_dict(state_dict["model_state_dict"], strict=True)
    logging.info(
        f"Loaded flow model at iter= {state_dict['kimg']}, val_loss= {state_dict['val_loss']}"
    )

    # Load datasets
    # Build data iterators
    dataloaders, _ = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
        num_workers=2,
        infinite_sampler=False,
    )

    _, inlier_dl, ood_dl = dataloaders

    # Get negative log-likelihoods

    x_inlier_nlls = []
    for x in tqdm(inlier_dl):
        x = x["image"].to(config.device)
        h = scorer(x)
        x.to("cpu")
        z = -flownet.log_density(h).cpu()
        h.to("cpu")
        del h
        x_inlier_nlls.append(z)

    x_ood_nlls = []
    for x in tqdm(ood_dl):
        x = x["image"].to(config.device)
        h = scorer(x)
        x.to("cpu")
        z = -flownet.log_density(h).cpu()
        h.to("cpu")
        del h
        x_ood_nlls.append(z)

    x_inlier_nlls = torch.cat(x_inlier_nlls).numpy()
    x_ood_nlls = torch.cat(x_ood_nlls).numpy()

    np.savez_compressed(
        f"{flow_path}/anomaly_scores.npz",
        **{"inliers": x_inlier_nlls, "lesions": x_ood_nlls},
    )

    return
