"""All functions related to loss computation and optimization.
"""
#######
## TODO: Move optimizer and scheduler and step_fn to sade/optim.py
######
import torch
import torch.optim as optim
import numpy as np
from sade.models.registry import get_score_fn

avail_optimizers = {
    "Adam": optim.Adam,
    "Adamax": optim.Adamax,
    "AdamW": optim.AdamW,
    "RAdam": optim.RAdam,
}


def get_optimizer(config, params):
    """Returns an optimizer object based on `config`."""
    if config.optim.optimizer in avail_optimizers:
        opt = avail_optimizers[config.optim.optimizer]

        optimizer = opt(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    return optimizer


def get_scheduler(config, optimizer):
    """Returns a scheduler object based on `config`."""

    if config.optim.scheduler == "skip":
        scheduler = None

    if config.optim.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(0.3 * config.training.n_iters),
            gamma=0.3,
            verbose=False,
        )

    if config.optim.scheduler == "cosine":
        # Assumes LR in opt is initial learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.n_iters,
            eta_min=1e-6,
        )

    print("Using scheduler:", scheduler)
    return scheduler


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(
        optimizer,
        params,
        step,
        scheduler=None,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
        amp_scaler=None,
    ):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if step <= warmup:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            if amp_scaler is not None:
                amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        if amp_scaler is not None:
            amp_scaler.step(optimizer)
            # amp_scaler.update(
            #     new_scale=2.0**8 if amp_scaler.get_scale() > 2.0**10 else None
            # )
            amp_scaler.update()
        else:
            optimizer.step()

        if step > warmup and scheduler is not None:
            scheduler.step()

    return optimize_fn

def get_sde_loss_fn(
    sde,
    train,
    reduce_mean=True,
    likelihood_weighting=False,
    amp=False,
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.

    Returns:
      A loss function.
    """

    # For numerical stability - the smallest time step to sample from.
    eps=1e-5
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )
    def loss_fn(model, batch, log_alpha=False):
        """Compute the loss function.

        Args:
        model: A score model.
        batch: A mini-batch of training data.

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = get_score_fn(
            sde, model, train=train, amp=amp
        )
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + sde._unsqueeze(std) * z
        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * sde._unsqueeze(std) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / sde._unsqueeze(std))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn

def get_step_fn(
    sde,
    train,
    optimize_fn=None,
    reduce_mean=False,
    likelihood_weighting=False,
    scheduler=None,
    use_fp16=False,
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """

    loss_fn = get_sde_loss_fn(
        sde,
        train,
        reduce_mean=reduce_mean,
        likelihood_weighting=likelihood_weighting,
        amp=use_fp16,
    )

    if use_fp16:
        print(f"Using AMP for {'training' if train else 'evaluation'}.")

        def step_fn(state, batch):
            """Running one step of training or evaluation with AMP"""
            model = state["model"]
            if train:
                optimizer = state["optimizer"]
                loss_scaler = state["grad_scaler"]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch)

                loss_scaler.scale(loss).backward()
                optimize_fn(
                    optimizer,
                    model.parameters(),
                    step=state["step"],
                    scheduler=scheduler,
                    amp_scaler=loss_scaler,
                )
                state["step"] += 1
                if not torch.isnan(loss):
                    state["ema"].update(model.parameters())
            else:
                with torch.inference_mode():
                    loss = loss_fn(model, batch)

            return loss

    else:

        def step_fn(state, batch):
            """Running one step of training or evaluation.

            Args:
            state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.

            Returns:
            loss: The average loss value of this state.
            """
            model = state["model"]
            if train:
                optimizer = state["optimizer"]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch)
                loss.backward()

                optimize_fn(
                    optimizer,
                    model.parameters(),
                    step=state["step"],
                    scheduler=scheduler,
                    # amp_scaler=loss_scaler,
                )
                state["step"] += 1
                state["ema"].update(model.parameters())
            else:
                with torch.no_grad():
                    ema = state["ema"]
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    loss = loss_fn(model, batch)
                    ema.restore(model.parameters())

            return loss

    return step_fn


def get_scorer(sde, continuous=True, eps=1e-5):
    def scorer(model, batch):
        """Compute the weighted scores function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          score: A tensor that represents the weighted scores for each sample in the mini-batch.
        """
        score_fn = get_score_fn(sde, model, train=False, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        score = score_fn(batch, t)
        return score

    return scorer


def score_step_fn(sde, continuous=True, eps=1e-5):
    scorer = get_scorer(
        sde,
        continuous=continuous,
    )

    def step_fn(state, batch):
        """Running one step of scoring

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          score: The average loss value of this state.
        """
        # FIXME!!!! I was restoring original params back
        model = state["model"]
        ema = state["ema"]
        ema.copy_to(model.parameters())
        with torch.no_grad():
            score = scorer(model, batch)
        return score

    return step_fn


def get_diagnsotic_fn(
    sde,
    reduce_mean=False,
    likelihood_weighting=False,
    eps=1e-5,
    steps=5,
    use_fp16=False,
):
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def sde_loss_fn(model, batch, t):
        """Compute the per-sigma loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = get_score_fn(
            sde, model, train=False, amp=use_fp16
        )
        _t = torch.ones(batch.shape[0], device=batch.device) * t * (sde.T - eps) + eps

        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, _t)
        perturbed_data = mean + sde._unsqueeze(std) * z

        score = score_fn(perturbed_data, _t)
        score_norms = torch.linalg.norm(score.reshape((score.shape[0], -1)), dim=-1)
        score_norms = score_norms * std

        if not likelihood_weighting:
            losses = torch.square(score * sde._unsqueeze(std) + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), _t)[1] ** 2
            losses = torch.square(score + z / sde._unsqueeze(std))
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)

        return loss, score_norms

    final_timepoint = 1.0
    loss_fn = sde_loss_fn

    def step_fn(state, batch):
        model = state["model"]
        with torch.no_grad():
            # ema = state["ema"]
            # ema.store(model.parameters())
            # ema.copy_to(model.parameters())

            losses = {}

            for t in torch.linspace(0.0, final_timepoint, steps, dtype=torch.float32):
                loss, norms = loss_fn(model, batch, t)
                losses[f"{t:.3f}"] = (loss.item(), norms.cpu())

            # ema.restore(model.parameters())

        return losses

    return step_fn
