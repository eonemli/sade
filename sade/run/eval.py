import logging
import os
import numpy as np

import torch
from tqdm import tqdm
from datasets.loaders import get_dataloaders
import models.registry as registry
from sade.ood_detection_helper import auxiliary_model_analysis

from .utils import restore_pretrained_weights


def evaluator(config, workdir):
    """Runs the evaluation pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

   
    # Initialize model.
    score_model = registry.create_model(config, print_summary=True)
    sde = registry.create_sde(config)

    # Load checkpoint
    state = dict(model=score_model, step=0)
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    if config.eval.checkpoint_num is not None:
        checkpoint_num = config.eval.checkpoint_num
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num}.pth")
    state = restore_pretrained_weights(checkpoint_path, state, config.device)
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(config, score_model, return_norm=True)

    # Create save directory
    save_dir = os.path.join(workdir, "eval", f"ckpt_{checkpoint_num}")
    experiment_name = "{config.data.dataset.lower()}_{config.data.ood_ds.lower()}"
    os.makedirs(save_dir, exist_ok=True)

    # Build data iterators

    test_dataloaders, datasets = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
        num_workers=4,
        infinite_sampler=False,
    )

    _, inlier_dl, ood_dl = test_dataloaders

    #! Fixme: This should be part of the config
    # Maybe make a ood_experiment config which specifies inlier and ood datasets
    config.data.dataset = "ABCD"
    dataloaders, datasets = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=False,
        num_workers=4,
        infinite_sampler=False,
    )

    _, eval_dl, _ = dataloaders

    logging.info(f"Evaluating model at checkpoint {checkpoint_num:d}...")

    # Run score norm evaluation
    x_eval_scores = []
    x_inlier_scores = []
    x_ood_scores = []

    for res_arr, ds in zip(
        [x_eval_scores, x_inlier_scores, x_ood_scores], [eval_dl, inlier_dl, ood_dl]
    ):
        ds_iter = iter(ds)
        for batch in tqdm(ds_iter):
            x = batch["image"].to(config.device)
            scores = scorer(x).cpu()
            res_arr.append(scores)

    x_eval_scores = torch.cat(x_eval_scores).numpy()
    x_inlier_scores = torch.cat(x_inlier_scores).numpy()
    x_ood_scores = torch.cat(x_ood_scores).numpy()

    # Run ood detection

    results = auxiliary_model_analysis(
        x_eval_scores,
        x_inlier_scores,
        [x_ood_scores],
        components_range=range(3, 6, 1),
        labels=["Train", "Inlier", "OOD"],
        verbose=0,
    )

    experiment_dict = {
        "eval_score_norms": x_eval_scores,
        "inlier_score_norms": x_inlier_scores,
        "ood_score_norms": x_ood_scores,
        "results": results,
    }

    np.savez_compressed(
        f"{save_dir}/{experiment_name}_results.npz",
        **experiment_dict,
    )
