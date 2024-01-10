import glob
import logging
import os
import re

import models.registry as registry
import numpy as np
import torch
from datasets.loaders import get_dataloaders
from tqdm import tqdm

from sade.metrics import (
    compute_segmentation_metrics,
    erode_brain_masks,
    get_best_thresholds,
    post_processing,
)
from sade.models.ema import ExponentialMovingAverage
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
    state = dict(
        model=score_model,
        step=0,
        ema=ExponentialMovingAverage(score_model.parameters(), decay=0),
        model_checkpoint_step=0,
    )
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    if config.eval.checkpoint_num > 0:
        checkpoint_num = config.eval.checkpoint_num
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num}.pth")
    else:
        # Get the latest checkpoint
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
        if len(checkpoint_paths) > 0:
            checkpoint_path = max(
                checkpoint_paths, key=lambda x: int(re.search(r"_(\d+)\.pth", x).group(1))
            )
        else:  # Try to check for checkpoint-meta
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-meta.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Could not find checkpoint or checkpoint-meta at {os.path.dirname(checkpoint_path)}"
                )

    state = restore_pretrained_weights(checkpoint_path, state, config.device)
    checkpoint_step = state["model_checkpoint_step"]
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(config, score_model, return_norm=True)
    logging.info(f"Evaluating model at checkpoint {checkpoint_step:d}...")

    # Create save directory
    save_dir = os.path.join(workdir, "eval", f"ckpt_{checkpoint_step}")
    experiment = config.eval.experiment
    experiment_name = f"{experiment.train}_{experiment.inlier}_{experiment.ood}"
    # experiment_name += config.eval.experiment.id
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving evaluation results to {save_dir}")

    # Build data iterators

    test_dataloaders, datasets = get_dataloaders(
        config,
        evaluation=True,
        num_workers=4,
        infinite_sampler=False,
    )

    eval_dl, inlier_dl, ood_dl = test_dataloaders

    # Run score norm evaluation
    logging.info(f"Running experiment: {experiment_name}")
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

    # Print results
    print(results["GMM"]["metrics"])


def segmentation_evaluator(config, workdir):
    config.device = torch.device("cpu")
    experiment = config.eval.experiment
    experiment_name = f"{experiment.inlier}_{experiment.ood}"
    scores_path = f"{workdir}/{experiment_name}_results.npz"
    assert os.path.exists(scores_path), f"Scores not found at {scores_path}"
    data = np.load(scores_path, allow_pickle=True)
    x_ood_scores = data["ood"]

    if "-enhanced" in experiment.ood:
        experiment.ood = experiment.ood.split("-")[0]

    (_, inlier_ds, ood_ds), _ = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
    )

    x_ood = []
    x_ood_labels = []
    for x in ood_ds:
        x_ood.append(x["image"])
        x_ood_labels.append(x["label"])

    x_ood = torch.cat(x_ood)
    x_ood_labels = torch.cat(x_ood_labels)
    ood_brain_masks = (x_ood != -1.0).sum(dim=1).bool()

    ### Run the segmentation evaluation pipeline

    # Ensure same spatial size
    anomaly_scores = (
        torch.nn.functional.interpolate(
            torch.from_numpy(x_ood_scores).unsqueeze(1), size=x_ood.shape[2:]
        )
        .squeeze(1)
        .numpy()
    )
    anomaly_scores = anomaly_scores * ood_brain_masks
    true_labels = x_ood_labels.sum(1).cpu().numpy() > 0
    true_labels_masked = true_labels * ood_brain_masks

    # Computing Hausdorff distance to determine bets thresholds for segmentation
    best_seg_thresholds, _ = get_best_thresholds(
        anomaly_scores, true_labels_masked, ood_brain_masks, min_component_size=3
    )

    post_proc_preds = []
    post_proc_labels = []

    eroded_brain_masks = erode_brain_masks(ood_brain_masks)
    brain_mask_rims = (ood_brain_masks * (~eroded_brain_masks)).astype(float)
    for sample_idx, thresh in enumerate(best_seg_thresholds):
        skull_mask = brain_mask_rims[sample_idx]
        pred_scores = anomaly_scores[sample_idx]
        best_thresh = best_seg_thresholds[sample_idx]
        pred = post_processing(
            pred_scores > best_thresh, skull_mask, dilate=True, min_component_size=3
        )
        ref_mask = post_processing(
            true_labels[sample_idx], skull_mask, min_component_size=3
        )
        post_proc_preds.append(pred)
        post_proc_labels.append(ref_mask)

    # Save the predictions
    post_proc_preds = np.stack(post_proc_preds)
    post_proc_labels = np.stack(post_proc_labels)
    np.savez_compressed(
        f"{workdir}/{experiment_name}_segs.npz",
        **{"preds": post_proc_preds, "labels": post_proc_labels},
    )

    metrics_df = compute_segmentation_metrics(post_proc_preds, post_proc_labels)
    metrics_df.to_csv(f"{workdir}/{experiment_name}_seg_eval.csv")
    print(metrics_df.dropna().describe())
