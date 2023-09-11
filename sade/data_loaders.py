"""Return training and evaluation/test datasets from config files."""
import functools
import os

import ants
import numpy as np
import torch
from datasets.filenames import get_image_files_list
from datasets.transforms import (
    get_lesion_transform,
    get_train_transform,
    get_tumor_transform,
    get_val_transform,
)
from monai.data import CacheDataset
from monai.transforms import *
from torch.utils.data import DataLoader, RandomSampler

def get_data_inverse_scaler(config):
    # """Inverse data normalizer."""
    # if config.data.centered:
    #     # Rescale [-1, 1] to [0, 1]
    #     return lambda x: (x + 1.0) / 2.0
    # else:

    return lambda x: x


def get_dataloaders(
    config,
    evaluation=False,
    ood_eval=False,
    num_workers=6,
    infinite_sampler=False,
):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      evaluation: If `True`, only val_transform will be used

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    if infinite_sampler:
        inf_sampler = functools.partial(
            RandomSampler, replacement=True, num_samples=int(1e100)
        )

    # Sanity checks
    assert config.data.dataset.lower() in ["abcd", "ibis"]
    assert config.data.ood_ds.lower() in ["tumor", "lesion", "ds-sa"]

    # Directory that holds files with train/test splits and other filenames
    splits_dir = config.data.splits_dir
    # Directory that holds the data samples
    data_dir_path = config.data.dir_path
    cache_rate = config.data.cache_rate
    dataset_name = config.data.dataset.lower()

    train_file_list = None
    val_file_list = None
    test_file_list = None

    train_transform = get_train_transform(config)
    val_transform = get_val_transform(config)
    test_transform = get_val_transform(config)

    if ood_eval:
        # We will only return inlier and ood samples
        train_file_list = None
        ood_dataset_name = config.data.ood_ds.lower()
        if "lesion" in ood_dataset_name:
            dirname = f"slicer_lesions/{dataset_name}"
            data_dir_path = os.path.realpath(os.path.join(data_dir_path, "..", dirname))

            # Getting lesion samples
            _, _, test_file_list = get_image_files_list(
                ood_dataset_name, data_dir_path, splits_dir
            )
            # lesions will be loaded alongside label masks
            test_transform = get_lesion_transform(config)
        elif ood_dataset_name == "tumor":
            _, _, test_file_list = get_image_files_list(
                dataset_name, data_dir_path, splits_dir
            )
            test_transform = get_tumor_transform(config)
        else:  # image-only ood dataset
            _, _, test_file_list = get_image_files_list(
                dataset_name, data_dir_path, splits_dir
            )

        # Inlier samples
        _, val_file_list, _ = get_image_files_list(dataset_name, data_dir_path, splits_dir)
    elif evaluation:
        _, val_file_list, test_file_list = get_image_files_list(
            dataset_name, data_dir_path, splits_dir
        )
    else:  # Training data
        train_file_list, val_file_list, test_file_list = get_image_files_list(
            dataset_name, data_dir_path, splits_dir
        )

    train_ds = None
    if train_file_list is not None:
        train_ds = CacheDataset(
            train_file_list,
            transform=train_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=True,
        )

    eval_ds = CacheDataset(
        val_file_list,
        transform=val_transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    # Note: Will be an ood dataset if ood_eval==True
    test_ds = CacheDataset(
        test_file_list,
        transform=test_transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    # Get data loaders
    train_loader = None
    if train_ds:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=num_workers > 0,
            sampler=inf_sampler(train_ds) if infinite_sampler else None,
        )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        sampler=inf_sampler(eval_ds) if infinite_sampler else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        sampler=inf_sampler(test_ds) if infinite_sampler else None,
    )

    dataloaders = (train_loader, eval_loader, test_loader)
    datasets = (train_ds, eval_ds, test_ds)

    return dataloaders, datasets
