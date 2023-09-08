import os
import pytest
import torch
from sade.configs.ve import toy_config
from sade.datasets.filenames import get_image_files_list
from sade.data_loaders import get_dataloaders
import ml_collections


@pytest.fixture
def test_config():
    config = ml_collections.ConfigDict()
    data = config.data = ml_collections.ConfigDict()
    data.cache_rate = 0
    data.dataset = "ABCD"
    data.ood_ds = "tumor"
    data.image_size = (192, 224, 160)
    data.num_channels = 1
    data.spacing_pix_dim = 1.0
    data.dir_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "dummy_data"
    )
    data.splits_dir = data.dir_path

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 1

    config.eval = ml_collections.ConfigDict()
    config.eval.batch_size = 1

    return config


def test_dataset_image_lists(test_config):
    assert os.path.exists(test_config.data.dir_path)

    train, val, test = get_image_files_list(
        test_config.data.dataset, test_config.data.dir_path, test_config.data.splits_dir
    )

    assert train is not None
    assert val is not None
    assert test is not None


def test_dataset_shapes(test_config):
    data = test_config.data
    _, datasets = get_dataloaders(
        test_config,
        evaluation=False,
        ood_eval=False,
        num_workers=2,
    )

    assert len(datasets) == 3
    for ds in datasets:
        assert ds is not None
        x = ds[0]
        assert x is not None
        assert x["image"].dtype is torch.float32
        assert x["image"].shape == (data.num_channels, 176, 208, 160)


def test_data_loader_iter(test_config):

    dataloaders, _ = get_dataloaders(
        test_config,
        evaluation=False,
        ood_eval=False,
        num_workers=2,
    )

    assert len(dataloaders) == 3

    for dl in dataloaders:
        assert dl is not None
        x = next(iter(dl))
        assert x is not None

def test_inf_data_loader(test_config):

    (_, eval_dl, _), _ = get_dataloaders(
        test_config,
        evaluation=True,
        ood_eval=False,
        num_workers=2,
        infinite_sampler=True,
    )

    assert eval_dl is not None
    dl_iter = iter(eval_dl)
    # Test data was only one batch
    # Infinite sampler should loop
    for _ in range(3):
        x = next(dl_iter)
        assert x is not None

def test_ood_data_loader(test_config):

    dataloaders, _ = get_dataloaders(
        test_config,
        evaluation=True,
        ood_eval=True,
        num_workers=2,
    )

    train_dl, eval_dl, test_dl = dataloaders
    assert train_dl is None
    
    for dl in (eval_dl, test_dl):
        assert dl is not None
        x = next(iter(dl))
        assert x is not None
