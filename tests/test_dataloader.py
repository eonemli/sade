import os
from sade.configs.ve import toy_config
from sade.datasets.filenames import get_image_files_list
# from sade.data_loaders import get_dataloaders
# import torch

CONFIG = toy_config.get_config()


def test_dataset_image_lists():
    data = CONFIG.data

    data.cache_rate = 0
    data.dataset = "ABCD"
    data.image_size = (192, 224, 160)
    data.num_channels = 1
    data.spacing_pix_dim = 1.0
    data.dir_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "dummy_data"
    )

    assert os.path.exists(CONFIG.data.dir_path)

    train, val, test = get_image_files_list(
        CONFIG.data.dataset, CONFIG.data.dir_path, CONFIG.data.dir_path
    )

    assert train is not None
    assert val is not None
    assert test is not None


# def test_dataset_shapes():
#     data = CONFIG.data

#     data.cache_rate = 0
#     data.dataset = "ABCD"
#     data.image_size = (192, 224, 160)
#     data.spacing_pix_dim = 1.0
#     data.dir_path = "/DATA/Users/amahmood/braintyp/processed_v2/"

#     _, datasets = get_dataloaders(
#         CONFIG,
#         evaluation=False,
#         ood_eval=False,
#         num_workers=2,
#     )

#     assert len(datasets) == 3
#     for ds in datasets:
#         assert ds is not None
#         x = ds[0]
#         assert x is not None
#         assert x.shape == data.image_size


# def test_data_loader_iter():
#     CONFIG.data.dataset = "ABCD"
#     CONFIG.data.cache_rate = 0

#     dataloaders, _ = get_dataloaders(
#         CONFIG,
#         evaluation=False,
#         ood_eval=False,
#         num_workers=2,
#     )

#     assert len(dataloaders) == 3

#     for dl in dataloaders:
#         assert dl is not None
#         x = next(iter(dl))
#         assert x is not None
