import os

import ml_collections
import pytest
import torch

from sade.models import registry


@pytest.fixture
def test_config():
    config = ml_collections.ConfigDict()
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    data = config.data = ml_collections.ConfigDict()
    data.cache_rate = 0
    data.dataset = "ABCD"
    data.ood_ds = "tumor"
    data.image_size = (48, 64, 48)
    data.spacing_pix_dim = 4.0
    data.num_channels = 1
    data.dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dummy_data")
    data.splits_dir = data.dir_path

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 1
    config.training.sde = "vesde"
    config.training.use_fp16 = False

    config.eval = ml_collections.ConfigDict()
    config.eval.batch_size = 1

    model = config.model = ml_collections.ConfigDict()
    # for sde
    model.sigma_min = 1e-2
    model.sigma_max = 1.0
    model.num_scales = 3
    # arch specifc
    model.name = "ncsnpp3d"
    model.resblock_type = "biggan"
    model.act = "memswish"
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nf = 8
    model.norm_num_groups = 2
    model.blocks_down = (1, 2, 1)
    model.blocks_up = (1, 1)
    model.time_embedding_sz = 32
    model.init_scale = 0.0
    model.conv_size = 3
    model.self_attention = False
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = False
    model.dilation = 1
    model.jit = False
    # This should not be here eventually...
    model.resblock_pp = True

    return config


def test_model_initialization(test_config):
    test_config.model.blocks_down = (1, 2)
    test_config.model.blocks_up = (1,)
    score_model = registry.create_model(test_config)
    assert score_model is not None


def test_model_output_shape(test_config):
    test_config.data.num_channels = 7
    test_config.data.image_size = (16, 8, 16)

    score_model = registry.create_model(test_config)
    score_fn = registry.get_model_fn(score_model, train=False)
    N, C, H, W, D = 3, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D)
    y = torch.tensor([1] * N)

    with torch.no_grad():
        score = score_fn(x, y)

    assert not torch.isnan(score).any()
    assert score.shape == (N, C, H, W, D)


def test_mixed_precision(test_config):
    score_model = registry.create_model(test_config)
    score_fn = registry.get_model_fn(score_model, amp=True, train=False)
    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D)
    y = torch.tensor([1] * N)

    with torch.no_grad():
        score = score_fn(x, y)

    if test_config.device == torch.device("cpu"):
        assert score.dtype == torch.float32
    else:
        assert score.dtype == torch.float16
