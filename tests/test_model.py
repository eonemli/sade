from sade.models import registry
from sade.configs.ve import toy_config
import torch

CONFIG = toy_config.get_config()


def test_model_initialization():
    CONFIG.model.blocks_down = (1, 2)
    CONFIG.model.blocks_up = (1,)
    score_model = registry.create_model(CONFIG)
    assert score_model is not None


def test_model_output_shape():
    CONFIG.data.num_channels = 7
    CONFIG.data.image_size = (16, 8, 16)

    score_model = registry.create_model(CONFIG)
    score_fn = registry.get_model_fn(score_model, train=False)
    N, C, H, W, D = 3, CONFIG.data.num_channels, *CONFIG.data.image_size
    x = torch.ones(N, C, H, W, D)
    y = torch.tensor([1] * N)

    with torch.no_grad():
        score = score_fn(x, y)

    assert not torch.isnan(score).any()
    assert score.shape == (N, C, H, W, D)


def test_mixed_precision():
    score_model = registry.create_model(CONFIG)
    score_fn = registry.get_model_fn(score_model, amp=True, train=False)
    N, C, H, W, D = 1, CONFIG.data.num_channels, *CONFIG.data.image_size
    x = torch.ones(N, C, H, W, D)
    y = torch.tensor([1] * N)

    with torch.no_grad():
        score = score_fn(x, y)
    
    if CONFIG.device == torch.device("cpu"):
        assert score.dtype == torch.float32
    else:
        assert score.dtype == torch.float16
