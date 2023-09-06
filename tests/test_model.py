from sade.models import ncsnpp3d
from sade.models import utils
from sade.configs.ve import toy_config
import torch

CONFIG = toy_config.get_config()


def test_model_initialization():
    CONFIG.model.blocks_down = (1, 2)
    CONFIG.model.blocks_up = (1,)
    score_model = utils.create_model(CONFIG)
    assert score_model is not None


def test_output_shape():
    CONFIG.data.num_channels = 7
    CONFIG.data.image_size = (16, 8, 16)

    score_model = utils.create_model(CONFIG)
    N, C, H, W, D = 3, CONFIG.data.num_channels, *CONFIG.data.image_size
    x = torch.ones(N, C, H, W, D)
    y = torch.tensor([1] * N)

    with torch.no_grad():
        score = score_model(x, y)

    assert score.shape == (N, C, H, W, D)
