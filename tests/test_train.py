from sade.models import ncsnpp3d
from sade.models import utils
from sade.configs.ve import toy_config
from sade.losses import get_sde_loss_fn
from sade.sde_lib import VESDE
import torch

CONFIG = toy_config.get_config()


def test_sde_initialization():
    CONFIG.model.sigma_min = 1e-2
    CONFIG.model.sigma_max = 3.0
    CONFIG.model.num_scales = 2

    sde = VESDE(
        sigma_min=CONFIG.model.sigma_min,
        sigma_max=CONFIG.model.sigma_max,
        N=CONFIG.model.num_scales,
    )

    assert sde is not None
    assert torch.isclose(sde.discrete_sigmas, torch.tensor([1e-2, 3.0])).all()
    assert torch.isclose(sde.noise_schedule_inverse(sde.discrete_sigmas), torch.tensor([0.0, 1.0])).all()    
    
    x = torch.zeros(1,2,2,2)
    _, std = sde.marginal_prob(x, torch.tensor(sde.T))
    assert torch.isclose(std,  torch.tensor([3.0])).all()

def test_train_step():
    CONFIG.model.sigma_min = 1e-2
    CONFIG.model.sigma_max = 1.0
    CONFIG.model.num_scales = 1

    score_model = utils.create_model(CONFIG)
    sde = VESDE(
        sigma_min=CONFIG.model.sigma_min,
        sigma_max=CONFIG.model.sigma_max,
        N=CONFIG.model.num_scales,
    )
    loss_fn = get_sde_loss_fn(
        sde, train=True, reduce_mean=True, likelihood_weighting=False, amp=False
    )
    N, C, H, W, D = 1, CONFIG.data.num_channels, *CONFIG.data.image_size
    x = torch.zeros(N, C, H, W, D)

    with torch.no_grad():
        loss = loss_fn(score_model, x)
    
    assert loss.shape == torch.Size([])
    assert not loss.isnan()
    expected_loss = (x + sde.sigma_max).sum()
    assert (loss < expected_loss)