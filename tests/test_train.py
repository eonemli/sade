from sade import losses
from sade.models import ncsnpp3d
from sade.models.registry import create_model
from sade.losses import (
    get_optimizer,
    get_sde_loss_fn,
    optimization_manager,
    get_step_fn,
)
from sade.models.ema import ExponentialMovingAverage
from sade.sde_lib import VESDE
import torch
import ml_collections
import pytest
import os


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
    data.image_size = (16, 16, 16)
    data.num_channels = 2
    data.spacing_pix_dim = 1.0
    data.dir_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "dummy_data"
    )
    data.splits_dir = data.dir_path

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 1
    config.training.sde = "vesde"

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
    model.resblock_pp = True
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = False
    model.dilation = 1
    model.jit = False

    # optimization
    optim = config.optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.scheduler = "skip"
    optim.lr = 1e-3
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = -1
    optim.grad_clip = 1.0

    return config


def test_sde_initialization(test_config):
    test_config.model.sigma_min = 1e-2
    test_config.model.sigma_max = 3.0
    test_config.model.num_scales = 2

    sde = VESDE(
        sigma_min=test_config.model.sigma_min,
        sigma_max=test_config.model.sigma_max,
        N=test_config.model.num_scales,
    )

    assert sde is not None
    assert torch.isclose(sde.discrete_sigmas, torch.tensor([1e-2, 3.0])).all()
    assert torch.isclose(
        sde.noise_schedule_inverse(sde.discrete_sigmas), torch.tensor([0.0, 1.0])
    ).all()

    x = torch.zeros(1, 2, 2, 2)
    _, std = sde.marginal_prob(x, torch.tensor(sde.T))
    assert torch.isclose(std, torch.tensor([3.0])).all()


def test_loss_fn(test_config):
    test_config.model.sigma_min = 1e-2
    test_config.model.sigma_max = 1.0
    test_config.model.num_scales = 1

    score_model = create_model(test_config)
    sde = VESDE(
        sigma_min=test_config.model.sigma_min,
        sigma_max=test_config.model.sigma_max,
        N=test_config.model.num_scales,
    )
    loss_fn = get_sde_loss_fn(
        sde, train=True, reduce_mean=True, likelihood_weighting=False, amp=False
    )
    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.zeros(N, C, H, W, D, device=test_config.device)

    with torch.no_grad():
        loss = loss_fn(score_model, x)

    assert loss.shape == torch.Size([])
    assert not loss.isnan()
    expected_loss = (x + sde.sigma_max).sum()
    assert loss < expected_loss


def test_optimization_fn(test_config):
    test_config.model.num_scales = 1
    score_model = create_model(test_config)
    sde = VESDE(
        sigma_min=test_config.model.sigma_min,
        sigma_max=test_config.model.sigma_max,
        N=test_config.model.num_scales,
    )
    optimizer = get_optimizer(test_config, score_model.parameters())
    optimize_fn = optimization_manager(test_config)
    loss_fn = get_sde_loss_fn(
        sde, train=True, reduce_mean=False, likelihood_weighting=False, amp=False
    )
    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D, device=test_config.device)

    torch.manual_seed(42)
    pre_opt_loss = loss_fn(score_model, x)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(score_model, x)
        loss.backward()
        optimize_fn(optimizer, score_model.parameters(), step=0)
    torch.manual_seed(42)  # trying to get the same results
    post_opt_loss = loss_fn(score_model, x)
    assert post_opt_loss < pre_opt_loss


def test_train_step(test_config):
    score_model = create_model(test_config)
    sde = VESDE(
        sigma_min=test_config.model.sigma_min,
        sigma_max=test_config.model.sigma_max,
        N=test_config.model.num_scales,
    )
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=test_config.model.ema_rate
    )
    optimizer = losses.get_optimizer(test_config, score_model.parameters())
    optimize_fn = optimization_manager(test_config)

    state = dict(
        optimizer=optimizer,
        model=score_model,
        ema=ema,
        step=0,
        scheduler=None,
        grad_scaler=None,
    )

    train_step_fn = get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=True,
        use_fp16=False,
    )

    N, C, H, W, D = 1, test_config.data.num_channels, *test_config.data.image_size
    x = torch.ones(N, C, H, W, D, device=test_config.device)

    loss = train_step_fn(state, x)
    assert loss.shape == torch.Size([])
    assert not loss.isnan()
    expected_loss = (x + sde.sigma_max).sum()
    assert loss < expected_loss
