"""Training NCSN++ on Brains with VE SDE.
   Keeping it consistent with CelebaHQ config from Song
"""
from sade.configs.default_brain_configs_abcd_asd import get_default_configs
import torch

def get_config():
    config = get_default_configs()

    config.device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = True
    training.likelihood_weighting = False
    training.reduce_mean = True
    training.batch_size = 8
    training.n_iters = 1500001
    training.pretrained_checkpoint = "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/checkpoints/checkpoint_150.pth"

    data = config.data
    data.image_size = (96, 112, 80)
    data.spacing_pix_dim = 2.0
    data.num_channels = 2
    data.cache_rate = 0.0
    data.dir_path = "/BEE/Connectome/ABCD/Users/amahmood/braintyp/processed_v2/"#"/BEE/Connectome/ABCD/Users/emre/braintyp/processed_v3/"
    data.splits_dir = "/ASD2/emre_projects/OOD/scripts/braintypicality-scripts/split-keys/"

    evaluate = config.eval
    evaluate.sample_size = 8
    evaluate.batch_size = 64
    evaluate.checkpoint_num = 150

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.warmup = 1000
    optim.scheduler = "skip"

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"
    sampling.probability_flow = False
    sampling.snr = 0.17
    sampling.n_steps_each = 1
    sampling.noise_removal = True

    # model
    model = config.model
    model.name = "ncsnpp3d"
    model.resblock_type = "biggan"
    model.act = "memswish"
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nf = 24
    model.blocks_down = (2, 2, 2, 2, 4)
    model.blocks_up = (1, 1, 1, 1)
    model.time_embedding_sz = 64
    model.init_scale = 0.0
    model.num_scales = 2000
    model.conv_size = 3
    model.self_attention = False
    model.dropout = 0.0
    model.resblock_pp = True
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = True
    model.norm_num_groups = 8

    return config
