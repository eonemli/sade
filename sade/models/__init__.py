import torch
from .registry import create_sde

class BaseScoreModel(torch.nn.Module):
    """Base class for score models."""
    
    def __init__(self, config):
        super().__init__()
        self.model_config = config.__module__
        self.sde = create_sde(config)
        self.register_buffer("sigmas", self.sde.discrete_sigmas)