import torch.nn as nn
import torch
from torch.distributions import Normal

class Actor(nn.Module):
    """
    An actor network which uses a common backbone to compute both mu and the std
    of multiple normal distributions.

    Returns a distribution when called, allowing for the sampling of actions
    and computation of log probabilities
    """
    def __init__(
        self, in_dim: int, out_dim: int, layer_1_size: int=256, layer_2_size: int=256):
        super(Actor, self).__init__()

        self.network_base = nn.Sequential(
                nn.Linear(in_dim, layer_1_size),
                nn.ReLU(),
                nn.Linear(layer_1_size, layer_2_size),
                nn.ReLU()
            )
        
        self.layer_2_size = layer_2_size
        self.out_dim = out_dim

    def forward(self, state: torch.Tensor) -> torch.distributions.Normal:
        x = self.network_base(state)
        
        mu = nn.Linear(self.layer_2_size, self.out_dim)(x)

        # Force the log_std to be positive
        log_std = nn.Tanh()(nn.Linear(self.layer_2_size, self.out_dim)(x))
        
        dist = Normal(mu, torch.exp(log_std))
        
        return dist

class Critic(nn.Module):
    def __init__(
        self, in_dim: int, layer_1_size: int=256, layer_2_size: int=256):

        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, layer_1_size),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
