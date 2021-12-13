from typing import List, Any, Callable
import attr

from geometry import ColoredRays
from flax import linen as nn
import jax.numpy as jnp

@attr.s(frozen=True, auto_attribs=True)
class MlpConfig:
    mlp_depth: int = 8  # The depth of the first part of MLP.
    mlp_width: int = 256  # The width of the first part of MLP.
    mlp_depth_condition: int = 1  # The depth of the second part of MLP.
    mlp_width_condition: int = 128  # The width of the second part of MLP.
    mlp_activation: Callable = nn.relu  # The activation function.
    skip_layer: int = 4  # The layer to add skip layers to.
    num_rgb_channels: int = 3  # The number of RGB channels.
    num_sigma_channels: int = 1  # The number of sigma channels.

class Mlp(nn.Module):
    def __init__(self, config: MlpConfig):
        self._config = config
    
    @nn.compact
    def __call__(self, x):
        return x
