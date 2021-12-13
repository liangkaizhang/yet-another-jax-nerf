from typing import List, Any, Callable
import attr

from flax import linen as nn
import jax.numpy as jnp

from geometry import ColoredRays
from mlp import MlpConfig

@attr.s(frozen=True, auto_attribs=True)
class ModelConfig:
    # Sampling config.
    sample_near: float = 0.001  # The near clipping distance when sampling along rays.
    sample_far: float = 1000.  # The far clipping distance when sampling along rays.
    num_samples: int = 100  # Number of sampling points along rays.
    # Model structure config.
    mpl_config: MlpConfig = MlpConfig()


class ModelBase(nn.Module):
    def __init__(self, config: ModelConfig):
        self._config = config

    @nn.compact
    def __call__(self, rays: ColoredRays):
        x = self._sample_along_rays(rays)
        x = self._preprocess_fn(x)
        x = self._model_impl(x)
        return x
    
    def _sample_along_rays(rays: ColoredRays):
        
        return 

    def _preprocess_fn(self, x):
        return NotImplemented
    
    def _model_impl(self, x):
        return NotImplemented

class SimpleNerfModel(ModelBase):

    def _preprocess_fn(self, x):
        return x

    def _model_impl(self, x):

        return x




