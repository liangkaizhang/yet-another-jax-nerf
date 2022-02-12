from typing import List, Any, Callable
import attr

from flax import linen as nn
import jax.numpy as jnp

from geometry import Rays
from nerf_utils import sample_along_rays, positional_encoding


@attr.s(frozen=True, auto_attribs=True)
class NerfModuleConfig:
    sample_near: float = 0.001  # The near clipping distance when sampling along rays.
    sample_far: float = 1000.  # The far clipping distance when sampling along rays.
    num_samples: int = 100  # Number of sampling points along rays.
    is_random: bool = True


class NerfModule(nn.Module):
    config: NerfModuleConfig

    @nn.compact
    def __call__(self, rng: jnp.ndarray, rays: Rays,
                 weights: jnp.ndarray=None, bins: jnp.ndarray=None):
        points = sample_along_rays(rng, rays,
                                   self.config.near,
                                   self.config.far, 
                                   self.config.num_samples,
                                   self.config.is_random,
                                   weights,
                                   bins)
        x = positional_encoding(points)
        
        return rgb, sigma, weight


@attr.s(frozen=True, auto_attribs=True)
class NerfModelConfig:
    coarse_module_config: NerfModuleConfig 
    fine_module_config: NerfModuleConfig


class NerfModel(nn.Module):
    config: NerfModelConfig

    @nn.compact
    def __call__(self, rng_0: jnp.ndarray, rng_1: jnp.ndarray, rays: Rays):
        """Nerf Model.
        Args:
            rng_0: random number generator for coarse model sampling.
            rng_1: random number generator for fine model sampling.
            rays: input ColoredRays, a namedtuple of ray origins, directions, and colors.
        Returns:
            ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
        """
        x = sample_along_rays(rays, self.config.near, self.config.far,
                              self.config.num_samples,  self.config.random_sample)
        
        return x

    def _preprocess_fn(self, x):
        return NotImplemented
    
    def _model_impl(self, x):
        return 



