from typing import List, Any, Callable
import attr
import functools

from flax import linen as nn
import jax.numpy as jnp
from jax import jit, random, vmap

from geometry import Rays
from nerf_utils import sample_along_rays, positional_encoding


@attr.s(frozen=True, auto_attribs=True)
class NetConfig:
    near_clip: float = 0.001  # The near clipping distance when sampling along rays.
    far_clip: float = 1000.  # The far clipping distance when sampling along rays.
    num_samples: int = 100  # Number of sampling points along rays.
    randomized: bool = True  # Whether random sample.

class Network(nn.Module):
    config: NetConfig

    @nn.compact
    def __call__(self, rng: jnp.ndarray, posi_encode, view_ecode):
        pass


@attr.s(frozen=True, auto_attribs=True)
class NerfModuleConfig:
    near_clip: float = 0.001  # The near clipping distance when sampling along rays.
    far_clip: float = 1000.  # The far clipping distance when sampling along rays.
    num_samples: int = 100  # Number of sampling points along rays.
    randomized: bool = True  # Whether random sample.
    use_views: bool = True
    net_config: NetConfig = NetConfig()


class NerfModule(nn.Module):
    config: NerfModuleConfig

    @nn.compact
    def __call__(self, rng: jnp.ndarray, rays: Rays,
                 weights: jnp.ndarray=None, bins: jnp.ndarray=None):
        key, rng = random.split(rng)
        if weights is None:
            num_rays = rays.origins.shape[0]
            num_bins = self.config.num_samples
            bins = jnp.linspace(self.config.near_clip, self.config.far_clip, num_bins)
            bins = jnp.broadcast_to(bins, (num_rays, num_bins))
            weights = jnp.ones_like(bins)

        sampler_fn = jit(functools.partial(sample_along_rays,
                                           num_samples=self.config.num_samples,
                                           is_random=self.config.randomized))
        points = sampler_fn(key, rays, bins, weights)
        posi_encode = positional_encoding(points)
        view_encode = None
        if self.config.use_views:
            views = jnp.broadcast_to(rays.directions[:, jnp.newaxis, :], points.shape)
            view_encode = positional_encoding(views)

        rgb_raw, sigma_raw = Network(rng, posi_encode, view_encode)

        return rgb, sigma, new_bins, new_weights


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



