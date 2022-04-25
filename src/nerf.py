from typing import Any, Callable
import attr
import functools

import gin
from flax import linen as nn
import jax.numpy as jnp
import jax
from jax import jit, random

from geometry import Rays
import nerf_utils


#@gin.configurable
@attr.s(auto_attribs=True)
class MLPConfig:
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  skip_layer: int = 4  # The layer to add skip layers to.
  num_rgb_channels: int = 3  # The number of RGB channels.
  num_sigma_channels: int = 1  # The number of sigma channels.
  noise_std: float = 0.01  # Std dev of sigma noise.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The net activation function.
  rgb_activation: Callable[Ellipsis, Any] = nn.sigmoid  # The rgb activation function.
  sigma_activation: Callable[Ellipsis, Any] = nn.relu  # The sigma activation function.


class MLP(nn.Module):
    """A simple MLP."""
    config: MLPConfig

    @nn.compact
    def __call__(self, rng, x, condition=None, randomized=True):
        """Evaluate the MLP.
        Args:
        x: jnp.ndarray(float32), [batch, num_samples, feature], points.
        condition: jnp.ndarray(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.
        Returns:
        raw_rgb: jnp.ndarray(float32), with a shape of
            [batch, num_samples, num_rgb_channels].
        raw_sigma: jnp.ndarray(float32), with a shape of
            [batch, num_samples, num_sigma_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.reshape([-1, feature_dim])
        dense_layer = functools.partial(
            nn.Dense, kernel_init=nn.initializers.glorot_uniform())
        inputs = x
        for i in range(self.config.net_depth):
            x = dense_layer(self.config.net_width)(x)
            x = self.config.net_activation(x)
            if i % self.config.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)
            raw_sigma = dense_layer(self.config.num_sigma_channels)(x).reshape(
                [-1, num_samples, self.config.num_sigma_channels])

        if condition is not None:
            # Output of the first part of MLP.
            bottleneck = dense_layer(self.config.net_width)(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.reshape([-1, condition.shape[-1]])
            x = jnp.concatenate([bottleneck, condition], axis=-1)
            # Here use 1 extra layer to align with the original nerf model.
            for i in range(self.config.net_depth_condition):
                x = dense_layer(self.config.net_width_condition)(x)
                x = self.config.net_activation(x)
        raw_rgb = dense_layer(self.config.num_rgb_channels)(x).reshape(
            [-1, num_samples, self.config.num_rgb_channels])

        # Add activations.
        raw_rgb = self.config.rgb_activation(raw_rgb)

        if randomized and self.config.noise_std:
            noise = random.normal(rng, raw_sigma.shape, dtype=raw_sigma.dtype)
            raw_sigma += noise * self.config.noise_std

        raw_sigma = self.config.sigma_activation(raw_sigma)
        return raw_rgb, raw_sigma


#@gin.configurable
@attr.s(auto_attribs=True)
class NerfModuleConfig:
    near_clip: float = 0.01  # The near clipping distance when sampling along rays.
    far_clip: float = 100.  # The far clipping distance when sampling along rays.
    num_samples: int = 100  # Number of sampling points along rays.
    min_deg_point: int = 0   # The minimum degree of positional encoding for positions.
    max_deg_point: int = 10  # The maximum degree of positional encoding for positions.
    white_bkgd: bool = True
    mlp_config: MLPConfig = MLPConfig()


class NerfModule(nn.Module):
    config: NerfModuleConfig

    @nn.compact
    def __call__(self,
                 rng: jnp.ndarray,
                 rays: Rays,
                 weights: jnp.ndarray=None,
                 bins: jnp.ndarray=None,
                 randomized=True):
        if weights is None:
            num_rays = rays.origins.shape[0]
            num_bins = self.config.num_samples
            bins = jnp.linspace(self.config.near_clip, self.config.far_clip, num_bins)
            bins = jnp.broadcast_to(bins, (num_rays, num_bins))
            weights = jnp.ones_like(bins)

        sampler_fn = jit(functools.partial(nerf_utils.sample_along_rays,
                                           num_samples=self.config.num_samples,
                                           randomized=randomized))
        key, rng = random.split(rng)
        points, points_z = sampler_fn(key, rays, bins, weights)
        posi_encode = nerf_utils.positional_encoding(points, self.config.min_deg_point,
                                          self.config.max_deg_point)
        views = rays.directions[:, jnp.newaxis, :]
        view_encode = nerf_utils.positional_encoding(views, self.config.min_deg_point,
                                            self.config.max_deg_point)

        key, rng = random.split(rng)
        mlp =  MLP(self.config.mlp_config)
        raw_rgb, raw_sigma =mlp(key, posi_encode, view_encode, randomized)

        rgb, depth, acc, weights = nerf_utils.volumetric_rendering(
            raw_rgb, raw_sigma, points_z, rays.directions, self.config.white_bkgd)
        return rgb, depth, acc, points_z, weights



#@gin.configurable
@attr.s(auto_attribs=True)
class NerfConfig:
    coarse_module_config: NerfModuleConfig = NerfModuleConfig()
    fine_module_config: NerfModuleConfig = NerfModuleConfig()


class Nerf(nn.Module):
    config: NerfConfig = NerfConfig()

    @nn.compact
    def __call__(self, rng: jnp.ndarray, rays: Rays, randomized=True):
        """Nerf Model"""
        key0, key1 = random.split(rng)

        coarse_nerf = NerfModule(self.config.coarse_module_config)
        coarse_rgb, coarse_depth, _, bins, weights = coarse_nerf(key0, rays, randomized=randomized)

        fine_nerf = NerfModule(self.config.fine_module_config)
        fine_rgb, fine_depth, _, _, _ = fine_nerf(key1, rays, bins, weights, randomized=randomized)
        return coarse_rgb, coarse_depth, fine_rgb, fine_depth


def nerf_builder(rng: jnp.ndarray, config: NerfConfig):
    def _tmp_rays():
        tmp = jnp.zeros([1, 3], dtype=jnp.float32)
        return Rays(tmp, tmp)

    model = Nerf(config)
    key0, key1 = random.split(rng)
    params = model.init(key0, rng=key1, rays=_tmp_rays())
    return model, params