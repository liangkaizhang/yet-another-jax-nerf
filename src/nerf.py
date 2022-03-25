from typing import Any, Callable
import attr
import functools

from flax import linen as nn
import jax.numpy as jnp
import jax
from jax import jit, random

from geometry import Rays
from nerf_utils import sample_along_rays, positional_encoding, volumetric_rendering


@attr.s(frozen=True, auto_attribs=True)
class MlpNetConfig:
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  skip_layer: int = 4  # The layer to add skip layers to.
  num_rgb_channels: int = 3  # The number of RGB channels.
  num_sigma_channels: int = 1  # The number of sigma channels.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The net activation function.
  rgb_activation: Callable[Ellipsis, Any] = nn.sigmoid  # The rgb activation function.
  sigma_activation: Callable[Ellipsis, Any] = nn.relu  # The sigma activation function.


class MlpNet(nn.Module):
    """A simple MLP."""
    config: MlpNetConfig

    @nn.compact
    def __call__(self, x, condition=None):
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
        raw_sigma = self.config.sigma_activation(raw_sigma)
        return raw_rgb, raw_sigma


@attr.s(frozen=True, auto_attribs=True)
class NerfModuleConfig:
    near_clip: float = 0.001  # The near clipping distance when sampling along rays.
    far_clip: float = 1000.  # The far clipping distance when sampling along rays.
    num_samples: int = 100  # Number of sampling points along rays.
    randomized: bool = True  # Whether random sample.
    use_views: bool = True
    min_deg_point: int = 0   # The minimum degree of positional encoding for positions.
    max_deg_point: int = 10  # The maximum degree of positional encoding for positions.
    white_bkgd: bool = False
    mlp_config: MlpNetConfig = MlpNetConfig()


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
                                           randomized=self.config.randomized))
        points, points_z = sampler_fn(key, rays, bins, weights)
        posi_encode = positional_encoding(points, self.config.min_deg_point,
                                          self.config.max_deg_point)
        view_encode = None
        if self.config.use_views:
            # views = jnp.broadcast_to(rays.directions[:, jnp.newaxis, :], points.shape)
            views = rays.directions[:, jnp.newaxis, :]
            view_encode = positional_encoding(views, self.config.min_deg_point,
                                              self.config.max_deg_point)
        raw_rgb, raw_sigma = MlpNet(self.config.mlp_config)(posi_encode, view_encode)
        rgb, _, acc, new_weights = volumetric_rendering(raw_rgb,
                                                        raw_sigma,
                                                        points_z,
                                                        rays.directions,
                                                        self.config.white_bkgd)
        return rgb, acc, points_z, new_weights


@attr.s(frozen=True, auto_attribs=True)
class NerfConfig:
    coarse_module_config: NerfModuleConfig = NerfModuleConfig()
    fine_module_config: NerfModuleConfig = NerfModuleConfig()


class Nerf(nn.Module):
    config: NerfConfig = NerfConfig()

    @nn.compact
    def __call__(self, rng: jnp.ndarray, rays: Rays):
        """Nerf Model"""
        rng0, rng1 = random.split(rng)
        coarse_nerf = NerfModule(self.config.coarse_module_config)
        coarse_rgb, _, bins, weights = coarse_nerf(rng0, rays)

        fine_nerf = NerfModule(self.config.fine_module_config)
        fine_rgb, _, _, _ = fine_nerf(rng1, rays, bins, weights)
        return coarse_rgb, fine_rgb


def nerf_builder(rng: jnp.ndarray, config: NerfConfig, examplar_rays: Rays):
    model = Nerf(config)
    key1, key2, rng = random.split(rng, num=3)
    rays = jax.tree_map(lambda x: x[0], examplar_rays)
    params = model.init(key1, rng=key2, rays=rays)
    return model, params