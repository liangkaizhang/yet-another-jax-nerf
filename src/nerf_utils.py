import jax.numpy as jnp
from jax import random, vmap
from geometry import Rays

EPS = 1e-5

def sample_pdf(rng, bins, weights, num_samples, randomized=False):
    weights += EPS
    pdf = weights / jnp.sum(weights, axis=-1, keepdims=True)
    cdf = jnp.cumsum(pdf, axis=-1)
    cdf = jnp.concatenate([jnp.zeros_like(cdf[..., :1]), cdf], axis=-1)

    if randomized:
        u = random.uniform(rng, (num_samples,))
    else:
        u = jnp.linspace(0.0, 1.0, num_samples)

    index_higher = jnp.searchsorted(cdf, u, side='right')
    index_lower = jnp.maximum(0, index_higher - 1)

    cdf_higher = cdf[index_higher]
    cdf_lower = cdf[index_lower]

    cdf_span = cdf_higher - cdf_lower
    cdf_span = jnp.where(cdf_span < EPS, 1.0, cdf_span)
    slope = (u - cdf_lower) / cdf_span
    
    bin_span = bins[index_higher] - bins[index_lower]
    sampled_values = bins[index_lower] + slope * bin_span
    return sampled_values


def sample_along_rays(rng: jnp.ndarray, rays: Rays, all_bins, all_weights, num_samples: int, randomized: bool = False) -> jnp.ndarray:
    
    def sample_helper(rng, origin, direction, bins, weights):
        z_samples = sample_pdf(rng, bins, weights, num_samples, randomized)
        direction = z_samples[..., jnp.newaxis] * direction[jnp.newaxis, :]
        points = origin[jnp.newaxis, :] + direction
        return points

    num_rays = rays.origins.shape[0]
    rngs = random.split(rng, num_rays)
    points = vmap(sample_helper)(rngs, rays.origins, rays.directions, all_bins, all_weights)
    return points


def positional_encoding(points: jnp.ndarray):
    return points


def volumetric_rendering(points_rgb, points_sigma):
    return
