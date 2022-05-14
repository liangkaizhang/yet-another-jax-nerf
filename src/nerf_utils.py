import jax.numpy as jnp
from jax import random, vmap
from geometry import Rays

EPS = 1e-10


def sample_pdf(rng, bin_start, bin_width, bin_weights, num_samples, randomized=False):
    pdf = (bin_weights + EPS) / (jnp.sum(bin_weights, axis=-1, keepdims=True) + EPS)
    cdf = jnp.cumsum(pdf, axis=-1)
    cdf = jnp.concatenate([jnp.zeros_like(cdf[..., :1]), cdf], axis=-1)

    if randomized:
        uniform_samples = random.uniform(rng, (num_samples,))
    else:
        uniform_samples = jnp.linspace(0.0, 1.0, num_samples)

    index_upper = jnp.searchsorted(cdf, uniform_samples, side='right')
    index_lower = jnp.maximum(0, index_upper - 1)

    cdf_upper = cdf[index_upper]
    cdf_lower = cdf[index_lower]

    slope = jnp.nan_to_num((uniform_samples - cdf_lower) / (cdf_upper - cdf_lower), 0.0)
    slope = jnp.clip(slope, 0., 1.)
    
    sampled_values = bin_start[index_lower] + slope * bin_width[index_lower]
    return sampled_values


def sample_along_rays(rng: jnp.ndarray, rays: Rays, all_bins: jnp.ndarray, all_weights: jnp.ndarray, num_samples: int, randomized: bool = False, combine=False) -> jnp.ndarray:
    def sample_helper(rng, origins, directions, bins, weights):
        bin_start = bins[..., :-1]
        bin_width = bins[..., 1:] - bins[..., :-1]
        bin_weights = .5 * (weights[..., :-1] + weights[..., 1:])
        z_samples = sample_pdf(rng, bin_start, bin_width, bin_weights, num_samples, randomized)
        if combine:
            z_samples = jnp.concatenate([bins, z_samples], axis=-1)
        z_samples = jnp.sort(z_samples)
        points = origins[None, :] + z_samples[..., None] * directions[None, :]
        return points, z_samples

    num_rays = rays.origins.shape[0]
    keys = random.split(rng, num_rays)
    points, z_values = vmap(sample_helper)(keys, rays.origins, rays.directions, all_bins, all_weights)
    return points, z_values


def positional_encoding(x: jnp.ndarray, min_deg: int, max_deg: int):
    if min_deg == max_deg:
        return x
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    xb = jnp.reshape(x[..., None, :] * scales[:, None], list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)


def volumetric_rendering(points_rgb, points_sigma, points_distance, dirs, white_bkgd=True):
    dists = points_distance[..., 1:] - points_distance[..., :-1]
    dists = jnp.concatenate([dists, 1 / EPS * jnp.ones_like(dists[..., :1])], axis=-1)
    dists = dists * jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    dists_sigma = points_sigma[..., -1] * dists
    alpha = 1.0 - jnp.exp(-dists_sigma)

    transmit = jnp.exp(-jnp.cumsum(dists_sigma[:, :-1], axis=-1))
    transmit = jnp.concatenate([jnp.ones_like(transmit[:, :1]), transmit], axis=-1)
    weights = alpha * transmit
    
    rgb = jnp.sum(weights[..., None] * points_rgb, axis=1)
    depth = jnp.sum(weights * points_distance, axis=1, keepdims=True)
    
    acc = jnp.sum(weights, axis=-1, keepdims=True)
    if white_bkgd:
        rgb = rgb + (1. - acc)
    return rgb, depth, acc, weights
