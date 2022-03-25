import jax.numpy as jnp
from jax import random, vmap
from geometry import Rays

EPS = 1e-5

def sample_pdf(rng, bin_start, bin_width, bin_weights, num_samples, randomized=False):
    bin_weights += EPS
    pdf = bin_weights / jnp.sum(bin_weights, axis=-1, keepdims=True)
    cdf = jnp.cumsum(pdf, axis=-1)
    cdf = jnp.concatenate([jnp.zeros_like(cdf[..., :1]), cdf], axis=-1)

    if randomized:
        uniform_samples = random.uniform(rng, (num_samples,))
    else:
        uniform_samples = jnp.linspace(0.0, 1.0, num_samples)

    index_higher = jnp.searchsorted(cdf, uniform_samples, side='right')
    index_lower = jnp.maximum(0, index_higher - 1)

    cdf_higher = cdf[index_higher]
    cdf_lower = cdf[index_lower]

    bin_prob  = cdf_higher - cdf_lower
    bin_prob = jnp.where(bin_prob < EPS, 1.0, bin_prob)
    slope = (uniform_samples - cdf_lower) / bin_prob
    
    sampled_values = bin_start[index_lower] + slope * bin_width[index_lower]
    return jnp.sort(sampled_values)


def sample_along_rays(rng: jnp.ndarray, rays: Rays, all_bins, all_weights, num_samples: int, randomized: bool = False) -> jnp.ndarray:
    
    def sample_helper(rng, origin, direction, bins, weights):
        bin_start = bins[..., :-1]
        bin_width = bins[..., 1:] - bins[..., :-1]
        bin_weights = .5 * (weights[..., 1:] + weights[..., :-1])
        z_samples = sample_pdf(rng, bin_start, bin_width, bin_weights, num_samples, randomized)
        direction = z_samples[..., jnp.newaxis] * direction[jnp.newaxis, :]
        points = origin[jnp.newaxis, :] + direction
        return points, z_samples

    num_rays = rays.origins.shape[0]
    rngs = random.split(rng, num_rays)
    points, z_values = vmap(sample_helper)(rngs, rays.origins, rays.directions, all_bins, all_weights)
    return points, z_values


def positional_encoding(x: jnp.ndarray, min_deg: int, max_deg: int):
    if min_deg == max_deg:
        return x
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    xb = jnp.reshape(x[..., jnp.newaxis] * scales, [x.shape[0], x.shape[1], -1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)


def volumetric_rendering(points_rgb, points_sigma, points_z, dirs, white_bkgd=True):
    dists = points_z[..., 1:] - points_z[..., :-1]
    dists = jnp.concatenate([dists, 1e10 * jnp.ones_like(dists[..., :1])], axis=-1)
    dists = dists * jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    dists_sigma = points_sigma[..., -1] * dists
    alpha = 1.0 - jnp.exp(-dists_sigma)

    transmit = jnp.exp(-jnp.cumsum(dists_sigma[:, :-1], axis=-1))
    transmit = jnp.concatenate([jnp.ones_like(transmit[:, :1]), transmit], axis=-1)
    weights = alpha * transmit
    
    rgb = jnp.sum(weights[..., jnp.newaxis] * points_rgb, axis=1)
    depth = jnp.sum(weights * points_z, axis=1, keepdims=True)
    
    acc = jnp.sum(weights, axis=-1, keepdims=True)
    disp = acc / (depth + EPS)
    if white_bkgd:
        rgb = rgb + (1. - acc)
    return rgb, disp, acc, weights
