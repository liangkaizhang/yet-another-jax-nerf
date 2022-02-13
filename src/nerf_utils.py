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
        return points, z_samples

    num_rays = rays.origins.shape[0]
    rngs = random.split(rng, num_rays)
    points, z_values = vmap(sample_helper)(rngs, rays.origins, rays.directions, all_bins, all_weights)
    return points, z_values


def positional_encoding(x: jnp.ndarray, min_deg: int, max_deg: int):
    if min_deg == max_deg:
        return x
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                        list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)


def volumetric_rendering(points_rgb, points_sigma, points_z, dirs, white_bkgd=True):
    dists = points_z[..., 1:] - points_z[..., :-1]
    dists = jnp.concatenate([dists, 1e10 * jnp.ones_like(dists[..., :1])], axis=-1)
    dists = dists * jnp.linalg.norm(dirs, axis=-1)

    dists_sigma = points_sigma * dists
    alpha = 1.0 - jnp.exp(-dists_sigma)

    transmit = jnp.exp(-jnp.cumsum(dists_sigma[..., :-1], axis=-1))
    transmit = jnp.concatenate([jnp.ones_like(transmit[..., :1]), transmit], axis=-1)
    weights = alpha * transmit

    rgb = jnp.sum(weights[..., jnp.newaxis] * points_rgb, axis=-2)
    depth = jnp.sum(weights * points_z, axis=-1)
    acc = jnp.sum(weights, axis=-1)
    disp = acc / (depth + EPS)
    if white_bkgd:
        rgb = rgb + (1. - acc[..., jnp.newaxis])
    return rgb, disp, acc, weights
