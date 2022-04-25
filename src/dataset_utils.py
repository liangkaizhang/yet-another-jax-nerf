from typing import Tuple

import tensorflow as tf

from geometry import Rays, Camera


def interpolate_bilinear(image: tf.Tensor, query_points: tf.Tensor) -> tf.Tensor:
    height, width, _ = image.shape
    u = tf.cast(query_points[..., 1], tf.float32)
    v = tf.cast(query_points[..., 0], tf.float32)

    u_lower = tf.minimum(tf.math.floor(u), height - 2)
    u_upper = u_lower + 1.

    v_lower = tf.minimum(tf.math.floor(v), width - 2)
    v_upper = v_lower + 1.

    uv_lower_lower = tf.cast(tf.stack((u_lower, v_lower), axis=-1), tf.int32)
    uv_lower_upper = tf.cast(tf.stack((u_lower, v_upper), axis=-1), tf.int32)
    uv_upper_lower = tf.cast(tf.stack((u_upper, v_lower), axis=-1), tf.int32)
    uv_upper_upper = tf.cast(tf.stack((u_upper, v_upper), axis=-1), tf.int32)

    values_lower_lower = tf.gather_nd(image, uv_lower_lower)
    values_lower_upper = tf.gather_nd(image, uv_lower_upper)
    values_upper_lower = tf.gather_nd(image, uv_upper_lower)
    values_upper_upper = tf.gather_nd(image, uv_upper_upper)

    values_lower_lower *= ((u_upper - u) * (v_upper - v))[..., None]
    values_lower_upper *= ((u_upper - u) * (v - v_lower))[..., None]
    values_upper_lower *= ((u - u_lower) * (v_upper - v))[..., None]
    values_upper_upper *= ((u - u_lower) * (v - v_lower))[..., None]

    values = values_lower_lower + values_lower_upper + values_upper_lower + values_upper_upper
    return values


def sample_pixels(image: tf.Tensor, num_samples: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Random sample `num_samples` pixel values and locations from input image."""
    height, width, _ = image.shape
    assert num_samples <= height * width

    image = tf.reshape(image, [-1 ,3])
    idxs = tf.range(tf.shape(image)[0])
    if num_samples > 0:
        ridxs = tf.random.shuffle(idxs)[:num_samples]
    else:
        ridxs = idxs
    colors = tf.gather(image, ridxs)
    locations = tf.unravel_index(
        indices=ridxs, dims=[height, width])
    return colors, tf.transpose(locations)

def generate_rays(x: tf.Tensor, y: tf.Tensor, camera: Camera, use_pixel_centers: bool=True) -> Rays:
    """Generate rays emitting from pixel locations."""
    pixel_center = 0.5 if use_pixel_centers else 0.0
    x = x + pixel_center
    y = y + pixel_center
    return camera.to_world_rays(x, y)

