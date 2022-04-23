from typing import Tuple

import tensorflow as tf

from geometry import Rays, Camera


def interpolate_bilinear(image: tf.Tensor, query_points: tf.Tensor) -> tf.Tensor:
    return

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
    return colors, locations