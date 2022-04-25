from typing import Union, NamedTuple

import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class Rays(NamedTuple):
    origins: Union[tf.Tensor, np.ndarray, jnp.ndarray]
    directions: Union[tf.Tensor, np.ndarray, jnp.ndarray]
    colors: Union[tf.Tensor, np.ndarray, jnp.ndarray] = None
    depths: Union[tf.Tensor, np.ndarray, jnp.ndarray] = None
    weights: Union[tf.Tensor, np.ndarray, jnp.ndarray] = None


class Camera(object):
    def __init__(self,
                 focal: float, cx: float, cy: float, skew: float,
                 width: int, height: int,
                 rotation: np.ndarray, translation: np.ndarray,
                 near: float=0.0, far: float=1000.):
        self.fx = focal
        self.fy = focal
        self.cx = cx
        self.cy = cy
        self.skew = skew
        self.width = int(width)
        self.height = int(height)
        self.rotation = rotation
        self.translation = translation
        self.near = near
        self.far = far


    def to_local_rays(self, x: np.ndarray, y: np.ndarray, use_ndc=False) -> Rays:
        xn = (x - self.cx) / self.fx
        yn = (y - self.cy) / self.fy
        zn = np.ones_like(xn)
    
        def _normalize(x):
            norm = np.atleast_1d(np.linalg.norm(x, axis=-1))
            return x / np.expand_dims(norm + 1e-5, -1)

        directions = _normalize(np.stack((xn, yn, zn), axis=-1))
        origins = np.zeros_like(directions)
        return Rays(origins.T, directions.T)

    def to_world_rays(self, x: np.ndarray, y: np.ndarray, use_ndc=False) -> Rays:
        c_rays = self.to_local_rays(x, y, use_ndc)
        # Transform origins to world.
        w_origins = c_rays.origins - self.translation
        w_origins = np.matmul(self.rotation.T, w_origins)
        # Transform directions to world.
        w_directions = np.matmul(self.rotation.T, c_rays.directions)
        return Rays(w_origins.T, w_directions.T)