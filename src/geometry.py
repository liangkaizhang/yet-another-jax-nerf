from typing import List
import numpy as np
from collections import namedtuple
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

Rays = namedtuple("Rays", ["origins", "directions"])
ColoredRays = namedtuple("ColoredRays", ["origins", "directions", "colors"])

class Camera(object):
    def __init__(self,
                 fx: float, fy: float, cx: float, cy: float,
                 rotation: np.ndarray, translation: np.ndarray,
                 near: float=0.0, far: float=1000.):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._rotation = rotation
        self._translation = translation
        self._near = near
        self._far = far

    def to_local_rays(self, x: np.ndarray, y: np.ndarray, use_ndc=False) -> Rays:
        xn = (x - self._cx) / self._fx
        yn = (y - self._cy) / self._fy
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
        w_origins = c_rays.origins - self._translation
        w_origins = np.matmul(self._rotation.T, w_origins)
        # Transform directions to world.
        w_directions = np.matmul(self._rotation.T, c_rays.directions)
        return Rays(w_origins.T, w_directions.T)