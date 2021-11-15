from typing import List
import attr
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


@attr.s(frozen=True, auto_attribs=True)
class Rays:
    origins: np.ndarray
    directions: np.ndarray


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
        directions = np.stack((xn, yn, zn), axis=-1)
        origins = np.zeros_like(directions)
        return Rays(origins, directions)


    def to_world_rays(self, x: np.ndarray, y: np.ndarray, use_ndc=False) -> Rays:
        c_rays = self.to_local_rays(x, y, use_ndc)
        # Transform origins to world.
        w_origins = c_rays.origins.T - self._translation
        w_origins = np.matmul(self._rotation.T, w_origins)
        # Transform directions to world.
        w_directions = np.matmul(self._rotation.T, c_rays.directions.T)
        return Rays(w_origins, w_directions)
