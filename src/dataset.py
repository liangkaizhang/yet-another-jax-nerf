from typing import List
import attr
import os
import random
import tensorflow as tf
import numpy as np
import gin

from read_write_model import read_model, qvec2rotmat
from geometry import Rays, Camera
import dataset_utils 

tf.config.experimental.set_visible_devices([], 'GPU')


@gin.configurable
@attr.s(frozen=True, auto_attribs=True)
class DatasetConfig:
    """Dataset configuration class."""
    model_dir: str = ""
    images_dir: str = ""
    train_batch_size: int = 256
    test_batch_size: int = -1
    train_split_ratio: float = 0.9
    depth_sample_ratio: float = 0.5
    batch_from_single_image: bool = False
    prefetch_size: int = 10
    float_image: bool = True
    use_pixel_centers = True
    use_ndc: bool = True

class DatasetBuilder(object):
    """Build a tf.data.Dataset class."""
    def __init__(self, config: DatasetConfig):
        self._config = config
        self._cameras_meta, self._images_meta, self._points_3d = read_model(
                self._config.model_dir, ext=".bin")
        self._create_split()

    def _create_split(self):
        image_indices = list(self._images_meta.keys())
        random.shuffle(image_indices)
        num_train = len(image_indices) * self._config.train_split_ratio
        num_train = int(num_train)
        self._train_split_indices = image_indices[:num_train]
        self._test_split_indices = image_indices[num_train:]
    
    def build_train_dataset(self):
        return self._build(self._train_split_indices,
                           self._config.train_batch_size, True)

    def build_test_dataset(self):
        return self._build(self._test_split_indices,
                           self._config.test_batch_size, False)

    def _build(self, image_indices: List,
               batch_size: int, is_training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(image_indices)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 48
        ds = ds.with_options(options)

        if is_training:
            ds = ds.repeat()

        def _parser_fn(image_id: tf.Tensor):
            return tf.py_function(func=self._parse_single_image,
                                  inp=[image_id, batch_size],
                                  Tout=[tf.float32, tf.float32, tf.float32])
            
        ds = ds.map(_parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_training and (not self._config.batch_from_single_image):
             # Unbatch rays sampled from the same image, shuffle and re-batch.
            ds = ds.unbatch()
            ds = ds.shuffle(16 * batch_size, seed=0)
            ds = ds.batch(batch_size)

        if not is_training:
            ds = ds.repeat()
        ds = ds.prefetch(self._config.prefetch_size)
        return ds

    def _parse_single_image(self, image_id: tf.Tensor, sample_per_image: int):
        # Decode image.
        image_meta =self._images_meta[image_id.numpy()]  # tf.Tensor is unhashable!
        image_filename = os.path.join(self._config.images_dir, image_meta.name)
        image = tf.image.decode_image(tf.io.read_file(image_filename.replace('JPG', 'png')), channels=3)
        if self._config.float_image:
            image = tf.image.convert_image_dtype(image, tf.float32)
        # Parse camera.
        camera_meta = self._cameras_meta[image_meta.camera_id]
        params = camera_meta.params
        width = camera_meta.width
        height = camera_meta.height
        rotation = qvec2rotmat(image_meta.qvec)
        translation = tf.expand_dims(image_meta.tvec, -1)
        camera = Camera(*params, width, height, rotation, translation)
    
        # Genrate rays samples.
        num_depth_samples = int(sample_per_image * self._config.depth_sample_ratio)
        num_color_samples = sample_per_image - num_depth_samples
        origins, directions, colors = self._sample_color_rays(image, camera, num_color_samples)
        depth = tf.ones_like(colors[..., -1:]) * np.nan
        weights = tf.ones_like(colors[..., -1:]) * np.nan

        # if num_depth_samples > 0:
        #     # Parse 2D  key-points.
        #     points_2d = image_meta.xys
        #     points_idx = image_meta.point3D_ids
        #     valid_idx = points_idx != -1
        #     points_2d = points_2d[valid_idx]
        #     points_idx = points_idx[valid_idx]

        #     origins_d, directions_d, colors_d, depth, weights = self._sample_depth_rays(
        #         image, camera, points_2d, points_idx, num_depth_samples)

        #     origins = tf.concat((origins, origins_d), axis=0)
        #     directions = tf.concat((directions, directions_d), axis=0)
        #     colors = tf.concat((colors, colors_d), axis=0)
        #     depth = tf.concat((depth, depth))
        #     weights = tf.concat((weights, weights))

        return origins, directions, colors, depth, weights

    def _sample_color_rays(self, image, camera, num_samples):
        colors, locations = dataset_utils.sample_pixels(image, num_samples)
        x = tf.cast(locations[1, ...], tf.float32)
        y = tf.cast(locations[0, ...], tf.float32)
        rays = self.generate_rays(x, y, camera)
        origins = tf.convert_to_tensor(rays.origins, dtype=tf.float32)
        directions = tf.convert_to_tensor(rays.directions, dtype=tf.float32)
        return origins, directions, colors

    def _sample_depth_rays(self, image, camera, points_2d, points_idx, num_samples):
        idxs = tf.range(tf.shape(points_2d)[0])
        ridxs = tf.random.shuffle(idxs)[:num_samples]
        points_2d = points_2d[ridxs]
        points_idx = points_idx[ridxs]
        points_3d = np.array([self._points_3d[idx].xyz for idx in points_idx])
        weights = np.array([self._points_3d[idx].error for idx in points_idx])
        colors = dataset_utils.interpolate_bilinear(image, points_2d)

        x = points_2d[..., 0]
        y = points_2d[..., 1]
        rays = self.generate_rays(x, y, camera)
        origins = tf.convert_to_tensor(rays.origins, dtype=tf.float32)
        directions = tf.convert_to_tensor(rays.directions, dtype=tf.float32)
        points_3d = tf.convert_to_tensor(points_3d)
        depth = tf.norm(tf.points_3d - origins, axis=-1, keepdims=True)
        weights = tf.convert_to_tensor(weights[..., None])
        return origins, directions, colors, depth, weights


    def generate_rays(self, x, y, camera: Camera) -> Rays:
        """Generate rays emitting from pixel locations."""
        pixel_center = 0.5 if self._config.use_pixel_centers else 0.0
        x = x + pixel_center
        y = y + pixel_center
        return camera.to_world_rays(x, y)

