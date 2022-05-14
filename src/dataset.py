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
        with tf.device('/cpu:0'):
            return self._build(self._train_split_indices,
                            self._config.train_batch_size,
                            is_training=True)

    def build_test_dataset(self):
        with tf.device('/cpu:0'):
            return self._build(self._test_split_indices,
                            self._config.test_batch_size,
                            is_training=False)

    def _build(self, image_indices: List,
               batch_size: int, is_training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(image_indices)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 48
        ds = ds.with_options(options)

        if is_training:
            ds = ds.repeat()

        if is_training:
            num_depth_samples = int(batch_size * self._config.depth_sample_ratio)
            num_color_samples = batch_size - num_depth_samples
        else:
            num_depth_samples = 0
            num_color_samples = batch_size

        def _parser_fn(image_id: tf.Tensor):
            return tf.py_function(func=self._parse_single_example,
                                  inp=[image_id, num_color_samples, num_depth_samples, is_training],
                                  Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            
        ds = ds.map(_parser_fn) #, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_training and (not self._config.batch_from_single_image):
             # Unbatch rays sampled from the same image, shuffle and re-batch.
            ds = ds.unbatch()
            ds = ds.shuffle(16 * batch_size, seed=0)
            ds = ds.batch(batch_size)

        if not is_training:
            ds = ds.repeat()
        ds = ds.prefetch(self._config.prefetch_size)
        return ds

    def _parse_single_example(self,
                              image_id: tf.Tensor,
                              num_color_samples: int,
                              num_depth_samples: int,
                              is_training: bool=False):
        """Parse single example from dataset.

            Finds values for query points on a grid using bilinear interpolation.
    
        Args:
            image_id: id of the input image.
            num_color_samples: number of ray samples with color only.
            num_depth_samples: number of ray samples with depth.
            is_training: whether is training.

        Returns:
            origins: ray origins of shape `[num_samples, 3]`.
            directions: ray directions of shape `[num_samples, 3]`.
            colors: ray colors of shape `[num_samples, 3]`.
            depths: ray depths of shape `[num_samples, 1]`.
            weights: ray weights of shape `[num_samples, 1]`.

        Raises:
            ValueError: if both `num_color_samples` and `num_depth_samples` are zeros.
        """
        if (not num_color_samples) and (not num_depth_samples):
            raise ValueError("Inputs `num_color_samples` or `num_depth_samples` can't be both zeros.")

        # Decode image.
        image_meta = self._images_meta[image_id.numpy()]  # tf.Tensor is unhashable!
        image_filename = os.path.join(self._config.images_dir, image_meta.name)
        image = tf.image.decode_image(tf.io.read_file(image_filename), channels=3)
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
    
        origins = []
        directions = []
        colors = []
        depths = []
        weights = []

        # Genrate color rays.
        if num_color_samples != 0:
            origins_c, directions_c, colors_c = self._sample_color_rays(
                image, camera, num_color_samples, is_training)
            origins.append(origins_c)
            directions.append(directions_c)
            colors.append(colors_c)
            depths.append(tf.ones_like(colors_c[..., -1:]) * np.nan)
            weights.append(tf.ones_like(colors_c[..., -1:]) * np.nan)

        # Genrate depth rays.
        if is_training and num_depth_samples:
            # Parse 2D/3D key-points.
            points_2d = image_meta.xys
            points_3d_idx = image_meta.point3D_ids
            valid_points_idx = (points_3d_idx != -1)
            points_2d = points_2d[valid_points_idx]
            points_3d_idx = points_3d_idx[valid_points_idx]
            points_3d = np.array([self._points_3d[i].xyz for i in points_3d_idx])
            errors = np.array([self._points_3d[i].error for i in points_3d_idx])

             # Genrate depth rays.
            origins_d, directions_d, colors_d, depths_d, weights_d = self._sample_depth_rays(
                image, camera, points_2d, points_3d, errors, num_depth_samples)

            origins.append(origins_d)
            directions.append(directions_d)
            colors.append(colors_d)
            depths.append(depths_d)
            weights.append(weights_d)
    

        origins = tf.concat(origins, axis=0)
        directions = tf.concat(directions, axis=0)
        colors = tf.concat(colors, axis=0)
        depths = tf.concat(depths, axis=0)
        weights = tf.concat(weights, axis=0)

        if is_training:
            # Random shuffle all results.
            idxs = tf.range(origins.shape[0])
            idxs = tf.random.shuffle(idxs)
            origins = origins[idxs, :]
            directions = directions[idxs, :]
            colors = colors[idxs, :]
            depths = depths[idxs]
            weights = weights[idxs]

        return origins, directions, colors, depths, weights


    def _sample_color_rays(self, image, camera, num_samples, randomized=True):
        height, width, _ = image.shape
        if num_samples > height * width:
            raise ValueError("Input `num_samples` exceeds image resolution!")

        colors, locations = dataset_utils.sample_pixels(image, num_samples, randomized)
        x = tf.cast(locations[..., 1], tf.float32)
        y = tf.cast(locations[..., 0], tf.float32)
        rays = dataset_utils.generate_rays(x, y, camera, self._config.use_pixel_centers)
        origins = tf.convert_to_tensor(rays.origins, dtype=tf.float32)
        directions = tf.convert_to_tensor(rays.directions, dtype=tf.float32)
        return origins, directions, colors

    def _sample_depth_rays(self, image, camera, all_points_2d, all_points_3d, all_errors, num_samples):
        # Random sample 2D points.
        num_points = all_points_2d.shape[0]
        idx = np.random.randint(0, num_points, size=num_samples)
        points_2d = all_points_2d[idx, :]
  
        # Retrieve corresponding 3D points.
        points_3d = all_points_3d[idx, :]

        # Retrieve corresponding point weights.
        errors = all_errors[idx]
        mean_error = np.mean(all_errors)
        weights = np.exp(-(errors / mean_error) ** 2)

        # Sample color values for all 2D points.
        colors = dataset_utils.interpolate_bilinear(image, points_2d)

        # Generate rays for all 2D points.
        x = points_2d[..., 0]
        y = points_2d[..., 1]
        rays = dataset_utils.generate_rays(x, y, camera, self._config.use_pixel_centers)

        # Convert all results to tf tensors.
        origins = tf.convert_to_tensor(rays.origins, dtype=tf.float32)
        directions = tf.convert_to_tensor(rays.directions, dtype=tf.float32)
        points_3d = tf.convert_to_tensor(points_3d, dtype=tf.float32)
        depths = tf.norm(points_3d - origins, axis=-1, keepdims=True)
        weights = tf.convert_to_tensor(weights[..., None], dtype=tf.float32)
        colors = tf.convert_to_tensor(colors, dtype=tf.float32)
        return origins, directions, colors, depths, weights


