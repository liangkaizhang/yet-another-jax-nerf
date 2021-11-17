import threading
from typing import Callable, Dict, List, Tuple
import attr
import os
from collections import defaultdict
import tensorflow as tf

from read_write_model import read_model, qvec2rotmat
from geometry import  Rays, Camera


@attr.s(frozen=True, auto_attribs=True)
class DatasetConfig:
    """Dataset configuration class."""
    model_dir: str
    images_dir: str
    batch_size: int
    batch_per_image: int = 4
    batch_single_image: bool = True
    is_training: bool = False
    prefetch_size: int = 10
    float_image: bool = True
    use_pixel_centers = True
    use_ndc: bool = True


class DatasetBuilder(object):
    """Build a tf.data.Dataset class."""
    def __init__(self, config: DatasetConfig):
        self._config = config
        self._cameras_meta, self._images_meta, _ = read_model(self._config.model_dir, ext=".bin")
    
    def build(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            self._parse_dataset,
            {"origins": tf.float32, "directions": tf.float32, "pixels": tf.float32},
        )
        options = tf.data.Options()
        options.threading.private_threadpool_size = 48
        ds = ds.with_options(options)

        if self._config.is_training:
            ds = ds.repeat()
            ds = ds.shuffle(4 * len(self._images_meta), seed=0)

        ds = ds.unbatch()  # Unbatch rays sampled from images into individual rays. 

        if self._config.is_training:
            sample_per_image = self._config.batch_per_image * self._config.batch_size
            ds = ds.shuffle(4 * sample_per_image, seed=0)

        ds = ds.batch(self._config.batch_size)

        if not self._config.is_training:
            ds = ds.repeat()

        ds = ds.prefetch(self._config.prefetch_size)
        return ds

    def _parse_dataset(self):
        for image_id in self._images_meta:
            yield self._parse_single_image(image_id)

    def _parse_single_image(self, image_id: int) -> Dict[str, tf.Tensor]:
        image_meta = self._images_meta[image_id]
        image_filename = os.path.join(self._config.images_dir, image_meta.name)
        image = tf.image.decode_image(tf.io.read_file(image_filename), channels=3)
        if self._config.float_image:
            image = tf.image.convert_image_dtype(image, tf.float32)
        # Parse camera.
        camera_meta = self._cameras_meta[image_meta.camera_id]
        params = camera_meta.params
        rotation = qvec2rotmat(image_meta.qvec)
        translation = tf.expand_dims(image_meta.tvec, -1)
        camera = Camera(*params, rotation, translation)
        # Genrate rays from sampled pixels.
        height = camera_meta.height
        width = camera_meta.width
        pixels, locations = self._sample_pixels(image, height, width)
        rays = self._generate_rays(locations, camera)
        origins = tf.convert_to_tensor(rays.origins, dtype=tf.float32)
        directions = tf.convert_to_tensor(rays.directions, dtype=tf.float32)
        return {"origins": origins, "directions": directions, "pixels": pixels}

    def _sample_pixels(self, image: tf.Tensor,
                       height: tf.int32, width: tf.int32) -> Tuple[tf.Tensor, tf.Tensor]:
        """Random sample N pixel values and locations from input image."""
        image = tf.reshape(image, [-1 ,3])
        idxs = tf.range(tf.shape(image)[0])
        sample_per_image = self._config.batch_per_image * self._config.batch_size
        ridxs = tf.random.shuffle(idxs)[:sample_per_image]
        pixels = tf.gather(image, ridxs)
        locations = tf.unravel_index(
            indices=ridxs, dims=[height, width])
        return pixels, locations
    
    def _generate_rays(self, locations: tf.Tensor, camera: Camera) -> Rays:
        """Generate rays emitting from pixel locations."""
        pixel_center = 0.5 if self._config.use_pixel_centers else 0.0
        x = tf.cast(locations[1, ...], tf.float32) + pixel_center
        y = tf.cast(locations[0, ...], tf.float32) + pixel_center
        return camera.to_world_rays(x, y)