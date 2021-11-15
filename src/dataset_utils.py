from typing import Dict, List, Tuple
import attr
import tensorflow as tf
import numpy as np
from collections import defaultdict
from read_write_model import read_model, qvec2rotmat


def parse_model(model_dir: str) -> Dict[str, List]:
    """Read Colmap model into tensor dict."""
    cameras, images, _ = read_model(model_dir, ext=".bin")
    tensor_dict = defaultdict(list)
    for idx, image in images.items():
        camera = cameras[image.camera_id]
        tensor_dict['filename'].append(image.name)
        tensor_dict['params'].append(camera.params)
        tensor_dict['rotation'].append(qvec2rotmat(image.qvec))
        tensor_dict['translation'].append(np.expand_dims(image.tvec, -1))
        tensor_dict['width'].append(camera.width)
        tensor_dict['height'].append(camera.height)
        tensor_dict['raw_image'] = tf.io.read_file(image.name)
    return dict(tensor_dict)


def read_images(path_to_img: str, to_float=True) -> tf.Tensor:
    """Read image as tensor."""
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_image(image, channels=3)
    if to_float:
        image = tf.image.convert_image_dtype(image, tf.float32)
    return image