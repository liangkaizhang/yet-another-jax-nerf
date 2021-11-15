
def _read_image(path_to_img: str, to_float=True) -> tf.Tensor:
    """Read image as tensor."""
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_image(image, channels=3)
    if to_float:
        image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def _sample_pixels(image: tf.Tensor, height: int, width: int,
                   num_samples: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Random sample N pixel values and locations from input image."""
    image = tf.reshape(image, [-1 ,3])
    idxs = tf.range(tf.shape(image)[0])
    ridxs = tf.random.shuffle(idxs)[:num_samples]
    pixels = tf.gather(image, ridxs)
    locations = tf.unravel_index(indices=ridxs, dims=[height, width])
    return pixels, locations

def _generate_rays():
    """Generate rays emitting from pixel locations."""


class PreprocessFn(object):
    """Functor for preprocessing input data."""
    def __init__(self, )
