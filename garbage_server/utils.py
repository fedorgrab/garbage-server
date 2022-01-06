import numpy as np
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from PIL import Image
from io import BytesIO
import constants


def load_image_into_numpy_array(data) -> np.array:
    return np.array(Image.open(BytesIO(data)))


def load_images_as_tensor(data) -> np.array:
    images = [
        resize(decode_jpeg(img_bytes), constants.IMAGE_SIZE)
        for img_bytes in data
    ]
    return tf.stack(images)
