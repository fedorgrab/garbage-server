import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import constants


def load_image_into_numpy_array(data) -> np.array:
    return np.array(Image.open(BytesIO(data)))


def load_images_as_tensor(data) -> np.array:
    images = [
        cv2.resize(np.array(Image.open(BytesIO(img_bytes))), dsize=constants.IMAGE_SIZE)
        for img_bytes in data
    ]
    return np.stack(images)
