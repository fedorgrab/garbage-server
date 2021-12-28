import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO


def load_image_into_numpy_array(data) -> np.array:
    return np.array(Image.open(BytesIO(data)))


def load_images_as_tensor(data):
    images = [Image.open(BytesIO(img_bytes)) for img_bytes in data]
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    return torch.stack([tr(img) for img in images])
