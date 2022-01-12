import os
import typing as t
import secrets
import numpy as np
from scipy import stats
from fastapi import FastAPI, File, UploadFile
from .model import get_trained_model
import constants
import utils

app = FastAPI()
classifier = get_trained_model()


@app.get("/ping")
def ping():
    return {"message": "pong!"}


@app.post("/predict")
async def predict(files: t.List[UploadFile] = File(...)):
    file_bytes_list = []

    for request_file in files:
        request_file_bytes = request_file.file.read()
        file_bytes_list.append(request_file_bytes)

    tensor_image_batch = utils.load_images_as_tensor(file_bytes_list)
    classes = np.argmax(classifier.predict(tensor_image_batch), axis=1)
    predicted_class_label = stats.mode(classes, axis=None).mode[0]
    predicted_class_name = constants.CLASSES[predicted_class_label]

    for file_bytes in file_bytes_list:
        file_path = f"{constants.DATA_SAVING_PATH}/{predicted_class_name}"
        os.makedirs(file_path, exist_ok=True)
        with open(f"{file_path}/{secrets.token_hex(4)}.jpeg", "wb+") as out_file:
            out_file.write(file_bytes)

    return {"class": predicted_class_name}

