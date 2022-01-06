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
    for file in files:
        with open(f"{constants.DATA_SAVING_PATH}/{secrets.token_hex(4)}.jpeg", "wb+") as out_file:
            file_bytes = file.file.read()
            out_file.write(file_bytes)
            file_bytes_list.append(file_bytes)

    tensor_image_batch = utils.load_images_as_tensor(file_bytes_list)
    classes = np.argmax(classifier.predict(tensor_image_batch), axis=1)
    pred_class = stats.mode(classes, axis=None).mode[0]
    return {"class": constants.CLASSES[pred_class]}
