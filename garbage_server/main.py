import typing as t
import secrets
import torch
from fastapi import FastAPI, File, UploadFile
from .model import get_trained_model
from . import constants
from . import utils

app = FastAPI()
model = get_trained_model()


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
    classes = model.predict(tensor_image_batch)
    pred_class = torch.mode(classes)[0].item()
    return {"class": constants.CLASSES[pred_class]}
