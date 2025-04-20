import os
import json
import requests
from typing import Union

from io import BytesIO
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uuid
from pydantic import BaseModel

from src.my_log import get_logger
from src.image_recognition import SetClipModel
from src.poem_generator import PoemModel

# SETTING API
app = FastAPI()

# SETTING LOGGER
logger = get_logger(name=__name__)

clip_model = SetClipModel()


# GETTING FROM S3 THE AVAILABLE MODEL
URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Poem_models/model_available.json"

response = requests.get(URL)
if response.status_code == 200:
    logger.info("Download : Success --> Available Models")
    available_models = response.json()
else:
    logger.error(f"Download : Fail --> Available Models : error {response.status_code}")


URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Labels/labels.json"

response = requests.get(URL)
if response.status_code == 200:
    logger.info("Download : Success --> Labels")
    labels = response.json()
else:
    logger.error(f"Download : Fail --> Available Models : error {response.status_code}")


# Setting a dico to register the status of model
model_status = {}

for key in available_models.keys():
    model_status[key] = "Not initialized"

models = {}


### CODE POUR L'API

@app.get("/", response_class=HTMLResponse)
def homepage():
    return FileResponse("static/index.html")

@app.get("/g", response_class=HTMLResponse)
def gener_page():
    return FileResponse("static/generateur.html")



@app.get("/")
def read_root():
    """
    Only the root
    """
    return {"Welcome to Phoetry"}


@app.get("/available_models")
def read_model_available():
    logger.info("GET : /available")
    return model_status


@app.get("/available_themes")
def get_available_themes():
    logger.info("GET : /available")
    return list(labels.keys())



@app.post("/generate_poem_from_image/")
async def create_upload(
    file: UploadFile = File(...),
    theme: str = Form(...),
    model_name: str = Form(...)
):

    contents = await file.read()
    image = Image.open(BytesIO(contents))

    if model_name in available_models.keys():
        logger.debug("model name is available")

        if model_name in models.keys():
            logger.debug("model name is in models")
            poem_model = models[model_name]

        else:
            poem_model = PoemModel(available_models[model_name])
            models[poem_model.name] = poem_model

        predicted_word = clip_model.find_word(image, labels[theme])

        poem = poem_model.generate_poem(theme=predicted_word)
        
        return {"predicted_word": predicted_word, "poem": poem}

    else:
        return {
             "model": model_name + "not available"
        }
    

app.mount("/static", StaticFiles(directory="static"), name="static")
