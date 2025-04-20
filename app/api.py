import os
import json
import requests

from fastapi import FastAPI, File, UploadFile
import uuid
from pydantic import BaseModel

from src.my_log import get_logger
from src.image_recognition import SetClipModel
from src.poem_generator import PoemModel

# SETTING API
app = FastAPI()

# SETTING LOGGER
logger = get_logger(name=__name__)

# GETTING FROM S3 THE AVAILABLE MODEL
URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Poem_models/model_available.json"

response = requests.get(URL)
if response.status_code == 200:
    logger.info("Download : Success --> Available Models")
    available_models = response.json()
else:
    logger.error(f"Download : Fail --> Available Models : error {response.status_code}")


# Setting a dico to register the status of model
model_status = {}

for key in available_models.keys():
    model_status[key] = "Not initialized"

models = {}


### CODE TO CHANGE 


@app.get("/")
def read_root():
    """
    Only the root
    """
    return {"Welcome to Phoetry"}


@app.get("/available")
def read_model_available():
    logger.info("GET : /available")
    return model_status




@app.get("/init_model/{model_name}")
def init_model(model_name):

    if model_name in available_models.keys():

        poem_model = PoemModel(available_models[model_name])
        models[poem_model.name] = poem_model

        if model_status[model_name] == 'Initialized':
            return {
                f"Model {model_name}": "Valid",
                f"Model {model_name}": "Already initialized, ready to run"
            }
        else:
            model_status[model_name] = "Initialized"
            return {
                f"Model {model_name}": "Valid",
                f"Model {model_name}": "Initialized, ready to run"
            }

    else:
        return {"message": f"Modèle '{model_name}' not available"}




@app.get("/generate_poem/{model_name}/{theme}")
def gen_poem(model_name: str, theme: str):

    if model_name in available_models.keys():
        logger.debug("model name is available")

        if model_name in models.keys():
            logger.debug("model name is in models")
            poem_model = models[model_name]

            poem = poem_model.generate_poem(theme=theme)
            logger.debug(f"longueur du poeme : {len(poem)}")

            return {
                "Poem": poem
            }

    else:
        return {"message": f"Modèle '{model_name}' not available"}


@app.post("/upload/")
async def create_upload(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4}.jpg"
    contents = await file.read()

    with open(f"../data/image/{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}
    