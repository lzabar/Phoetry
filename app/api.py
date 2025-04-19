import os
import json
import requests

from fastapi import FastAPI
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



### CODE TO CHANGE 



class ModelRequest(BaseModel):
    model_name: str


class PoemRequest(BaseModel):
    model_name: str
    theme: str


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




@app.post("/init_model/")
def init_model(req: ModelRequest):

    if req.model_name in available_models.keys():

        poem_model = PoemModel(available_models[req.model_name])

        if model_status[req.model_name] == 'Initialized':
            return {
                f"Model {req.model_name}": "Valid",
                f"Model {req.model_name}": "Already initialized, ready to run"
            }
        else:

    else:
        return {"message": f"Modèle '{model_name}' not available"}
