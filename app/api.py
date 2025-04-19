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



@app.get("/")
def read_root():
    """
    Only the root
    """
    return {"Welcome to Phoetry"}


@app.get("/available")
def read_model_available():
    return available_models


class ModelRequest(BaseModel):
    model_name: str

@app.get("/load_model/")
def load_model(model_name: str):
    if model_name in available_models.keys():
        return {"message": f"Modèle '{model_name}' reçu via GET et prêt à être initialisé"}
    else:
        return {"message": f"Modèle '{model_name}' not available"}
