import os
import json
import requests

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.my_log import get_logger
from src.image_recognition import SetClipModel
from src.poem_generator import PoemModel

# SETTING LOGGER
logger = get_logger(name=__name__)

# GETTING THE JSON WHERE AVAILABLE MODELS ARE REGISTERED
URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Poem_models/model_available.json"

response = requests.get(URL)
if response.status_code == 200:
    print("Download : Success --> Available Models")
    available_models = response.json()

    i = 1
    for model in available_models.keys():
        print(f"Available Model [{i}/{len(available_models.keys())}] : {model}")
        i += 1


URL_en_poems = available_models['gpt2_fr_haiku']
poem_model = PoemModel(URL=URL_en_poems)

theme = 'moon'

poem = poem_model.generate_poem(theme=theme)
print(f"the poem : {len(poem)}", end="\n")
print(poem)
