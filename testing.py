import requests
import os
import s3fs
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.image_recognition import SetClipModel
from src.poem_generator import poem_generator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "poet-gpt2")


URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Poem_models/poet_gpt2/" 



fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})


MY_BUCKET = "paultoudret"

CHEMIN = "ensae-reproductibilite/Phoetry/Poem_models/poet_gpt2"
list_f = list(fs.ls(f"s3://{MY_BUCKET}/{CHEMIN}"))

# Dossier local pour stocker les fichiers téléchargés
download_dir = "./testing/poet_gpt2/"
os.makedirs(download_dir, exist_ok=True)

# URL de base du MinIO (ou autre serveur)
minio_url = "https://minio.lab.sspcloud.fr/"

# Téléchargement des fichiers
for file_name in list_f:
    if file_name[-5:] == ".keep":
        print("passed")
        pass
    else:
        name = file_name.split('/')[-1]
        print(name)
        file_url = minio_url + file_name
        local_path = os.path.join(download_dir, name)

        # Effectuer la requête HTTP pour télécharger le fichier
        print(f"Téléchargement de {file_name}...")
        response = requests.get(file_url)

        # Vérifier si le téléchargement a réussi
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"{name} téléchargé avec succès. Stored in {local_path}")
        else:
            print(f"Erreur lors du téléchargement de {file_name}: {response.status_code}")

print("Téléchargement terminé.")


model = GPT2LMHeadModel.from_pretrained(download_dir)
tokenizer = GPT2Tokenizer.from_pretrained(download_dir)

print(model)
