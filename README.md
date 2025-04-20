Faire un truc joli (mettre des images)``Licence



S3 Bucket

Phoetry
|
|- Datasets
|- Peom_models
    |- model_available.jon
    |- gpt2_en_poems
        |- gpt2_en_poems.json
        |- ...
    |- gpt2_en_haiku
        |- gpt_en_haiku.json
        |- ...
    |- ...
|- Image_models

1 Presentation du projet

Version : 1.0
Main features :
- Prend une image / thèmes possibles / Poemes générés (anglais + haiku)

  1.1 Detail reconnaissance d'image
  1.2 Detail generation de poem

  Disclaimer

  Future developpement
  
2 Comment utiliser (Docker, kuber ? Api)
Local / Pas local

3 Cote developpeur : how to train
  fichier de config.json



# Phoetry
The aim of the project if to generate a poem based on what's on the picture. The identification of elements on the picture focuses on nature-related categories. 

PS: This is a project just for practice.

The are two main parts on the project:


## 1. Image recognition
To identify specific objects in an image. Our focus here is nature (but can be changed according to preferences)
```
from src.image_recognition import image_label_detector

#Proposed labels
labels=["tree","flower","sunset","sunrise","cloud","mountain","beach","river","lake","waterfall","forest","grassland","desert","rain","snow",
                   "road","traffic jam","hill","valley","cave","farm","garden","coastline","field","pond","sky","animal","insect","fungi",
                   "leaf","pebble","stone", "dog", "cat","bird","butterfly","bee", "stars","moon","sun"]
image_path="path/image.jpg"
top3_predicted_labels=image_label_detector(image_path,labels)
print(top3_predicted_labels)
```
## 2. Poem generation
To identify specific objects in an image. Our focus here is nature (but can be changed according to preferences)
```
from src.poem_generator import poem_generator

poem=poem_generator(model_path=model_path, theme="poem_theme", 
          max_length=200,  temperature=0.5,  top_k=60, top_p=0.9, repetition_penalty=1.5)
print(poem)
```

## Make sure to install the requirements
```
pip install -r requirements.txt
```
## Disclaimer
The poems are a little wonky.
Points of improvement:
- Enrich the datasets with more poems
- Readjust the parameters for the fine-tuning of gpt2
- Readjust parameters for the poem generator
- Maybe create an API

#### Ideas to improve the poem generator:
Create a system to measure the quality of the poem:
- Check if all the words in the generated poem are in the english dictionnary and fix a threshold (ex: acceptable if 1% of the words in the poem are
 not in the dictionnary)
- Create an inverse model which identifies the theme of the poem based on the content, calculate similarity between predicted theme and true theme 
and fix a threhold (ex: acceptable if similarity (predictied_theme, true_theme)>=50%)
- Create a poem quality index such as:
    index ={1 if threshold_dico<=1% and similarity (predictied_theme, true_theme)>=50, else 0}
- Readjust generator parameters, generate 100 poems based on different themes and evaluates their poem quality index (proportion of poems with an
index=1)
- Keep readjusting to improve the proportion of poems with an index=1
