# Phoetry
The aim of the project if to generate a poem based on what's on the picture. The identification of elements on the picture focuses on nature-related categories. 
PS: This is a project just for pratice.

The are two main parts on the project:


## 1 Image recognition
To identify specific objects in an image. Our focus here is nature (but can be changed according to preferences)
```
from src.image_recognition import image_label_detector

#Proposed labels
labels=["tree","flower","sunset","sunrise","cloud","mountain","beach","river","lake","waterfall","forest","grassland","desert","rain","snow",
                   "road","traffic jam","hill","valley","cave","farm","garden","coastline","field","pond","sky","animal","insect","fungi",
                   "reaf","pebble","stone", "dog", "cat","bird","butterfly","bee", "stars","moon"]
image_path="path/image.jpg"
top3_predicted_labels=image_label_detector(image_path,labels)
print(top3_predicted_labels)
```
## 2 Poem generation
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
- Maybe created an API
