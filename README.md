<p align="center">
  <img src="https://github.com/user-attachments/assets/aed37485-afd3-4943-926f-8b962efd1d99" alt="image" width="500"/><br>
  <strong>Version : 1.0</strong>
</p>




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#i-project-presentation">Project presentation</a>
      <ul>
        <li><a href="#1-image-recognition">Image recognition</a></li>
        <li><a href="#2-poem-generation">Poem generation</a></li>
        <li><a href="#3-built-with">Built with</a></li>
        <li><a href="#4-improvements">Improvements</a></li>
      </ul>
    </li>
    <li>
      <a href="#ii-users-guide">User's guide</a>
      <ul>
        <li><a href="#1-projects-structure">Project's structure</a></li>
          <ul>
            <li><a href="#1.1-structure-of-repository">Structure of repository</a></li>
            <li><a href="#1.2-structure-of-s3-bucket">Structure of S3 bucket</a></li>
          </ul>
        <li><a href="#2-installation">Installation</a></li>
        <li><a href="#3-developers-guide">Developer's guide</a></li>
      </ul>
    </li>
    <li><a href="#iii-contributions">Contributions</a></li>
    <li><a href="#iv-license">License</a></li>
    <li><a href="#v-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- README TOP -->
<a name="readme-top"></a>

<!-- Project presentation -->
## 👨‍🏫I. Project presentation
The aim of the project if to generate a poem based on what's on the picture. There are two main parts on the project:

### 🖼️1. Image recognition
We use OpenAI's Clip model for image recognition which is a neural network which learns visual concepts from natural language supervision ([OpenAI's CLIP](https://openai.com/index/clip/)). It take in input an image and classify it based on provided labels.
Here we focus on nature-related and food-related images and the final label of the image is one of the top 3 most probable labels for the image.

### 📜2. Poem generation 
We use OpenAI's GPT-2 as base, which is a trained large-scale unsupervised language model, which generates coherent paragraphs of text ([OpenAI's GPT-2 ](https://openai.com/index/better-language-models/)). We use GPT-2 as it is fully open source with no API costs and have low hardware requirements, even though recent versions (GPT-3, GPT-4) are more efficient.
The model is fine-tuned on poems's dataset. For the generation, the label of the input image is used as the theme of the poem, whether it's a classical english poem or an haiku.


### 🧰3. Built with
* logiciel 1
* logiciel 2...

### 📈4. Improvements
Points of improvement:
* Enrich the datasets with more poems, with better filtering of texts on their quality
* Readjust the parameters for the fine-tuning of gpt2
* Readjust parameters for the poem generator
* More esthetic use friendly API
......

  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- User's guide -->
## 📄II. User's guide
### 🧬1. Project's structure
#### 📂1.1 Structure of repository
```
Phoetry
  |
  |- .github
  |- app
  |- data
  |- deployment
  |- notebooks
      |- dump_bucket.ipynb
      |- phoetry.ipynb
      |- test.ipynb
      |-...
  |- src
      |- image_recognition.py
      |- my_log.py
      |- poem_generator.py
      |- training_setup.py
      |-...
  |- .gitignore
  |- Dockerfile
  |- install.sh
  |- main.py
  |- README.md
  |- requirements.txt
  |- train.py
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### 🧺1.2 Structure of S3 bucket
```
Phoetry
  |
  |- Datasets
      |- labels.json
      |- other datasets...
  |- Poem_models
      |- model_available.json  -> --
      |- gpt2_en_poems              |
          |- gpt2_en_poems.json   <-    
          |- ...                    |
      |- gpt2_en_haiku              |
          |- gpt_en_haiku.json    <-
          |- ...
      |- ...
  |- Image_models
```

### 💻2. Installation
```
pip install -r requirements.txt
```
...


### 👨‍💻3. Developer's guide
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Contributions -->
## 🧑‍🤝‍🧑III. Contributions
@kodro23
@lzabar
@PaulToudret
<!-- License -->
## 📜IV. License
Distributed under the MIT License. See `LICENSE` for more information.
<!-- Acknowledgments -->
## 🤝V. Acknowledgments

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
