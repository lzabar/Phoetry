<p align="center">
  <!--img src="https://github.com/user-attachments/assets/aed37485-afd3-4943-926f-8b962efd1d99" alt="image" width="500"/><br>-->
  <img src="https://github.com/user-attachments/assets/e8a4a5ba-8ab8-4a79-9b8f-acb8d8ac06fd" alt="image" width="500"/><br>
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


### 3. Disclaimer
The poems are a little wonky.

### 🧰4. Built with
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
### 🧬1. How to use
#### 1.1 Simply via an URL

The API is accessible at : ....
The interface is intuitive but here are the instructions :
- Click on generate poems
- Upload a photo with .jpg format
- Chose a theme
- Click on generate poem
Your poem should appear in seconds


#### 1.2 Locally

You can use the API locally.
To do so, clone the repository in python environment and go to dir "/Phoetry"

Create a virtual environment simply with the command :
`chmod +x install.sh`
`sudo ./install.sh`

Your virtual environment should install all requirements and be activated 

Then, to run the API use the command
`uvicorn app.api:app --reload --host "0.0.0.0" --port 8000`

The API should start run locally on port 8000


### 🧬2. Project's structure
#### 📂2.1 Structure of repository
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

#### 🧺2.2 Structure of S3 bucket
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


### 👨‍💻3. Developer's guide
<p align="right">(<a href="#readme-top">back to top</a>)</p>

To run the training script on SSPCloud, you need to define the `BUCKET_NAME` environment variable:

```
export BUCKET_NAME={your_sspcloud_id}
python train.py
```

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

