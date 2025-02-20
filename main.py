#Install required packages
# pip install -r requirements.txt

#Import libraries
from src.image_recognition import image_label_detector, choose_label
from src.poem_generator import poem_generator

# Define paths
project_folder_path="C:/Users/Annek/Documents/pytho exos/Phoetry/Phoetry_2"
images_path=project_folder_path+"/images"
model_path=project_folder_path+"/trained_model/poet-gpt2"

# Proposed labels
labels=["tree","flower","sunset","sunrise","cloud","mountain","beach","river","lake","waterfall","forest","grassland","desert","rain","snow",
                   "road","traffic jam","hill","valley","cave","farm","garden","coastline","field","pond","sky","animal","insect","fungi",
                   "reaf","pebble","stone", "dog", "cat","bird","butterfly","bee", "stars","moon"]

# Poem generation from picture
image_path=images_path+"/sunset.jpg"
def generate_poem_from_picture(image_path,labels):
    top3_predicted_labels=image_label_detector(image_path,labels)
    theme=choose_label(top3_predicted_labels)
    poem=poem_generator(model_path=model_path, theme=theme, 
          max_length=200,  temperature=0.5,  top_k=60, top_p=0.9, repetition_penalty=1.5)
    return poem
print(generate_poem_from_picture(image_path,labels))
