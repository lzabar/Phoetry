# Import libraries
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


class IntializeClipModel:
    def __init__(self):
        self.model_id="openai/clip-vit-base-patch32"
        self.processor=CLIPProcessor.from_pretrained(self.model_id)
        self.model=CLIPModel.from_pretrained(self.model_id)
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        

def create_label_tokens(labels):
    """
    Take the specific objects we want to identify in the image and create label tokens to feed the model
    """
    clip_model=IntializeClipModel()
    #generate sentences
    clip_labels=[f"a photo of a {label}" for label in labels ]
    #Create label tokens
    label_tokens=clip_model.processor(
    text=clip_labels,
    padding=True,
    images=None,
    return_tensors='pt'
).to(clip_model.device)
    #encode tokens to sentence embeddings
    label_emb=clip_model.model.get_text_features(**label_tokens)
#   detach from pytorch gradient computation
    label_emb=label_emb.detach().cpu().numpy()
    return label_emb


#Predict label function
def image_label_detector(image_path,labels):
    """ 
    Take the image path in input and predict the 3 most probable labels
    """
    clip_model=IntializeClipModel()
    image = mpimg.imread(image_path)#Load image
    processed_image=clip_model.processor( #Process image
    text=None,
    images=image,
    return_tensors='pt'
)["pixel_values"].to(clip_model.device)
    image_emb=clip_model.model.get_image_features(processed_image).detach().cpu().numpy() #embedding
    scores=np.dot(image_emb,create_label_tokens(labels).T) #predicted lables scores (among the 39 original labels)
    top3_scores_indexes=list(np.argsort(scores)[0][-3:,][::-1])

    return np.array(labels)[top3_scores_indexes].tolist()

def choose_label(top3labels):
    """ 
    Randomly choose one label among the top 3 predicted labels
    """
    num_label=np.random.randint(0,3)
    theme = top3labels[num_label]
    return theme



if __name__ == "__main__":
  #Define arguments
    parser = argparse.ArgumentParser(description="Specify image path and labels")
    parser.add_argument("--image_path", type=str, help="Image path")
    parser.add_argument("--labels", type=str, help="Comma-separated labels")
    
    # Parse arguments
    args = parser.parse_args()
    labels = args.labels.split(",") if args.labels else [] 
    image_label_detector(args.image_path,labels)



