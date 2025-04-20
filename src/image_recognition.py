import torch
import argparse
import numpy as np
import matplotlib.image as mpimg
from transformers import CLIPProcessor, CLIPModel


class SetClipModel:
    """
    A class to set the model in charge of the
    recognition of images.
    This class is based on transformers
    """
    model_id : str
    """The id of the model used"""
    processor : CLIPProcessor
    """The processor needed for the model, set with model_id"""
    model : CLIPModel
    """The model used based on the model_id, set with model_id"""
    device : str
    """Type of devide on with we used the model processor"""

    def __init__(
        self,
        model_id : str = "openai/clip-vit-base-patch32"
        ):
        """Initialize the model thanks to model_id"""
        self.model_id = model_id
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Clip Model {model_id} initialized and running")


    def find_word(self, image, labels):
        """
        
        """
        label_tokens = self.processor(
            text=labels,
            padding=True,
            images=None,
            return_tensors="pt"
            ).to(self.model.device)

        # encode tokens to sentence embeddings
        label_emb = self.model.get_text_features(**label_tokens)

        # detach from pytorch gradient computation
        label_emb = label_emb.detach().cpu().numpy()

        processed_image = self.processor(  # Process image
            text=None,
            images=image,
            return_tensors="pt"
            )["pixel_values"].to(self.model.device)
            
        image_emb = (
            self.model.get_image_features(processed_image).detach().cpu().numpy()
        )  # embedding

         
        scores = np.dot(
            image_emb, 
            label_emb.T
            )  # predicted lables scores (among the 39 original labels)

        top3_scores_indexes = list(np.argsort(scores)[0][-3:,][::-1])
        top3labels = np.array(labels)[top3_scores_indexes].tolist()

        return top3labels[0]

