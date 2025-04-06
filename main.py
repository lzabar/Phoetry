# Install required packages
# pip install -r requirements.txt

import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.image_recognition import SetClipModel
from src.poem_generator import poem_generator


model_path = "trained_model/poet-gpt2"

if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
    print("Model not found, download in progress...")
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Model saved!")
else:
    print("Model already available, loading in progress..")

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)


# Proposed labels
labels = [
    "tree", "flower", "sunset", "sunrise", "cloud", "mountain", "beach", "river", "lake",
    "waterfall", "forest", "grassland", "desert", "rain", "snow", "road", "traffic jam", "hill",
    "valley", "cave", "farm", "garden", "coastline", "field", "pond", "sky", "animal", "insect",
    "fungi", "leaf", "pebble", "stone", "dog", "cat", "bird", "butterfly",
    "bee", "stars", "moon", "sun"
]


def generate_poem_from_picture(image_path, labels):
    """
    Args:
        image_path (str): path to picture.
        labels (list of str): List of possible thematic labels to be detected in the image.

    Returns:
        str: generate a poem according to the visual content of the picture.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The specified image does not exist : {image_path}")

    theme = SetClipModel().image_label_detector(image_path, labels)
    poem = poem_generator(
        model_path=model_path, theme=theme, max_length=200, temperature=0.5,
        top_k=60, top_p=0.9, repetition_penalty=1.5
    )
    return poem


if __name__ == "__main__":
    image_path = os.path.join("images", "sunset.jpg")
    print(generate_poem_from_picture(image_path, labels))
