import os
import json
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.image_recognition import SetClipModel
from src.poem_generator import poem_generator


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)


model_path = "trained_model/poet-gpt2"
if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
    logging.info("Model not found, download in progress...")
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logging.info("Model saved!")
else:
    logging.info("Model already available, loading in progress..")

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)


def load_labels_from_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['labels']
    except FileNotFoundError:
        logging.error(f"JSON file not found : {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error("JSON file decoding error.")
        raise


labels = load_labels_from_json("labels.json")

set_clip_model = SetClipModel()


def generate_poem_from_picture(image_path, labels):
    """
    Args:
        image_path (str): path to picture.
        labels (list of str): List of possible thematic labels to be detected in the image.

    Returns:
        str: generate a poem according to the visual content of the picture.
    """
    if not os.path.exists(image_path):
        logging.error(f"The specified image does not exist: {image_path}")
        raise FileNotFoundError(f"The specified image does not exist: {image_path}")

    try:
        top_labels = set_clip_model.image_label_detector(image_path, labels)
        theme = top_labels[0]
        logging.info(f"Chosen theme: {theme}")

        poem = poem_generator(
            model_path=model_path, theme=theme, max_length=200, temperature=0.5,
            top_k=60, top_p=0.9, repetition_penalty=1.5
        )
        return poem

    except Exception as e:
        logging.error(f"Error during poem generation: {str(e)}")
        raise


if __name__ == "__main__":
    image_path = os.path.join("images", "sunset.jpg")

    try:
        poem = generate_poem_from_picture(image_path, labels)
        print(poem)
    except Exception as e:
        logging.error(f"Failed to generate poem: {str(e)}")
