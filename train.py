import argparse
import logging
from src.training_setup2 import TrainingLLM

# LOGGER CONFIGURATION ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PATH TO BUCKET ---------------------------
s3_uri = "s3://lzabar"

# POEM CHOICE ---------------------------
parser = argparse.ArgumentParser(description="Specify type of poem")
parser.add_argument(
    "--poem_type",
    type=str,
    choices=["haiku", "classic"],
    required=True,
    help="Poem type between 'haiku' and 'classic'"
)
args = parser.parse_args()

# TRAINING SETUP ---------------------------
training_setup = TrainingLLM(poem_type=args.poem_type, s3_uri=s3_uri)

# DATASET RETRIEVE ---------------------------
poems = training_setup.retrieve_dataset()

# TOKENISATION ---------------------------
tokenized_poems = training_setup.tokenize_dataset(poems)

# MODEL TRAINING ---------------------------
training_setup.train(tokenized_poems)

# SAVING THE FINETUNED MODEL ---------------------------
training_setup.save_model()

logger.info(f"Fine-tuning of {args.poem_type} model complete. Model and tokenizer saved to S3.")
