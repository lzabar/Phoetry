import os
import time

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, DataCollatorForLanguageModeling

from src.set_llm import tokenize_function, CausalLMTrainer
from src.my_log import get_logger

logger = get_logger(name=__name__)


class PoemModelTrainer():
    """
    A class to get GPT2 model from huggingface and fine tuned them for our purposes
    """
    name: str
    """
    The name of our model in training.
    Should be : gpt2_xx_y--y
    xx : language used (2 digits)
    y--y : type of poems to generate (n digits)
    """
    local_dir: str
    """
    Local path where to store the model before stocking it on S3 SSP Cloud
    """
    model_name: str
    """
    The name of the hugging face model. Used to call the model from hugging face
    """
    model: GPT2LMHeadModel
    """
    Type of model used
    """
    tokenizer: GPT2Tokenizer
    """
    tokenizer used for GPT2 model
    """

    def __init__(self, name: str, model_name: str):
        """
        Initialize the model from hugging face
        """
        start = time.time()
        logger.info(f"Initialization : Start --> {name}")
        self.name = name

        # CONFIGURATION OF ENVIRONMENT
        self.local_dir = './training_model/' + self.name + '/'    # where the model will be stored localy
        os.makedirs(self.local_dir, exist_ok=True)

        logger.info(f"local dir for training set to {self.local_dir}")
        self.model_name = model_name

        try:        # TRY TO IMPORT MODEL FROM HUGGINGFACE
            logger.info("Download : Trying --> model")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.local_dir)
            self.model = GPT2LMHeadModel.from_pretrained(self.local_dir)
            
            # ADD TOKEN PADDING IF MISSING
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

            end = time.time()
            delta = round((end - start), 1)

            logger.info(f'Initialization : Complete in {delta} sec --> Model is running')

        except Exception as e:
            raise RuntimeError(f"Error loading model from {self.local_dir}: {e}")

    def save_model(self):
        """
        Simple method to save the model in local_dir
        """
        self.model.save_pretrained(self.local_dir)
        self.tokenizer.save_pretrained(self.local_dir)
        logger.info(f"Saving : Success --> {self.name}")

    def load_dataset(
        self,
        URL_dataset: str,
        from_s3: bool = True
        ):
        """
        A method to import a dataset from S3.
        Can take as input types :
        - Dataset Class from HuggingFace
        - .parquet
        - .csv
        """
        if from_s3:
            extension = str(URL_dataset).split('.')[-1]

            if extension = "parquet":
                dataset.from_parquet()

        try:
            dataset = load_dataset(URL_dataset)
            logger.info("Download : Success --> Dataset as Hugging Face Dataset")
        except Exception:
            

            if extension == ".parquet":

        


# DATA IMPORT ---------------------------
print("Loading haiku dataset...")
dataset = load_dataset("statworx/haiku")




# DATA TOKENISATION ---------------------------
print("Tokenizing dataset...")
poem_column = "text"
tokenized_dataset = dataset["train"].map(
    lambda x: tokenize_function(x, tokenizer, poem_column),
    batched=True
)


# DATA PREPARATION ---------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    return_tensors='pt',
    mlm=False
)


# TRAINING ARGUMENTS ---------------------------
training_args = TrainingArguments(
    output_dir=SAVE_DIR,                        # output directory for model checkpoints
    do_eval=False,                              # evaluate every few steps
    learning_rate=5e-5,                         # learning rate for optimizer
    per_device_train_batch_size=2,              # batch size for training
    num_train_epochs=2,                         # number of training epochs
    save_steps=10_000,                          # save checkpoints every 10,000 steps
    save_total_limit=5,                         # only keep the 2 most recent checkpoints
    logging_dir="./logs",                       # directory to save logs
    logging_steps=500,                          # log every 500 steps
    report_to=None,
    no_cuda=False,                              # If False, forces GPU usage (set True if you want CPU)
    fp16=True                                   # Use mixed precision for speedup (if using GPU
)


# TRAIN ---------------------------
print("Starting training...")
trainer = CausalLMTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()


