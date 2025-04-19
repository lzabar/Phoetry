import logging
import joblib
import s3fs
import kagglehub
import pandas as pd
from datasets import (
    load_dataset,
    DatasetDict,
    concatenate_datasets,
    Dataset
)
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer


class CausalLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Ensure we set up the labels correctly for causal language modeling
        """
        labels = inputs["input_ids"].clone()            # Set labels as input_ids for causal language modeling
        inputs["labels"] = labels
        outputs = model(**inputs)
        loss = outputs.loss                             # Get the loss from model outputs
        return (loss, outputs) if return_outputs else loss


class TrainingLLM:
    """
    A class to train the model from retrieving the dataset to fine-tune gpt-2
    """
    def __init__(self, poem_type, s3_uri=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.poem_type = poem_type
        self.s3_uri = s3_uri.rstrip("/") if s3_uri else None
        self.fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})
        self.model, self.tokenizer = self._load_gpt2()

        # add token padding if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _s3_exists(self, path):
        return self.fs.exists(path)

    def _load_from_s3(self, path):
        with self.fs.open(path, "rb") as f:
            return joblib.load(f)

    def _save_to_s3(self, obj, path):
        with self.fs.open(path, "wb") as f:
            joblib.dump(obj, f)

    def _load_gpt2(self):
        gpt2_s3_path = f"{self.s3_uri}/gpt2"

        if self._s3_exists(gpt2_s3_path):
            self.logger.info(f"Loading model and tokenizer from {gpt2_s3_path} on S3...")
            model = GPT2LMHeadModel.from_pretrained(gpt2_s3_path)
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2_s3_path)
        else:
            self.logger.info("Model and tokenizer not found on S3. Downloading from HuggingFace...")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            try:
                model.save_pretrained(gpt2_s3_path)
                tokenizer.save_pretrained(gpt2_s3_path)
                self.logger.info(f"Model and tokenizer saved to {gpt2_s3_path} on S3.")
            except Exception as e:
                self.logger.error(f"Error saving model and tokenizer to S3: {e}")
    
        return model, tokenizer

    def retrieve_dataset(self):
        dataset_s3_path = f"{self.s3_uri}/datasets/{self.poem_type}.joblib"

        if self._s3_exists(dataset_s3_path):
            self.logger.info("Loading dataset from S3...")
            return self._load_from_s3(dataset_s3_path)

        self.logger.info("Dataset not found on S3. Downloading...")
        if self.poem_type == "haiku":
            self.logger.info("Loading haiku dataset...")
            dataset = load_dataset("statworx/haiku")
        else:
            self.logger.info("Loading classic poems datasets...")
            foundation_poems = load_dataset("shahules786/PoetryFoundationData")
            mexwell_poems = pd.read_csv(f"{kagglehub.dataset_download('mexwell/poem-dataset')}/final_df_emotions(remove-bias).csv")
            mexwell_poems["author"] = "unknown"
            mexwell_poems = mexwell_poems[["label", "poem content", "author", "type", "age"]]
            mexwell_poems.columns = ["poem name", "content", "author", "type", "age"]
            abiemo_poems = pd.read_csv(f"{kagglehub.dataset_download('pkkazipeta143/americanbritishindian-emotion-poetry-dataset')}/ABIEMO_2334.csv")
            abiemo_poems["author"] = "unknown"
            abiemo_poems["age"] = "unknown"
            abiemo_poems = abiemo_poems[["Emotions", "poems", "author", "class", "age"]]
            abiemo_poems.columns = ["poem name", "content", "author", "type", "age"]
            dataset = DatasetDict()
            dataset["train"] = concatenate_datasets(
                [
                    foundation_poems["train"],
                    Dataset.from_pandas(mexwell_poems),
                    Dataset.from_pandas(abiemo_poems)
                ]
            )

        self._save_to_s3(dataset, dataset_s3_path)
        self.logger.info("Dataset saved to S3.")
        return dataset

    def tokenize_dataset(self, dataset):
        poem_column = "text" if self.poem_type == "haiku" else "content"

        def tokenize_function(batch):
            return self.tokenizer(batch[poem_column], truncation=True, padding="max_length", max_length=512)

        self.logger.info("Tokenizing dataset...")
        return dataset["train"].map(tokenize_function, batched=True)

    def training_preparation(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        training_args = TrainingArguments(
            output_dir=f"{self.s3_uri}/tmp/{self.poem_type}",           # output directory for model checkpoints
            do_eval=False,                                              # evaluate every few steps
            learning_rate=5e-5,                                         # learning rate for optimizer
            per_device_train_batch_size=2,                              # batch size for training
            num_train_epochs=2,                                         # number of training epochs
            save_steps=5000,                                            # save checkpoints every 10,000 steps
            save_total_limit=2,                                         # only keep the 2 most recent checkpoints
            logging_dir=f"{self.s3_uri}/logs/{self.poem_type}",         # directory to save logs
            logging_steps=500,                                          # log every 500 steps
            report_to=None,
            no_cuda=False,                                              # If False, forces GPU usage (set True if you want CPU)
            fp16=True                                                   # Use mixed precision for speedup (if using GPU)
        )

        return training_args, data_collator

    def train(self, tokenized_dataset):
        training_args, data_collator = self.training_preparation()
        trainer = CausalLMTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        self.logger.info("Starting training...")
        trainer.train()
        self.logger.info("Training complete.")

    def save_model(self):
        model_s3_dir = f"{self.s3_uri}/models_finetuned/{self.poem_type}"
        self.logger.info(f"Saving model to {model_s3_dir} on S3...")
        self.model.save_pretrained(model_s3_dir)
        self.tokenizer.save_pretrained(model_s3_dir)
        self.logger.info("Model and tokenizer saved to S3.")
