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
from src.set_llm import (
    tokenize_function,
    CausalLMTrainer
)
from transformers import Trainer

class CausalLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure we set up the labels correctly for causal language modeling
        labels = inputs["input_ids"].clone()  # Set labels as input_ids for causal language modeling
        inputs["labels"] = labels
        outputs = model(**inputs)
        loss = outputs.loss  # Get the loss from model outputs
        return (loss, outputs) if return_outputs else loss

class TrainingLLM:
    """
    A class to train the model from retrieving the dataset to fine-tune gpt-2
    """

    def __init__(self,poem_type):
        """
        Initialize model, tokenizer and define poem type (poem_type £ ["haiku", "classic"])
        """

        self.poem_type=poem_type
        # ENVIRONMENT CONFIGURATION ---------------------------
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.base_dir, "trained_model", "poet-gpt2",self.poem_type)
        self.log_dir = os.path.join(self.base_dir, "logs",self.poem_type)

        #MODEL INITIALIZATION
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        # ADD TOKEN PADDING IF MISSING ---------------------------
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def retrieve_datasets(self):
        """
        To import data
        """
        if self.poem_type=="haiku":
            print("Loading haiku dataset...")
            dataset = load_dataset("statworx/haiku")
        else:
            print("Loading classic poems datasets...")
            #   dataset1
            foundation_poems = load_dataset("shahules786/PoetryFoundationData")
            #   dataset2
            mexwell_poems = pd.read_csv(f"{kagglehub.dataset_download("mexwell/poem-dataset")}/final_df_emotions(remove-bias).csv")
            mexwell_poems["author"] = "unknown"
            mexwell_poems = mexwell_poems[["label", "poem content", "author", "type", "age"]]
            mexwell_poems.columns = ["poem name", "content", "author", "type", "age"]
            mexwell_poems
            #   dataset3
            abiemo_poems = pd.read_csv(f"{kagglehub.dataset_download("pkkazipeta143/americanbritishindian-emotion-poetry-dataset")}/ABIEMO_2334.csv")
            abiemo_poems["author"] = "unknown"
            abiemo_poems["age"] = "unknown"
            abiemo_poems = abiemo_poems[["Emotions", "poems", "author", "class", "age"]]
            abiemo_poems.columns = ["poem name", "content", "author", "type", "age"]
            abiemo_poems
            #   concatenate datasets
            dataset = DatasetDict()
            dataset["train"] = concatenate_datasets(
                [
                    foundation_poems["train"],
                    Dataset.from_pandas(mexwell_poems),
                    Dataset.from_pandas(abiemo_poems)
                ]
                )
        return dataset 

        
        def tokenize_dataset(self, dataset):
            if self.poem_type="haiku":
                poems_column="text"
            else:
                poems_column="content"
            def tokenize_function(self, tokenizer, poems_column):
                poem = dataset[poems_column]
                tokenizer.truncation_side = "left"
                tokenized_inputs = tokenizer(
                    poem,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                    )
                return tokenized_inputs
            tokenized_dataset = dataset["train"].map(
                lambda x: tokenize_function(x, self.tokenizer, poems_column),
                batched=True)
            return tokenized_dataset

        def training_preparation(self):
            # DATA PREPARATION ---------------------------
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                return_tensors='pt',
                mlm=False
                )
                
            # TRAINING ARGUMENTS ---------------------------
            training_args = TrainingArguments(
                output_dir=self.save_dir,                        # output directory for model checkpoints
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
            return data_collator,training_args

        def train_model(self, tokenized_dataset):
            data_collator,training_args= TrainingLLM(self.poem_type).training_preparation()
            print("Starting training...")
            trainer = CausalLMTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator)
            trainer.train()
            print("Training complete!")
        
        def save_trained_model(self): #modifier pour sauvegarder dans s3
            print("Saving model to:", self.save_dir)
            self.model.save_pretrained(SAVE_DIR)
            self.tokenizer.save_pretrained(SAVE_DIR)
            print("Saving complete!")


            

