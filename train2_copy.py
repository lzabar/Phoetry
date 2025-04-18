import os
import requests
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, DataCollatorForLanguageModeling
from src.set_llm import tokenize_function, CausalLMTrainer




# ENVIRONMENT CONFIGURATION ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "poubelle", "poet-gpt2")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# DATA IMPORT ---------------------------
print("Loading haiku dataset...")
dataset = load_dataset("statworx/haiku")

print(type(dataset))
for split, dset in dataset.items():
    print(f'split : {split}')
    print(f'dset : {dset}')
    direr = "./poubelle/" + str(split) + ".parquet"
    dset.to_parquet(direr)

#dataset.to_parquet(path_or_buf="./dataset_temp")


URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Datasets/train.parquet"

resp = requests.get(URL)

if resp.status_code == 200:
    print(f'{resp.status_code}')

    os.makedirs("./poub", exist_ok=True)
    with open("./poub/train.parquet", "wb") as f:
        print("in")

        f.write(resp.content)

ds = DatasetDict.from_parquet({'train': "./poub/train.parquet"})


# TOKEN AND MODEL INITIALISATION ---------------------------
print("Initializing GPT-2 model and tokenizer...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# ADD TOKEN PADDING IF MISSING ---------------------------
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


# DATA TOKENISATION ---------------------------
print("Tokenizing dataset...")
poem_column = "text"
tokenized_dataset = dataset["train"].map(
    lambda x: tokenize_function(x, tokenizer, poem_column),
    batched=True
)


# DATA PREPARATION ---------------------------
print("data prepa")
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


# SAVING MODEL ---------------------------
print("Training complete! Saving model to:", SAVE_DIR)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
