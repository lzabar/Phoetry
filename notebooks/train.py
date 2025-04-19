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

# Load dataset
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
poems = DatasetDict()
poems["train"] = concatenate_datasets(
    [
     foundation_poems["train"],
     Dataset.from_pandas(mexwell_poems),
     Dataset.from_pandas(abiemo_poems)
    ]
)

# Initialize model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


tokenized_poems = poems["train"].map(lambda dataset: tokenize_function(dataset, tokenizer), batched=True)
tokenized_poems

# Format dataset
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    return_tensors='pt',
    mlm=False
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./trained_model",  # output directory for model checkpoints
    do_eval=False,  # evaluate every few steps
    learning_rate=5e-5,  # learning rate for optimizer
    per_device_train_batch_size=2,  # batch size for training
    num_train_epochs=2,  # number of training epochs
    save_steps=10_000,  # save checkpoints every 10,000 steps
    save_total_limit=5,  # only keep the 2 most recent checkpoints
    logging_dir="./logs",  # directory to save logs
    logging_steps=500,  # log every 500 steps
    report_to=None,
    no_cuda=False,  # If False, forces GPU usage (set True if you want CPU)
    fp16=True,  # Use mixed precision for speedup (if using GPU
)


# Set up the Trainer
trainer = CausalLMTrainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=tokenized_poems,          # training data    
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("trained_model/poet-gpt2")
tokenizer.save_pretrained("trained_model/poet-gpt2")
