from transformers import Trainer

# create a tokenize function
def tokenize_function(dataset, tokenizer):
    poem = dataset["content"]
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        poem,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    return tokenized_inputs

class CausalLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure we set up the labels correctly for causal language modeling
        labels = inputs.get(
            "input_ids"
        ).clone()  # Set labels as input_ids for causal language modeling
        outputs = model(**inputs)
        loss = outputs.loss  # Get the loss from model outputs
        return (loss, outputs) if return_outputs else loss