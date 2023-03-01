import numpy as np
import evaluate
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding

## Load dataset from the hub
dataset_train = load_dataset("flue", "CLS", split="train")
dataset_test = load_dataset("flue", "CLS", split="test")
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

## Load the model

#model_name = "flaubert/flaubert_base_cased"
#model_name = "camembert-base"
#model_name = "camembert/camembert-large"
model_name ="flaubert/flaubert_large_cased"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

## Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=480, truncation=True)

dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_test = dataset_test.map(tokenize_function, batched=True)

dataset_train = dataset_train.shuffle().select(range(100))

dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

wandb.init(project="allocine_sentiment_analysis",
           name="allocine-camembert-finetuning")

training_args = TrainingArguments(output_dir="allocine_sentiment_finetuning", evaluation_strategy="no", save_strategy="no",
                                  num_train_epochs=20, per_device_train_batch_size=8, per_device_eval_batch_size=8, learning_rate=1e-5)
data_collator = DataCollatorWithPadding(tokenizer)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    # compute the accuracy
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
