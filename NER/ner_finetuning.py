import numpy as np
import evaluate
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from torchmetrics import Accuracy

# Load dataset from the hub
dataset_train = load_dataset("Jean-Baptiste/wikiner_fr", split="train")
dataset_test = load_dataset("Jean-Baptiste/wikiner_fr", split="test")
print(dataset_train)

dataset_train = dataset_train.shuffle().select(range(10))
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

label_names = dataset_train.features["ner_tags"].feature.names
print(label_names)

#model_name = "camembert-base"
#model_name = "flaubert/flaubert_base_cased"
#model_name = "flaubert/flaubert_large_cased"
model_name = "camembert/camembert-large"
#model_name = "moussaKam/barthez"
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

dataset_train = dataset_train.map(tokenize_and_align_labels, batched=True)
dataset_test = dataset_test.map(tokenize_and_align_labels, batched=True)

dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

wandb.init(project="ner_finetuning",
           name="ner-camembert-finetuning")

training_args = TrainingArguments(output_dir="ner_camembert_finetuning", evaluation_strategy="no", save_strategy="no",
                                  num_train_epochs=100, per_device_train_batch_size=16, per_device_eval_batch_size=128, learning_rate=1e-5)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

acc = Accuracy(task="multiclass", num_classes=5, average="none")
def compute_metrics_torchmetrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    predictions[predictions == 3] = 0

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[l for l in label if l != -100] for label in labels]
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    for i, preds in enumerate(true_predictions):
        acc.update(torch.Tensor(preds).long(), torch.Tensor(true_labels[i]).long())

    all_metrics = acc.compute()
    print(all_metrics)

    return {}

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics_torchmetrics
)

trainer.train()
trainer.evaluate()
