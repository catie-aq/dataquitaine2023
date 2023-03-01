
import numpy as np
import evaluate
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample

from torchmetrics import F1Score, Accuracy

# Load dataset from the hub
dataset_test = load_dataset("Jean-Baptiste/wikiner_fr", split="test")
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "camembert-base")
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-xl")

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "non",
    "oui"
]

label_names = dataset_test.features["ner_tags"].feature.names
print(label_names)

dataset = []

for elem in dataset_test:
    tokens = elem["tokens"]
    labels = elem["ner_tags"]
    sentence = ' '.join(elem["tokens"])

    for i, tok in enumerate(tokens):
        if len(tok) > 3:
            dataset.append(InputExample(
            guid = i,
            text_a = sentence,
            text_b = tok,
            label = int((labels[i] == 1))
        ))

promptTemplate = ManualTemplate(
    text = '{"placeholder": "text_a"}. Est-ce que ce le mot "{"placeholder": "text_b"}" est un lieu ? {"mask"}.',
    tokenizer = tokenizer,
)

promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "non": ["non"],
        "oui": ["oui"],
    },
    tokenizer = tokenizer,
)

promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
    plm_eval_mode=True
)

data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size = 512
)

promptModel.eval()
f1 = F1Score(task="binary")
acc = Accuracy(num_classes=2)

with torch.no_grad():
    for batch in data_loader:
        labels = torch.Tensor([dataset[i].label for i in batch["guid"]]).long()
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        f1.update(preds, labels)
        acc.update(preds, labels)

    print(f1.compute())
    print(acc.compute())
