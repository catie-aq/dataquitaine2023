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
dataset_test = load_dataset("flue", "CLS", split="test")
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [
    InputExample(
        guid = i,
        text_a = dataset_test[i]["text"],
        label = dataset_test[i]["label"]
    ) for i in range(len(dataset_test))
]

#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "camembert-base")
#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "camembert/camembert-large")
plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "moussaKam/barthez")
#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "flaubert/flaubert_base_cased")
#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "flaubert/flaubert_large_cased")
#plm, tokenizer, model_config, WrapperClass = load_plm("t5", "bigscience/mt0-base")

promptTemplate = ManualTemplate(
    text = '{"placeholder": "text_a"}. Cet avis est-il positif ou négatif ? {"mask"}.',
    tokenizer = tokenizer,
)

promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["négatif", "mauvais", "horrible", "médiocre", "nul"],
        "positive": ["positif", "bon", "exceptionnel", "fabuleux", "correct"],
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
    batch_size = 512,
    max_seq_length=512, decoder_max_length=8
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
