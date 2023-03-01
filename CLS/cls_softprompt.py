import numpy as np
import evaluate
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate, PrefixTuningTemplate, PtuningTemplate
from openprompt.prompts import ManualVerbalizer, AutomaticVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample

from torchmetrics import F1Score, Accuracy

# Load dataset from the hub
dataset_train = load_dataset("flue", "CLS", split="train")
dataset_test = load_dataset("flue", "CLS", split="test")
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]

dataset_train = [
    InputExample(
        guid = i,
        text_a = dataset_train[i]["text"],
        label = dataset_train[i]["label"]
    ) for i in range(10)
]

dataset_test = [
    InputExample(
        guid = i,
        text_a = dataset_test[i]["text"],
        label = dataset_test[i]["label"]
    ) for i in range(len(dataset_test))
]

#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "camembert-base")
plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "camembert/camembert-large")
#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "flaubert/flaubert_base_cased")
#plm, tokenizer, model_config, WrapperClass = load_plm("automlm", "flaubert/flaubert_large_cased")


promptTemplate = PtuningTemplate(
    model = plm,
    text = '{"placeholder": "text_a"}. {"mask"}.',
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

'''
promptVerbalizer = AutomaticVerbalizer(
    classes = classes,
    tokenizer = tokenizer,

)
'''

promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
    freeze_plm=False,
    plm_eval_mode=False
).cuda()

data_loader_train = PromptDataLoader(
    dataset = dataset_train,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size = 16
)

data_loader_test = PromptDataLoader(
    dataset = dataset_test,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size = 16
)

# Training

loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in promptModel.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in promptModel.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    {'params': [p for n,p in promptModel.template.named_parameters() if "raw_embedding" not in n]}
]

optimizer1 = torch.optim.AdamW(optimizer_grouped_parameters1, lr=1e-6)
optimizer2 = torch.optim.AdamW(optimizer_grouped_parameters2, lr=1e-5)

for epoch in range(20):
    tot_loss = 0
    for step, inputs in enumerate(data_loader_train):
        inputs = inputs.cuda()
        logits = promptModel(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        print(tot_loss)

    #promptVerbalizer.optimize_to_initialize()

# Evaluation
promptModel.eval()
f1 = F1Score(task="binary")
acc = Accuracy(num_classes=2)

with torch.no_grad():
    for batch in data_loader_test:
        batch = batch.cuda()
        labels = torch.Tensor([dataset_test[i].label for i in batch["guid"]]).long()
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1).cpu()
        f1.update(preds, labels)
        acc.update(preds, labels)

    print(f1.compute())
    print(acc.compute())
