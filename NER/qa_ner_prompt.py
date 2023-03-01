
import numpy as np
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from transformers import pipeline

from torchmetrics import F1Score, Accuracy

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

nlp = pipeline('question-answering', model='CATIE-AQ/QAmembert', tokenizer='CATIE-AQ/QAmembert', topk = 10)

# Load dataset from the hub
dataset_test = load_dataset("Jean-Baptiste/wikiner_fr", split="test")
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

dataset = []

label_names = dataset_test.features["ner_tags"].feature.names
print(label_names)

acc = Accuracy(task="multiclass", num_classes=5, average="none")

for elem in dataset_test:
    tokens = elem["tokens"]
    labels = elem["ner_tags"]
    sentence = ' '.join(elem["tokens"])

    answers_loc = nlp({
        'question': "Quel est le lieu ou pays ?",
        'context': sentence
    })

    answers_per = nlp({
        'question': "Quelle est la personne ?",
        'context': sentence
    })

    answers_org = nlp({
        'question': "Quel est l'organisme ?",
        'context': sentence
    })

    probs_loc = [0.0 for t in tokens]
    for answer in answers_loc:
        if answer["score"] > 0.01:
            for word in answer["answer"].split(" "):
                if len(word) > 2 and word[0:1].isupper():
                    index = tokens.index(word)
                    probs_loc[index] = max([probs_loc[index], answer["score"]])

    probs_per = [0.0 for t in tokens]
    for answer in answers_per:
        if answer["score"] > 0.01:
            for word in answer["answer"].split(" "):
                if len(word) > 2 and word[0:1].isupper():
                    index = tokens.index(word)
                    probs_per[index] = max([probs_per[index], answer["score"]])

    probs_org = [0.0 for t in tokens]
    for answer in answers_org:
        if answer["score"] > 0.01:
            for word in answer["answer"].split(" "):
                if len(word) > 2 and word[0:1].isupper():
                    index = tokens.index(word)
                    probs_org[index] = max([probs_org[index], answer["score"]])

    preds = []
    indices = [1,2,4]
    for i, tok in enumerate(tokens):
        scores = [probs_loc[i], probs_per[i], probs_org[i]]

        max_i = argmax(scores)
        if scores[max_i] == 0:
            preds.append(0)
        else:
            preds.append(indices[max_i])

    #print(tokens)
    #print(labels)
    #print(preds)
    #print(answers_loc)
    #print(answers_per)
    #print(answers_org)


    acc.update(torch.Tensor(preds).long(), torch.Tensor(labels).long())

print(acc.compute())
