import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torchmetrics import Accuracy

dataset_test = load_dataset("Jean-Baptiste/wikiner_fr", split="test")
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

checkpoint = "bigscience/mt0-xl"
#checkpoint = "t5-base"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
model.eval()

acc = Accuracy(task="multiclass", num_classes=5, average="none")

for elem in dataset_test:
    tokens = elem["tokens"]
    labels = elem["ner_tags"]
    sentence = ' '.join(elem["tokens"])

    preds = []
    for i, tok in enumerate(tokens):
        if len(tok) > 3:
            inputs = tokenizer.encode(sentence + ". Is the word " + tok + " a location in the previous text ?", return_tensors="pt").to("cuda")
            outputs = model.generate(inputs)
            decoding = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Yes" in decoding or "yes" in decoding:
                preds.append(1)
                continue


            inputs = tokenizer.encode(sentence + ". Is the word " + tok + " a person's name in the previous text ?", return_tensors="pt").to("cuda")
            outputs = model.generate(inputs)
            decoding = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Yes" in decoding or "yes" in decoding:
                preds.append(2)
                continue

            inputs = tokenizer.encode(sentence + ". Is the word " + tok + " a organisation's name in the previous text ?", return_tensors="pt").to("cuda")
            outputs = model.generate(inputs)
            decoding = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Yes" in decoding or "yes" in decoding:
                preds.append(4)
                continue

            '''inputs = tokenizer.encode(sentence + ". Is the word " + tok + " a miscellaneous entity in the previous text ?", return_tensors="pt").to("cuda")
            outputs = model.generate(inputs)
            decoding = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Yes" in decoding or "yes" in decoding:
                preds.append(3)
                continue'''

            preds.append(0)
        else:
            preds.append(0)

    acc.update(torch.Tensor(preds).long(), torch.Tensor(labels).long())

print(acc.compute())
