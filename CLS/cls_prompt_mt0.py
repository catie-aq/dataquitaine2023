import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration
from datasets import load_dataset
from torchmetrics import Accuracy

dataset_test = load_dataset("flue", "CLS", split="test")
print(dataset_test)
dataset_test = dataset_test.shuffle(seed=32).select(range(500))

checkpoint = "bigscience/mt0-xl"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
model.eval()

acc = Accuracy(num_classes=1)

for elem in dataset_test:
    inputs = tokenizer.encode(elem["text"] + "Is this review positive or negative ?​​​", return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    decoding = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "positive" in decoding or "Positive" in decoding:
        pred = 1
    else:
        pred = 0

    acc.update(torch.Tensor([pred]), torch.Tensor([elem["label"]]).long())

print(acc.compute())
