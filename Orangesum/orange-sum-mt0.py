import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from torchmetrics import Accuracy, BLEUScore

from torchmetrics.text.rouge import ROUGEScore

dataset_test = load_dataset("orange_sum", "title", split="test")
print(dataset_test)

checkpoint = "bigscience/mt0-large"
#checkpoint = "CATIE-AQ/frenchT0"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
model.eval()

rouge = ROUGEScore()

for elem in dataset_test:
    text = elem["text"]
    inputs = tokenizer.encode("Générer un titre pour l'article suivant : " + text, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, generation_config=GenerationConfig(max_length=60))
    decoding = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoding)

    rouge.update(decoding, elem["summary"])

    #acc.update(torch.Tensor([pred]), torch.Tensor([elem["label"]]).long())

print(rouge.compute())
