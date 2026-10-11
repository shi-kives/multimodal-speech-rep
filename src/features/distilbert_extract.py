from transformers import AutoTokenizer, AutoModel
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval() # to disable droput

text = "this is a test drive."
inputs = tokenizer(text, return_tensors = "pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state

print("shape of last layer: ", last_hidden_state.shape)