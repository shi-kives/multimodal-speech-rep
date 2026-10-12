from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

manifest_path = ROOT / "results" / "artifacts" / "train_manifest.csv"
manifest = pd.read_csv(manifest_path)

embeddings = []

def mean_pooling(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float() # to add an extra dimension to input attention mask
    
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim = 1) # attention aware masked pooling

    sum_mask = torch.clamp(input_mask_expanded.sum(dim = 1), min = 1e-9) # total non-zero masks

    return sum_embeddings / sum_mask


model.eval() # to disable droput

for row in tqdm(manifest.itertuples(index = True), total = len(manifest), desc = "creating distilbert embeddings"):

    abs_path = ROOT / Path(row.Path_to_WAV)
    if Path(abs_path).is_file():
        if len(row.Reference_Transcript) != 0:
            inputs = tokenizer(row.Reference_Transcript, padding = True, truncation = True, return_tensors = "pt")

            with torch.no_grad():
                outputs = model(**inputs)

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            pooled_embeddings = mean_pooling(token_embeddings, attention_mask)
            embeddings.append([row.SNo, pooled_embeddings])
        else:
            print(f"error! file dia{row.Dialogue_ID}_utt{row.Utterance_ID} has no transcript.")
            continue

    else:
        print(f"error! file dia{row.Dialogue_ID}_utt{row.Utterance_ID} not found")
        continue

save_path = Path(ROOT / "embeddings" / "linguistic" / "distilbert_reference.pt")
save_path.parent.mkdir(parents = True, exist_ok = True)

torch.save(embeddings, save_path)
print("distilbert generated embeddings for reference transcripts saved at: ", save_path)