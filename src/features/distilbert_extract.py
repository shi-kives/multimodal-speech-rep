from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

mode = "TRANSCRIPTS"

def mean_pooling(embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float() # to add an extra dimension to input attention mask
        
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim = 1) # attention aware masked pooling

        sum_mask = torch.clamp(input_mask_expanded.sum(dim = 1), min = 1e-9) # total non-zero masks

        return (sum_embeddings / sum_mask).squeeze(0)

def get_embeddings(manifest: pd.DataFrame, mode: str):
    embeddings = []

    model.eval() # to disable droput

    for row in tqdm(manifest.itertuples(index = True), total = len(manifest), desc = f"creating distilbert embeddings for {mode}"):
        value = 0

        if mode == "TRANSCRIPTS":
            value = 1
        if mode == "REFERENCE":
            abs_path = ROOT / Path(row.Path_to_WAV)
            value = Path(abs_path).is_file()

        if value:
            if(pd.isna(row.Whisper_Transcript) != 1):
                transcript = ""
                if mode == "TRANSCRIPTS":
                    transcript = row.Whisper_Transcript.strip()
                elif mode == "REFERENCE":
                    transcript = row.Reference_Transcript.strip()

                inputs = tokenizer(transcript, padding = True, truncation = True, return_tensors = "pt")

                with torch.no_grad():
                    outputs = model(**inputs)

                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                pooled_embeddings = mean_pooling(token_embeddings, attention_mask)
                embeddings.append([row.SNo, pooled_embeddings, row.Label])
            else:
                print(f"error! file dia{row.Dialogue_ID}_utt{row.Utterance_ID} has no transcript.")
                continue
        else:
            print(f"\nerror! file dia{row.Dialogue_ID}_utt{row.Utterance_ID} not found")
            continue

    return embeddings


if mode == "REFERENCE":
    manifest_path = ROOT / "results" / "artifacts" / "train_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    embeddings = get_embeddings(manifest, mode = "REFERENCE")

    save_path = Path(ROOT / "embeddings" / "linguistic" / "distilbert_reference.pt")
    save_path.parent.mkdir(parents = True, exist_ok = True)

    torch.save(embeddings, save_path)
    print("distilbert generated embeddings for reference transcripts saved at: ", save_path)

if mode == "TRANSCRIPTS":
    transcript_path = ROOT / "results" / "artifacts" / "whisper_transcripts.csv"

    transcripts = pd.read_csv(transcript_path)
    embeddings = get_embeddings(transcripts, mode = "TRANSCRIPTS")

    save_path = ROOT / "embeddings" / "linguistic" / "distilbert_transcripts.pt"
    save_path.parent.mkdir(parents = True, exist_ok = True)

    torch.save(embeddings, save_path)
    print("distilbert generated embeddings for whisper transcripts saved at: ", save_path)   