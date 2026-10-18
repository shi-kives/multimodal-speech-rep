from pathlib import Path
import torch
import pandas as pd

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

acoustic_inputs = torch.load(ROOT / "embeddings" / "acoustic" / "wavlm_val.pt")
linguistic_inputs = torch.load(ROOT / "embeddings" / "linguistic" / "distilbert_transcript" / "distilbert_transcripts_val.pt")

acoustic_df = pd.DataFrame(acoustic_inputs, columns = ["Utterance_ID", "Embeddings", "Label"])
linguistic_df = pd.DataFrame(linguistic_inputs, columns = ["Utterance_ID", "Embeddings", "Label"])

fused = pd.merge(acoustic_df, linguistic_df, how='inner', on='Utterance_ID')
fused = fused.drop_duplicates()

save_path = ROOT / "results" / "artifacts" / "fused" / "fused_transcript" / "acou_ling_trans_val.csv"
save_path.parent.mkdir(parents = True, exist_ok = True)
fused.to_csv(save_path, index = False)
print(f"saved acoustic-linguistic merged csv of length {len(fused)} at {save_path}")