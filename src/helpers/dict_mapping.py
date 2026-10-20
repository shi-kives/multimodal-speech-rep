from pathlib import Path
import torch
import pandas as pd

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

acoustic_inputs = torch.load(ROOT / "embeddings" / "acoustic" / "wavlm_test.pt")
linguistic_inputs = torch.load(ROOT / "embeddings" / "linguistic" / "distilbert_reference" / "distilbert_reference_test.pt")

acoustic_df = pd.DataFrame(acoustic_inputs, columns = ["Utterance_ID", "Acoustic_Embeddings", "Label"])
linguistic_df = pd.DataFrame(linguistic_inputs, columns = ["Utterance_ID", "Linguistic_Embeddings", "Label"])

fused = pd.merge(acoustic_df, linguistic_df, how='inner', on='Utterance_ID')
fused = fused.drop_duplicates()

keep = ['Utterance_ID', 'Acoustic_Embeddings', 'Linguistic_Embeddings', 'Label_x']
save_inputs = fused[keep].values.tolist()

save_path = ROOT / "embeddings" / "fused" / "reference" / "fused_reference_test.pt"
save_path.parent.mkdir(parents = True, exist_ok = True)

torch.save(save_inputs, save_path)
print(f"saved acoustic-linguistic merged embeddings of length {len(fused)} at {save_path}")