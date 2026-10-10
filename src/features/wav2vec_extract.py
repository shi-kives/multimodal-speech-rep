from transformers import Wav2Vec2FeatureExtractor, AutoModel
import librosa
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

feature_ext = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
model = AutoModel.from_pretrained("microsoft/wavlm-base")
print("Wav2Vec model loaded.")

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

manifest_path = ROOT / "results" / "artifacts" / "train_manifest.csv"
manifest = pd.read_csv(manifest_path)
embeddings = []

for i in tqdm(manifest['Path_to_WAV'], desc = "extracting wavlm embeddings"):
    abs_path = ROOT / Path(i)
    if Path(abs_path).is_file():
        speech_array, sampling_rate = librosa.load(abs_path, sr=16000)

        inputs = feature_ext(speech_array, sampling_rate = sampling_rate, return_tensors = "pt")

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings.append(outputs.last_hidden_state.squeeze(0).cpu())

    else:
        print(f"\nfile not found. check absolute path: {abs_path}")
        continue

pooled_embeddings = torch.stack([e.mean(dim = 0) for e in embeddings])

save_path = ROOT / "embeddings" / "acoustic" / "wavlm_features.pt"
torch.save(pooled_embeddings, save_path)
print("WavLM feature extraction complete. Pooled embeddings saved at: ", save_path)