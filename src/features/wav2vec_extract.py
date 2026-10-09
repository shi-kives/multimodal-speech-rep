from transformers import Wav2Vec2FeatureExtractor, AutoModel
import librosa
import torch
import pandas as pd
from pathlib import Path

feature_ext = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
model = AutoModel.from_pretrained("microsoft/wavlm-base", device_map = "auto")
print("Wav2Vec model loaded.")

manifest = pd.read_csv('results/artifacts/train_manifest.csv')
embeddings = []
for i in manifest['Path_to_MP4']:
    if Path(i).is_file():
        speech_array, sampling_rate = librosa.load(i, sr=16000)

        inputs = feature_ext(speech_array, sampling_rate = sampling_rate, return_tensors = "pt")

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings.append(outputs.last_hidden_state.squeeze(0).cpu())

    else:
        print(f"file {i} not found! skipping.")
        continue

pooled_embeddings = torch.stack([e.mean(dim = 0) for e in embeddings])