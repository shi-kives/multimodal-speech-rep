from pathlib import Path
import torch

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

labels_mapping = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'neutral': 4,
    'sadness': 5,
    'surprise': 6
}

def perform_mapping(emb_path: str, save_path: str):
    embeddings = torch.load(emb_path)

    labels_col = [i[2] for i in embeddings]
    labels = [labels_mapping[i] for i in labels_col]
    for i in range(0, len(embeddings)):
        embeddings[i][2] = labels[i]

    torch.save(embeddings, save_path)
    print(f"successfully saved mapped labels to {save_path}")


path = ROOT / "embeddings" / "linguistic" / "distilbert_transcript" / "distilbert_transcripts_test.pt"

perform_mapping(path, path)