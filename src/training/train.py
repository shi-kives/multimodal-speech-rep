import torch
import torch.optim as optim
import torch.nn as nn
from src.models.mlp import MultiLayerPerceptron
from pathlib import Path

CONFIG = {
    'INPUT_DIM': 784,
    'HIDDEN_DIM': 128,
    'OUTPUT_DIM': 7,
    'LEARNING_RATE': 0.001
}

model = MultiLayerPerceptron(input_dim = CONFIG['INPUT_DIM'], hidden_dim = CONFIG['HIDDEN_DIM'], output_dim = CONFIG['OUTPUT_DIM'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), CONFIG['LEARNING_RATE'])

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]
emb_path = ROOT / "embeddings" / "acoustic" / "wavlm_features.pt"
embeddings = torch.load(emb_path)

inputs = embeddings[1].to(device)

labels_mapping = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'neutral': 4,
    'sadness': 5,
    'surprise': 6
}

embeddings[2] = embeddings[2].map(labels_mapping)
labels = embeddings[2]