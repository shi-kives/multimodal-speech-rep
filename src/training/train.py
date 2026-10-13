import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.mlp import MultiLayerPerceptron
from src.models.dataloader import SpeechEmbeddingsDataset
from pathlib import Path
from tqdm import tqdm
import mlflow
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/baseline-acoustic-only")

CONFIG = {
    'INPUT_DIM': 768,
    'HIDDEN_DIM': 128,
    'OUTPUT_DIM': 7,
    'LEARNING_RATE': 0.001,
    'EPOCHS': 20,
    'BATCH_SIZE': 100,
    'EMBEDDINGS_PATH': ROOT / "embeddings" / "acoustic" / "wavlm_train.pt",
    'SAVE_PATH': ROOT / "results" / "models" / "best_baseline_acoustic.pt"
}

def train():

    inputs = torch.stack([i[1] for i in embeddings])
    labels = torch.tensor([i[2] for i in embeddings], dtype = torch.long)

    dataset = SpeechEmbeddingsDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size = CONFIG['BATCH_SIZE'], shuffle = True)
    print("Dataset and DataLoader initialized....")

    model = MultiLayerPerceptron(input_dim = CONFIG['INPUT_DIM'], hidden_dim = CONFIG['HIDDEN_DIM'], output_dim = CONFIG['OUTPUT_DIM'])
    print("completed MLP....")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), CONFIG['LEARNING_RATE'])

    embeddings = torch.load(CONFIG['EMBEDDINGS_PATH'])

    print("setup complete, starting training")
    with mlflow.start_run() as run:
        mlflow.log_params(CONFIG)
        best_acc = 0.0

        for epoch in tqdm(range(CONFIG['EPOCHS']), desc = f"acoustic-only baseline training"):
            model.train()
            for 
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        print(f"{epoch}/{CONFIG['EPOCHS']} - loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")
        mlflow.log_metric("train_loss", loss.item(), step = epoch)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), CONFIG['SAVE_PATH'])
            print(f"model with best balanced accuracy: {best_acc:.4f} saved at {CONFIG['SAVE_PATH']}")

if __name__ == '__main__':
    train()
