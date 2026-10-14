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
    'EPOCHS': 30,
    'BATCH_SIZE': 100,
    'TRAIN_EMBEDDINGS_PATH': ROOT / "embeddings" / "acoustic" / "wavlm_train.pt",
    'VAL_EMBEDDINGS_PATH': ROOT / "embeddings" / "acoustic" / "wavlm_val.pt",
    'SAVE_PATH': ROOT / "results" / "models" / "base_acoustic.pt"
}

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_embeddings = torch.load(CONFIG['TRAIN_EMBEDDINGS_PATH'])

    train_inputs = torch.stack([i[1] for i in train_embeddings]).to(device)
    train_labels = torch.tensor([i[2] for i in train_embeddings], dtype = torch.long).to(device)

    train_dataset = SpeechEmbeddingsDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size = CONFIG['BATCH_SIZE'], shuffle = True)

    print("Train Dataset and DataLoader initialized....")

    val_embeddings = torch.load(CONFIG['VAL_EMBEDDINGS_PATH'])
    val_inputs = torch.stack([i[1] for i in val_embeddings]).to(device)
    val_labels = torch.tensor([i[2] for i in val_embeddings], dtype = torch.long).to(device)

    val_dataset = SpeechEmbeddingsDataset(val_inputs, val_labels)
    val_loader = DataLoader(val_dataset, batch_size = CONFIG['BATCH_SIZE'], shuffle = False)

    model = MultiLayerPerceptron(input_dim = CONFIG['INPUT_DIM'], hidden_dim = CONFIG['HIDDEN_DIM'], output_dim = CONFIG['OUTPUT_DIM'])
    print("completed MLP....")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), CONFIG['LEARNING_RATE'])


    print("setup complete, starting training")
    print("acoustic-only baseline training".upper())
    
    best_val_ba = 0.0
    best_val_mf1 = 0.0

    with mlflow.start_run() as run:
        mlflow.log_params({
            'INPUT_DIM': 768,
            'HIDDEN_DIM': 128,
            'OUTPUT_DIM': 7,
            'LEARNING_RATE': 0.001,
            'EPOCHS': 30,
            'BATCH_SIZE': 100,
        })

        for epoch in tqdm(range(CONFIG['EPOCHS']), desc = f"baseline acoustic training: "):
            # --- training ---
            all_preds, all_trues = [], []
            train_epoch_loss = 0.0
            model.train()

            for batch_inputs, batch_labels in train_loader:
                optimizer.zero_grad()

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)

                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()

                preds = torch.argmax(outputs, dim = 1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(batch_labels.cpu().numpy())

            avg_loss = train_epoch_loss / len(train_loader)
            accuracy = balanced_accuracy_score(all_trues, all_preds)
            print(f"\n{epoch}/{CONFIG['EPOCHS']} - TRAINING - loss: {avg_loss:.4f}, balanced accuracy: {accuracy:.4f}\n")

            mlflow.log_metric("avg_loss", avg_loss, step = epoch)
            mlflow.log_metric("bal_accuracy", accuracy, step = epoch)

            # --- validation ---
            model.eval()

            with torch.no_grad():
                val_epoch_loss = 0.0
                val_preds, val_trues = [], []

                for batch_inputs, batch_labels in val_loader:

                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_labels)

                    val_epoch_loss += loss.item()

                    preds = torch.argmax(outputs, dim = 1).detach().cpu().numpy()
                    val_preds.extend(preds)
                    val_trues.extend(batch_labels.cpu().numpy())

                avg_val_loss = val_epoch_loss / len(val_loader)
                val_ba = balanced_accuracy_score(val_trues, val_preds)
                val_mf1 = f1_score(val_trues, val_preds, average='macro')
                val_conf_mat = confusion_matrix(val_trues, val_preds)
                print(f"\nepoch {epoch}/{CONFIG['EPOCHS']} - VALIDATION - average loss: {avg_val_loss:.4f}, balanced accuracy {val_ba:.4f}, macro f1-score {val_mf1:.4f},confusion matrix: \n{val_conf_mat}\n")

                mlflow.log_metric("avg_val_loss", avg_val_loss, step = epoch)
                mlflow.log_metric("val_bal_acc", val_ba, step = epoch)
                mlflow.log_metric("val_mf1", val_mf1, step = epoch)

                if val_ba > best_val_ba:
                    best_val_ba = val_ba
                
                if val_mf1 > best_val_mf1:
                    best_val_mf1 = val_mf1
                    CONFIG['SAVE_PATH'].parent.mkdir(parents = True, exist_ok = True)
                    torch.save(model.state_dict(), CONFIG['SAVE_PATH'])
                    print(f"model with best balanced macro-f1: {best_val_mf1:.4f}, balanced accuracy: {best_val_ba:.4f} saved at {CONFIG['SAVE_PATH']}")
                    mlflow.log_artifact(CONFIG['SAVE_PATH'])

if __name__ == '__main__':
    train()