import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.mlp import MultiLayerPerceptron
from src.models.dataloader import SpeechEmbeddingsDataset
from pathlib import Path
from tqdm import tqdm
import mlflow
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, ConfusionMatrixDisplay
from src.helpers.mapping import labels_mapping
import matplotlib.pyplot as plt
from collections import Counter

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/baseline-linguistic-transcripts-only")

CONFIG = {
    'INPUT_DIM': 768,
    'HIDDEN_DIM': 128,
    'OUTPUT_DIM': 7,
    'LEARNING_RATE': 0.001,
    'EPOCHS': 30,
    'BATCH_SIZE': 100,
    'TRAIN_EMBEDDINGS_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_transcript" / "distilbert_transcripts_train.pt",
    'VAL_EMBEDDINGS_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_transcript" / "distilbert_transcripts_val.pt",
    'SAVE_PATH': ROOT / "results" / "models" / "base_transcript.pt"
}

def sanity_check(train_embeddings, val_embeddings, train_labels, val_labels, train_uids, val_uids):
    print(f"number of samples in train: {len(train_labels)}")
    print(f"number of samples in val: {len(val_labels)}")

    print(f"number of unique utterance ids in train: {train_uids.nunique()}")
    print(f"number of unique utterance ids in val: {val_uids.nunique()}")

    print(f"shape of train embeddings: {train_embeddings.shape}")
    print(f"shape of val embeddings: {val_embeddings.shape}")

    print(f"labels mapping: {labels_mapping}")

    print("emotion distribution: ")
    train_manifest_path = ROOT / "results" / "artifacts" / "train_manifest.csv"
    val_manifest_path = ROOT / "results" / "artifacts" / "val_manifest.csv"

    train = pd.read_csv(train_manifest_path)
    val = pd.read_csv(val_manifest_path)
    train_emo = dict(Counter(train['Label']).most_common())
    val_emo = dict(Counter(val['Label']).most_common())

    print(f"train set emotions distribution: {train_emo}")
    print(f"val set emotions distribution: {val_emo}")


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_embeddings = torch.load(CONFIG['TRAIN_EMBEDDINGS_PATH'])

    train_inputs = torch.stack([i[1] for i in train_embeddings]).to(device)
    train_labels = torch.tensor([i[2] for i in train_embeddings], dtype = torch.long).to(device)

    train_dataset = SpeechEmbeddingsDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size = CONFIG['BATCH_SIZE'], shuffle = True)

    print("train dataset and dataloader initialized....")

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

    train_uids = pd.Series([i[0] for i in train_embeddings], name = "uid")
    val_uids = pd.Series([i[0] for i in val_embeddings], name = "uid")

    sanity_check(train_inputs, val_inputs, train_labels, val_labels, train_uids, val_uids)

    print("setup complete, starting training")
    print("linguistic transcripts-only baseline training".upper())
    
    best_val_ba = 0.0
    best_val_mf1 = 0.0

    class_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    with mlflow.start_run() as run:
        mlflow.log_params({
            'INPUT_DIM': 768,
            'HIDDEN_DIM': 128,
            'OUTPUT_DIM': 7,
            'LEARNING_RATE': 0.001,
            'EPOCHS': 30,
            'BATCH_SIZE': 100,
        })

        for epoch in tqdm(range(CONFIG['EPOCHS']), desc = f"baseline transcripts training: "):
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

                val_recall = recall_score(val_trues, val_preds, labels=list(range(CONFIG['OUTPUT_DIM'])) ,average=None, zero_division=0)
                recall_dict = { f"val_recall_class_{i}": float(score) for i, score in enumerate(val_recall) }

                val_conf_mat = confusion_matrix(val_trues, val_preds)
                disp_cm = pd.DataFrame(val_conf_mat, index = class_labels, columns = class_labels)

                cm_png = ConfusionMatrixDisplay(confusion_matrix=val_conf_mat)
                fig, ax = plt.subplots(figsize=(5, 5))
                cm_png.plot(ax=ax)

                print(f"\nepoch {epoch}/{CONFIG['EPOCHS']} - VALIDATION - average loss: {avg_val_loss:.4f}, balanced accuracy {val_ba:.4f}, macro f1-score {val_mf1:.4f}, per-class recall: {val_recall}, confusion matrix: \n{disp_cm}\n")

                mlflow.log_metric("avg_val_loss", avg_val_loss, step = epoch)
                mlflow.log_metric("val_bal_acc", val_ba, step = epoch)
                mlflow.log_metric("val_mf1", val_mf1, step = epoch)
                mlflow.log_metrics(recall_dict, step = epoch)
                mlflow.log_figure(fig, f"val_cm_{epoch}.png")
                
                plt.close(fig)

                if val_ba > best_val_ba:
                    best_val_ba = val_ba
                
                if val_mf1 > best_val_mf1:
                    best_val_mf1 = val_mf1
                    CONFIG['SAVE_PATH'].parent.mkdir(parents = True, exist_ok = True)
                    torch.save(model.state_dict(), CONFIG['SAVE_PATH'])
                    print(f"model with best balanced macro-f1: {best_val_mf1:.4f}, balanced accuracy: {val_ba:.4f} saved at {CONFIG['SAVE_PATH']}")
                    mlflow.log_artifact(CONFIG['SAVE_PATH'])

if __name__ == '__main__':
    train()