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
import matplotlib.pyplot as plt
from src.training.sanity_check import sanity_check
from src.models.fusion import DirectFusedMLP
import json

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/meld-multimodal")

CONFIG = {
    'EXP_NAME': 'direct fusion baseline',
    'ACOUSTIC_INPUT_DIM': 768,
    'LINGUISTIC_INPUT_DIM': 768,
    'HIDDEN_DIM': 128,
    'OUTPUT_DIM': 7,
    'LEARNING_RATE': 0.001,
    'EPOCHS': 30,
    'BATCH_SIZE': 100,

    'ACOUSTIC_EMB_TRAIN_PATH': ROOT / "embeddings" / "acoustic" / "wavlm_train.pt",
    'ACOUSTIC_EMB_VAL_PATH': ROOT / "embeddings" / "acoustic" / "wavlm_val.pt",

    'LINGUISTIC_EMB_TRAIN_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_reference" / "distilbert_reference_train.pt",

    'LINGUISTIC_EMB_VAL_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_reference" / "distilbert_reference_val.pt",

    'SAVE_PATH': ROOT / "results" / "models" / "base_transcript.pt"
}

CONFIG_PATH = ROOT / "results" / "configs" / "baseline_fusion.json"
with open(CONFIG_PATH, 'r') as f:
    EXP_CONFIG = json.load(f)
EXP_CONFIG = dict(EXP_CONFIG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(path, mode, modality, modality_type, path_acoustic = None, path_linguistic = None):
    if modality_type == 'UNIMODAL':
        inputs = torch.load(path)

        embeddings = torch.stack([i[1] for i in inputs]).to(device)
        labels = torch.tensor([i[2] for i in inputs], dtype = torch.long).to(device)

        dataset = SpeechEmbeddingsDataset(embeddings, labels)
        loader = DataLoader(dataset, batch_size = CONFIG['BATCH_SIZE'], shuffle = True)

        uids = pd.Series([i[0] for i in inputs], name = "uid")

        print(f"{modality} {mode} dataset and dataloader initialized....".upper())

        model = MultiLayerPerceptron(input_dim = 768, hidden_dim = 128, output_dim = 7)

        print("completed MLP....")

        return inputs, embeddings, labels, uids, dataset, loader, model

    elif modality_type == 'MULTIMODAL':
        acoustic_inputs = torch.load(path_acoustic)
        linguistic_inputs = torch.load(path_linguistic)

        embeddings = 

def train():

    

    model = DirectFusedMLP(acoustic_input_dim = CONFIG['ACOUSTIC_INPUT_DIM'], linguistic_input_dim= CONFIG['LINGUISTIC_INPUT_DIM'], hidden_dim = CONFIG['HIDDEN_DIM'], output_dim = CONFIG['OUTPUT_DIM'], dropout = 0.0)
    print("completed MLP....")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), CONFIG['LEARNING_RATE'])

    train_manifest_path = ROOT / "results" / "artifacts" / "train_manifest.csv"
    val_manifest_path =  ROOT / "results" / "artifacts" / "val_manifest.csv"

    sanity_check(acoustic_train_inputs, acoustic_train_labels, acoustic_train_uids, train_manifest_path, "train", "acoustic")
    sanity_check(acoustic_val_inputs, acoustic_val_labels, acoustic_val_uids, val_manifest_path, "val", "acoustic")

    print("setup complete, starting training")
    print(CONFIG['EXP_NAME'].upper())
    
    best_val_ba = 0.0
    best_val_mf1 = 0.0

    class_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    with mlflow.start_run(run_name = "baseline-direct-fusion") as run:
        mlflow.log_params({
            'ACOUSTIC_INPUT_DIM': 768,
            'LINGUISTIC_INPUT_DIM': 768,
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
            print(f"\n{epoch + 1}/{CONFIG['EPOCHS']} - TRAINING - loss: {avg_loss:.4f}, balanced accuracy: {accuracy:.4f}\n")

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