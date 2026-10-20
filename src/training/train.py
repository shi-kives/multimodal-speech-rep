import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.mlp import MultiLayerPerceptron
from src.models.dataloader import SpeechEmbeddingsDataset, FusedDataset
from pathlib import Path
from tqdm import tqdm
import mlflow
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.training.sanity_check import sanity_check
from src.models.fusion import DirectFusedMLP
from src.models.early_stopping import EarlyStopping
import random
import numpy as np

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/tuning-experiments")

CONFIG = {

    'ACOUSTIC_INPUT_DIM': 768,
    'LINGUISTIC_INPUT_DIM': 768,
    'HIDDEN_DIM': 128,
    'OUTPUT_DIM': 7,

    'EPOCHS': 30,
    'BATCH_SIZE': 100,

    'LEARNING_RATE': 0.001,

    'ACOUSTIC_EMB_TRAIN_PATH': ROOT / "embeddings" / "acoustic" / "wavlm_train.pt",
    'ACOUSTIC_EMB_VAL_PATH': ROOT / "embeddings" / "acoustic" / "wavlm_val.pt",

    'LINGUISTIC_EMB_REF_TRAIN_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_reference" / "distilbert_reference_train.pt",
    'LINGUISTIC_EMB_REF_VAL_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_reference" / "distilbert_reference_val.pt",

    'LINGUISTIC_EMB_TRANS_TRAIN_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_transcript" / "distilbert_transcripts_train.pt",
    'LINGUISTIC_EMB_TRANS_VAL_PATH': ROOT / "embeddings" / "linguistic" / "distilbert_transcript" / "distilbert_transcripts_val.pt",

    'FUSION_REF_TRAIN_PATH': ROOT / "embeddings" / "fused" / "reference" / "fused_reference_train.pt",
    'FUSION_REF_VAL_PATH': ROOT / "embeddings" / "fused" / "reference" / "fused_reference_val.pt",

    'FUSION_TRANS_TRAIN_PATH': ROOT / "embeddings" / "fused" / "transcripts" / "fused_transcripts_train.pt",
    'FUSION_TRANS_VAL_PATH': ROOT / "embeddings" / "fused" / "transcripts" / "fused_transcripts_val.pt",

    'SAVE_PATH_BASE': ROOT / "results" / "models"
}

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(path, mode, modality, modality_type, dropout, hidden_dim, batch_size):
    model = None
    if modality_type.upper() == 'UNIMODAL':
        inputs = torch.load(path, map_location='cpu')

        embeddings = torch.stack([i[1] for i in inputs]).to(device)
        labels = torch.tensor([i[2] for i in inputs], dtype = torch.long).to(device)
        uids = pd.Series([i[0] for i in inputs], name = "uid")

        dataset = SpeechEmbeddingsDataset(embeddings, labels)

        generator = torch.Generator()
        generator.manual_seed(SEED)

        if mode.lower() == 'train':
            loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, generator=generator)
        elif mode.lower() == 'val':
            loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, generator=generator)

        print(f"{modality} {mode} dataset and dataloader initialized....".upper())

        if mode.lower() == 'train':
            model = MultiLayerPerceptron(input_dim = 768, hidden_dim = hidden_dim, output_dim = CONFIG['OUTPUT_DIM'], dropout = dropout)
            print("completed MLP....")

        return embeddings, labels, uids, loader, model

    elif modality_type.upper() == 'MULTIMODAL':
        inputs = torch.load(path, map_location='cpu')

        uids = pd.Series([i[0] for i in inputs], name = "uid")
        acoustic_embeddings = torch.stack([i[1] for i in inputs]).to(device)
        linguistic_embeddings = torch.stack([i[2] for i in inputs]).to(device)
        labels = torch.tensor([i[3] for i in inputs], dtype = torch.long).to(device)

        dataset = FusedDataset(acoustic_embeddings, linguistic_embeddings, labels)

        generator = torch.Generator()
        generator.manual_seed(SEED)
        
        if mode.lower() == 'train':
            loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, generator=generator)
        elif mode.lower() == 'val':
            loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, generator=generator)

        if mode.lower() == 'train':
            model = DirectFusedMLP(acoustic_input_dim = CONFIG["ACOUSTIC_INPUT_DIM"], linguistic_input_dim = CONFIG["LINGUISTIC_INPUT_DIM"], hidden_dim = hidden_dim, output_dim = CONFIG["OUTPUT_DIM"], dropout = dropout)
            print("completed MLP...")

        return acoustic_embeddings, linguistic_embeddings, labels, uids, loader, model

def train(train_path, val_path, modality, modality_type, r_name, save_path, lr, batch_size, no_epochs, hidden_dim, dropout, weight_decay, patience, min_delta):
    
    if modality_type.upper() == 'UNIMODAL':
        train_embeddings, train_labels, train_uids, train_loader, model = setup(train_path, mode = 'train', modality = modality, modality_type = modality_type, dropout = dropout, hidden_dim = hidden_dim, batch_size = batch_size)

        model = model.to(device)

        val_embeddings, val_labels, val_uids, val_loader, _ = setup(val_path, mode = 'val', modality = modality, modality_type = modality_type, dropout = None, hidden_dim = None, batch_size = batch_size)

    elif modality_type.upper() == 'MULTIMODAL':
        train_aco_emb, train_ling_emb, train_labels, train_uids, train_loader, model = setup(train_path, mode = 'train', modality = modality, modality_type = modality_type, dropout = dropout, hidden_dim = hidden_dim, batch_size = batch_size)

        model = model.to(device)

        val_aco_emb, val_ling_emb, val_labels, val_uids, val_loader, _ = setup(val_path, mode = 'val', modality = modality, modality_type = modality_type, dropout = None, hidden_dim = None, batch_size = batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    train_manifest_path = ROOT / "results" / "artifacts" / "train_manifest.csv"
    val_manifest_path =  ROOT / "results" / "artifacts" / "val_manifest.csv"

    if modality_type.upper() == 'UNIMODAL':
        sanity_check(train_embeddings, train_labels, train_uids, train_manifest_path, "train", modality)

        sanity_check(val_embeddings, val_labels, val_uids, val_manifest_path, 'val', modality)

    elif modality_type.upper() == 'MULTIMODAL':
        sanity_check(train_aco_emb, train_labels, train_uids, train_manifest_path, "train", "acoustic")
        sanity_check(val_aco_emb, val_labels, val_uids, val_manifest_path, "val", "acoustic")

        sanity_check(train_ling_emb, train_labels, train_uids, train_manifest_path, "train", "linguistic")
        sanity_check(val_ling_emb, val_labels, val_uids, val_manifest_path, "val", "linguistic")

    print("setup complete, starting training")
    print(r_name.upper())
    
    best_val_ba = 0.0
    best_val_mf1 = 0.0

    class_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    early_stopping = EarlyStopping(patience, min_delta)

    with mlflow.start_run(run_name = r_name) as run:
        mlflow.log_params({
            'ACOUSTIC_INPUT_DIM': 768,
            'LINGUISTIC_INPUT_DIM': 768,
            'HIDDEN_DIM': hidden_dim,
            'OUTPUT_DIM': 7,
            'LEARNING_RATE': lr,
            'EPOCHS': no_epochs,
            'BATCH_SIZE': batch_size,
            'DROPOUT': dropout,
            'WEIGHT_DECAY': weight_decay,
            'MODALITY': modality
        })

        for epoch in tqdm(range(no_epochs), desc = f"baseline fusion training: "):
            # --- training ---
            all_preds, all_trues = [], []
            train_epoch_loss = 0.0
            model.train()

            if modality_type.upper() == 'UNIMODAL':
                for batch_embeddings, batch_labels in train_loader:
                    batch_embeddings = batch_embeddings.to(device)
                    batch_labels = batch_labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(batch_embeddings)
                    loss = criterion(outputs, batch_labels)

                    loss.backward()
                    optimizer.step()

                    train_epoch_loss += loss.item()

                    preds = torch.argmax(outputs, dim = 1).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_trues.extend(batch_labels.cpu().numpy())

            elif modality_type.upper() == 'MULTIMODAL':
                for batch_acoustic, batch_linguistic, batch_labels in train_loader:
                    batch_acoustic = batch_acoustic.to(device)
                    batch_linguistic = batch_linguistic.to(device)
                    batch_labels = batch_labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(batch_acoustic, batch_linguistic)
                    loss = criterion(outputs, batch_labels)

                    loss.backward()
                    optimizer.step()

                    train_epoch_loss += loss.item()

                    preds = torch.argmax(outputs, dim = 1).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_trues.extend(batch_labels.cpu().numpy())

            avg_loss = train_epoch_loss / len(train_loader)
            accuracy = balanced_accuracy_score(all_trues, all_preds)
            macrof1 = f1_score(all_trues, all_preds, average='macro')
            print(f"\n{epoch + 1}/{CONFIG['EPOCHS']} - TRAINING - loss: {avg_loss:.4f}, balanced accuracy: {accuracy:.4f}, macro f1: {macrof1:.4f}\n")

            mlflow.log_metric("avg_loss", avg_loss, step = epoch)
            mlflow.log_metric("bal_accuracy", accuracy, step = epoch)
            mlflow.log_metric("macro_f1", macrof1, step = epoch)

            # --- validation ---
            model.eval()

            with torch.no_grad():
                val_epoch_loss = 0.0
                val_preds, val_trues = [], []

                if modality_type.upper() == 'UNIMODAL':
                    for batch_embeddings, batch_labels in val_loader:
                        batch_embeddings = batch_embeddings.to(device)
                        batch_labels = batch_labels.to(device)

                        outputs = model(batch_embeddings)
                        loss = criterion(outputs, batch_labels)

                        val_epoch_loss += loss.item()

                        preds = torch.argmax(outputs, dim = 1).detach().cpu().numpy()
                        val_preds.extend(preds)
                        val_trues.extend(batch_labels.cpu().numpy())
                    
                elif modality_type.upper() == 'MULTIMODAL':
                    for batch_acoustic, batch_linguistic, batch_labels in val_loader:
                        batch_acoustic = batch_acoustic.to(device)
                        batch_linguistic = batch_linguistic.to(device)
                        batch_labels = batch_labels.to(device)

                        outputs = model(batch_acoustic, batch_linguistic)
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

                cm_png = ConfusionMatrixDisplay(confusion_matrix=val_conf_mat, display_labels=class_labels)
                fig, ax = plt.subplots(figsize=(5, 5))
                cm_png.plot(ax=ax)

                print(f"\nepoch {epoch + 1}/{CONFIG['EPOCHS']} - VALIDATION - average loss: {avg_val_loss:.4f}, balanced accuracy {val_ba:.4f}, macro f1-score {val_mf1:.4f}, per-class recall: {val_recall}, confusion matrix: \n{disp_cm}\n")

                mlflow.log_metric("avg_val_loss", avg_val_loss, step = epoch)
                mlflow.log_metric("val_bal_acc", val_ba, step = epoch)
                mlflow.log_metric("val_mf1", val_mf1, step = epoch)
                mlflow.log_metrics(recall_dict, step = epoch)
                mlflow.log_figure(fig, f"val_cm_{epoch}.png")
                
                plt.close(fig)

                early_stopping(val_mf1, model)
                if early_stopping.early_stop:
                    print("early stopping triggered! halting loop")
                    early_stopping.load_best_model(model)
                    torch.save(model.state_dict(), save_path)
                    print(f"model with best balanced macro-f1: {best_val_mf1:.4f}, balanced accuracy: {val_ba:.4f} saved at {save_path}")
                    mlflow.log_artifact(save_path)
                    break

                if val_ba > best_val_ba:
                    best_val_ba = val_ba
                
                if val_mf1 > best_val_mf1:
                    best_val_mf1 = val_mf1
                    save_path.parent.mkdir(parents = True, exist_ok = True)
                    torch.save(model.state_dict(), save_path)
                    print(f"model with best balanced macro-f1: {best_val_mf1:.4f}, balanced accuracy: {val_ba:.4f} saved at {save_path}")
                    mlflow.log_artifact(save_path)
                
                mlflow.log_metric("best_val_ba", best_val_ba, step = epoch)
                mlflow.log_metric("best_val_mf1", best_val_mf1, step = epoch)

if __name__ == '__main__':
    save_path = CONFIG['SAVE_PATH_BASE'] / 'tuned' / 'fusion_ref_lr_0005.pt'
    
    train(CONFIG['FUSION_REF_TRAIN_PATH'], CONFIG['FUSION_REF_VAL_PATH'], 'fusion reference', 'MULTIMODAL', "testing lr = 0.005 on fusion reference", save_path, lr = 0.005, batch_size = 100, no_epochs = 30, hidden_dim = 128, dropout = 0.0, weight_decay = 0.0, patience = 30.0, min_delta = 0.0)
