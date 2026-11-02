from pathlib import Path
import torch
import mlflow
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay
from src.models.dataloader import SpeechEmbeddingsDataset, FusedDataset
from src.models.mlp import MultiLayerPerceptron
from src.models.fusion import DirectFusedMLP, ProjectionFusedMLP
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from src.training.sanity_check import sanity_check
from src.training.train import set_seed
from src.models.weighting import class_weighting

SEED = 42
set_seed(SEED)

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

BASE_MODEL_PATH = ROOT / 'results' / 'models'
BASE_EMB_PATH = ROOT / 'embeddings'
MODEL_PATHS = {
    'baseline_acoustic': BASE_MODEL_PATH / 'baselines' / 'base_acoustic.pt',
    'baseline_reference': BASE_MODEL_PATH / 'baselines' / 'base_reference.pt',
    'baseline_transcripts': BASE_MODEL_PATH / 'baselines' / 'base_transcripts.pt',
    'baseline_fused_ref': BASE_MODEL_PATH / 'baselines' / 'base_fusion_ref.pt',
    'baseline_fused_trans': BASE_MODEL_PATH / 'baselines' / 'base_fusion_trans.pt',

    'dir_fusion_ref_lr_0003': BASE_MODEL_PATH / 'tuned' / 'learning_rate' / 'dir_fusion_ref_lr_0003.pt', 
    'dir_fusion_ref_lr_0005': BASE_MODEL_PATH / 'tuned' / 'learning_rate' / 'dir_fusion_ref_lr_0005.pt',

    'dir_fusion_ref_hidden_64': BASE_MODEL_PATH / 'tuned' / 'hidden' / 'dir_fusion_ref_hidden_64.pt',

    'dir_fusion_ref_decay_e2': BASE_MODEL_PATH / 'tuned' / 'decay' / 'dir_fusion_ref_decay_e2.pt',
    'dir_fusion_ref_decay_e3': BASE_MODEL_PATH / 'tuned' / 'decay' / 'dir_fusion_ref_decay_e3.pt',

    'dir_fusion_ref_early_05_del_001': BASE_MODEL_PATH / 'tuned' / 'early_stopping' / 'dir_fusion_ref_early_05_del_001.pt',
    'dir_fusion_ref_early_05_del_0003': BASE_MODEL_PATH / 'tuned' / 'early_stopping' / 'dir_fusion_ref_early_05_del_0003.pt',

    'dir_fusion_ref_dropout_01': BASE_MODEL_PATH / 'tuned' / 'dropout' / 'dir_fusion_ref_dropout_01.pt',
    'dir_fusion_ref_dropout_03': BASE_MODEL_PATH / 'tuned' / 'dropout' / 'dir_fusion_ref_dropout_03.pt',

    'dir_fusion_ref_class_weights': BASE_MODEL_PATH / 'tuned' / 'weighting' / 'dir_fusion_ref_class_weights.pt',

    'dir_fusion_ref_25': BASE_MODEL_PATH / 'data' / 'dir_fusion_ref_25.pt',
    'dir_fusion_ref_50': BASE_MODEL_PATH / 'data' / 'dir_fusion_ref_50.pt',
    'dir_fusion_ref_75': BASE_MODEL_PATH / 'data' / 'dir_fusion_ref_75.pt',


    'proj_fusion_ref': BASE_MODEL_PATH / 'fusion' / 'proj_fusion_ref.pt',
    'proj_fusion_ref_256': BASE_MODEL_PATH / 'fusion' / 'proj_fusion_ref_256.pt',
    'proj_fusion_whisper': BASE_MODEL_PATH / 'fusion' / 'proj_fusion_whisper.pt',

    'dir_full_model': BASE_MODEL_PATH / 'ablations' / 'direct' / 'full_model.pt',
    'dir_no_decay': BASE_MODEL_PATH / 'ablations' / 'direct' / 'no_decay.pt',
    'dir_no_es': BASE_MODEL_PATH / 'ablations' / 'direct' / 'no_es.pt',
    'dir_no_weighting': BASE_MODEL_PATH / 'ablations' / 'direct' / 'no_weighting.pt',

    'proj_full_model': BASE_MODEL_PATH / 'ablations' / 'projected' / 'full_model.pt',
    'proj_no_decay': BASE_MODEL_PATH / 'ablations' / 'projected' / 'no_decay.pt',
    'proj_no_es': BASE_MODEL_PATH / 'ablations' / 'projected' / 'no_es.pt',
    'proj_no_weighting': BASE_MODEL_PATH / 'ablations' / 'projected' / 'no_weighting.pt',

    'dir_reference': BASE_MODEL_PATH / 'ablations' / 'reference' / 'direct.pt',
    'proj_reference': BASE_MODEL_PATH / 'ablations' / 'reference' / 'projected.pt',

    'dir_whisper': BASE_MODEL_PATH / 'ablations' / 'whisper' / 'direct.pt',
    'proj_whisper': BASE_MODEL_PATH / 'ablations' / 'whisper' / 'projected.pt',

}
INPUT_EMBEDDINGS = {
    'wavlm_test': BASE_EMB_PATH / 'acoustic' / 'wavlm_test.pt',
    'distilbert_reference': BASE_EMB_PATH / 'linguistic' / 'distilbert_reference' / 'distilbert_reference_test.pt',
    'distilbert_transcripts': BASE_EMB_PATH / 'linguistic' / 'distilbert_transcript' / 'distilbert_transcripts_test.pt',
    'fused_reference_test': BASE_EMB_PATH / 'fused' / 'reference' / 'fused_reference_test.pt',
    'fused_transcripts_test': BASE_EMB_PATH / 'fused' / 'transcripts' / 'fused_transcripts_test.pt'
}

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/ablations")

def run_evaluation(model_path, embeddings, modality, run_name, model_name, hidden_dim, dropout, projection_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    acoustic_embeddings = None # preventing UnboundLocalError
    linguistic_embeddings = None

    model = None
    if modality.upper() == 'UNIMODAL':
        model = MultiLayerPerceptron(input_dim = 768, hidden_dim = hidden_dim, output_dim = 7, dropout = dropout)
    elif modality.upper() == 'MULTIMODAL':
        if model_name.upper() == 'DIRECT':
            model = DirectFusedMLP(acoustic_input_dim = 768, linguistic_input_dim = 768, hidden_dim = hidden_dim, output_dim = 7, dropout = dropout)
        elif model_name.upper() == 'PROJECTION':
            model = ProjectionFusedMLP(acoustic_input_dim = 768, linguistic_input_dim = 768, projection_dim = projection_dim, hidden_dim = hidden_dim, output_dim = 7, dropout = dropout)
    
    state_dict = torch.load(model_path, map_location = device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    test_inputs = torch.load(embeddings)
    criterion = nn.CrossEntropyLoss()

    if modality.upper() == 'UNIMODAL':
        embeddings = torch.stack([i[1] for i in test_inputs])
        labels = [i[2] for i in test_inputs]

        test_dataset = SpeechEmbeddingsDataset(embeddings, labels)

    elif modality.upper() == 'MULTIMODAL':
        acoustic_embeddings = torch.stack([i[1] for i in test_inputs])
        linguistic_embeddings = torch.stack([i[2] for i in test_inputs])
        labels = [i[3] for i in test_inputs]

        test_dataset = FusedDataset(acoustic_embeddings, linguistic_embeddings, labels)

    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    class_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    uids = pd.Series([i[0] for i in test_inputs], name = "uid")
    manifest_path = ROOT / "results" / "artifacts" / "test_manifest.csv"

    '''if modality.upper() == 'UNIMODAL':
        sanity_check(embeddings, labels, uids, manifest_path, "test", modality)

    elif modality.upper() == 'MULTIMODAL':
        sanity_check(acoustic_embeddings, labels, uids, manifest_path, "test", "acoustic")
        sanity_check(linguistic_embeddings, labels, uids, manifest_path, "test", "linguistic")'''
 
    model.eval()
    with mlflow.start_run(run_name = run_name) as run:
        with torch.no_grad():
            all_preds, all_trues = [], []
            overall_loss = 0.0
            if modality.upper() == 'UNIMODAL':
                for batch_embeddings, batch_labels in test_loader:
                    batch_embeddings = batch_embeddings.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_embeddings)
                    loss = criterion(outputs, batch_labels)

                    preds = torch.argmax(outputs, dim = 1).detach().cpu().numpy()
                    
                    overall_loss += loss.item()
                    all_preds.extend(preds)
                    all_trues.extend(batch_labels.cpu().numpy())
                
            elif modality.upper() == 'MULTIMODAL':
                for batch_acoustic, batch_linguistic, batch_labels in test_loader:
                    batch_acoustic = batch_acoustic.to(device)
                    batch_linguistic = batch_linguistic.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_acoustic, batch_linguistic)
                    loss = criterion(outputs, batch_labels)

                    preds = torch.argmax(outputs, dim = 1).detach().cpu().numpy()
                    
                    overall_loss += loss.item()
                    all_preds.extend(preds)
                    all_trues.extend(batch_labels.cpu().numpy())

            avg_loss = overall_loss / len(test_loader)

            balanced_accuracy = balanced_accuracy_score(all_trues, all_preds)
            macro_f1 = f1_score(all_trues, all_preds, average='macro')

            recall_per_class = recall_score(all_trues, all_preds, labels = list(range(7)), average = None, zero_division=0)
            recall_dict = { f"test_recall_class_{i}": float(score) for i, score in enumerate(recall_per_class) }

            precision_per_class = precision_score(all_trues, all_preds, labels = list(range(7)), average=None, zero_division=0)
            precision_dict = { f"test_precision_class_{i}": float(score) for i, score in enumerate(precision_per_class)}

            conf_matrix = confusion_matrix(all_trues, all_preds, labels = list(range(7)))
            disp_cm = pd.DataFrame(conf_matrix, index = class_labels, columns = class_labels)
            cm_png = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
            fig, ax = plt.subplots(figsize=(5, 5))
            cm_png.plot(ax=ax)


            print(f"\n---- inference results for {run_name} ----\n")
            print(f"average loss: {avg_loss:.4f}")
            print(f"balanced accuracy: {balanced_accuracy:.4f}")
            print(f"macro f1 score: {macro_f1:.4f}")
            print(f"per-class recall score: {recall_per_class}")
            print(f"per-class precision score: {precision_per_class}")
            print(f"confusion matrix:\n{disp_cm}")

            mlflow.log_metric("average_test_loss", avg_loss)
            mlflow.log_metric("balanced_accuracy", balanced_accuracy)
            mlflow.log_metric("macro_f1", macro_f1)
            mlflow.log_metrics(recall_dict)
            mlflow.log_metrics(precision_dict)
            mlflow.log_figure(fig, f"test_confusion_matrix.png")

            plt.close(fig)

if __name__ == '__main__':
    run_evaluation(MODEL_PATHS['acoustic_lr_001'], INPUT_EMBEDDINGS['wavlm_test'], 'acoustic only with lr 0.01', 'UNIMODAL', 'testing acoustic only with lr 0.01')