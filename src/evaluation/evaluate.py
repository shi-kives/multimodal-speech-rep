from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_recall_curve, roc_auc_score
import scipy.stats as stats
import torch

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]
BASE_PATH = ROOT / 'results' / 'models'

MODEL_PATHS = {
    'baseline_acoustic': BASE_PATH / 'base_acoustic.pt',
    'baseline_reference': BASE_PATH / 'base_reference.pt',
    'baseline_transcripts': BASE_PATH / 'base_transcripts.pt'
}

def run_evaluation(model_path):
    model = torch.load(model_path)

    