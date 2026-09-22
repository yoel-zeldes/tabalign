import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_roc_auc(y_true, y_probs):
    """Calculates the appropriate metric based on the number of classes.
      - Binary classification: ROC AUC (higher is better)
      - Multiclass classification: -log_loss (higher is better)
    """
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_probs[:, 1])
    else:
        return -log_loss(y_true, y_probs)
