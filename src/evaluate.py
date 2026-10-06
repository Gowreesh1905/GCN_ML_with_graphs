"""
evaluate.py — Evaluation Metrics for Node Classification

Computes accuracy, precision, recall, F1 score, and confusion matrix
on masked subsets (train/val/test).
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model: nn.Module,
    A_norm: torch.Tensor,
    X: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Dict:
    """
    Evaluate a trained model on a masked subset of nodes.

    Parameters
    ----------
    model : nn.Module
        Trained GCN or MLP.
    A_norm : torch.Tensor, shape (N, N)
        Normalized adjacency matrix.
    X : torch.Tensor, shape (N, F)
        Node features.
    labels : torch.Tensor, shape (N,)
        Ground truth labels.
    mask : torch.Tensor, shape (N,), dtype=bool
        Mask indicating which nodes to evaluate.

    Returns
    -------
    metrics : dict
        Keys: accuracy, precision, recall, f1, confusion_matrix,
              predictions, probabilities
    """
    model.eval()
    with torch.no_grad():
        logits = model(A_norm, X)

        # Get predictions and probabilities
        masked_logits = logits[mask]
        masked_labels = labels[mask]

        # Softmax converts logits to probabilities
        # P_ic = exp(z_ic) / Σ_c' exp(z_ic')
        # 0 ≤ P_ic ≤ 1 and Σ_c P_ic = 1
        probs = torch.softmax(masked_logits, dim=1)
        preds = masked_logits.argmax(dim=1)

    y_true = masked_labels.numpy()
    y_pred = preds.numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # labels=[0, 1] keeps the matrix 2 × 2 even if a split contains
        # (or the model predicts) only one class.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": probs.numpy().tolist(),
        "y_true": y_true.tolist(),
    }

    return metrics


def compute_homophily(
    adjacency: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """
    Compute edge homophily ratio.

    H = #{(i,j) ∈ E : y_i = y_j} / |E|

    Homophily measures how often connected nodes share the same label.
    - H ≈ 1: strong homophily (same-class nodes cluster together)
    - H ≈ 0.5: random / no homophily
    - H ≈ 0: strong heterophily (opposite-class nodes connect)

    GCNs tend to perform better on graphs with higher homophily because
    neighbourhood aggregation reinforces class-consistent signals.

    Parameters
    ----------
    adjacency : np.ndarray, shape (N, N)
        Original adjacency matrix (without self-loops).
    labels : np.ndarray, shape (N,)
        Node labels.

    Returns
    -------
    result : dict
        Homophily statistics.
    """
    N = adjacency.shape[0]
    total_edges = 0
    same_label_edges = 0

    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] > 0:
                total_edges += 1
                if labels[i] == labels[j]:
                    same_label_edges += 1

    cross_label_edges = total_edges - same_label_edges
    homophily_ratio = same_label_edges / total_edges if total_edges > 0 else 0.0

    return {
        "total_edges": total_edges,
        "same_label_edges": same_label_edges,
        "cross_label_edges": cross_label_edges,
        "homophily_ratio": homophily_ratio,
    }
