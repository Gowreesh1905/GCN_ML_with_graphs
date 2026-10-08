"""
baselines.py — Non-learned Graph Baseline

Neighbour majority vote: predict each node's label as the most common label
among its *labelled* (training) neighbours. It uses only the graph and the
training labels — no node features and no learned parameters — so it shows
how much of the task the graph structure alone can solve.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def neighbour_majority_vote(
    adjacency: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """
    Predict every node's label by majority vote over its training neighbours.

    Nodes with no labelled neighbours, or with a tied vote, are assigned the
    majority class of the training set.

    Parameters
    ----------
    adjacency : np.ndarray, shape (N, N)
        Adjacency matrix without self-loops (a node does not vote for itself).
    labels : np.ndarray, shape (N,)
        Node labels; only entries where train_mask is True are used.
    train_mask : np.ndarray, shape (N,), dtype=bool
        Nodes whose labels are visible.

    Returns
    -------
    predictions : np.ndarray, shape (N,)
    """
    train_labels = labels[train_mask]
    fallback = 1 if train_labels.mean() >= 0.5 else 0

    # Votes from labelled neighbours only: A[:, train] @ y[train]
    labelled = (adjacency[:, train_mask] > 0).astype(float)
    votes_pass = labelled @ train_labels           # labelled neighbours with label 1
    votes_total = labelled.sum(axis=1)             # labelled neighbours in total
    votes_fail = votes_total - votes_pass

    predictions = np.full(len(labels), fallback)
    predictions[votes_pass > votes_fail] = 1
    predictions[votes_fail > votes_pass] = 0
    return predictions


def evaluate_neighbour_vote(
    adjacency: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict:
    """Accuracy and F1 of the neighbour majority vote on the nodes in eval_mask."""
    preds = neighbour_majority_vote(adjacency, labels, train_mask)
    y_true, y_pred = labels[eval_mask], preds[eval_mask]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
