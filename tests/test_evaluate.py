"""
test_evaluate.py — Tests for Evaluation Metrics

Tests:
- Confusion matrix is always 2 × 2, even for single-class subsets
- Homophily computed from the adjacency matches the graph statistics
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from src.evaluate import compute_homophily, evaluate_model
from src.gcn import GCN
from src.graph_data import build_dataset
from src.normalization import normalize_adjacency_torch


@pytest.fixture
def dataset():
    return build_dataset(seed=42)


class AlwaysPass(nn.Module):
    """Stub model that predicts class 1 (Pass) for every node."""

    def forward(self, A_norm, X):
        return torch.tensor([[0.0, 1.0]]).repeat(X.shape[0], 1)


class TestConfusionMatrix:

    def test_confusion_matrix_is_2x2(self, dataset):
        A_norm = normalize_adjacency_torch(dataset["adjacency"])
        model = GCN(n_layers=2)
        metrics = evaluate_model(
            model, A_norm, dataset["t_features"],
            dataset["t_labels"], dataset["t_test_mask"],
        )
        assert np.array(metrics["confusion_matrix"]).shape == (2, 2)

    def test_confusion_matrix_2x2_for_single_class_mask(self, dataset):
        """
        Only Pass nodes, all predicted Pass: just one class appears in both
        y_true and y_pred, which must still give a 2 × 2 matrix.
        """
        A_norm = normalize_adjacency_torch(dataset["adjacency"])
        model = AlwaysPass()
        pass_only = torch.tensor(dataset["labels"] == 1)
        metrics = evaluate_model(
            model, A_norm, dataset["t_features"],
            dataset["t_labels"], pass_only,
        )
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (2, 2)
        assert cm[0].sum() == 0, "No Fail nodes in the mask, so row 0 must be empty."
        assert cm.sum() == pass_only.sum().item()


class TestHomophily:

    def test_homophily_matches_graph_stats(self, dataset):
        h = compute_homophily(dataset["adjacency"], dataset["labels"])
        assert h["total_edges"] == dataset["stats"]["num_edges"]
        assert h["homophily_ratio"] == pytest.approx(dataset["stats"]["homophily_ratio"])
