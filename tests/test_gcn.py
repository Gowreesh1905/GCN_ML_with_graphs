"""
test_gcn.py — Tests for GCN Layer and Model

Tests:
- Dimension correctness through the forward pass
- Forward pass produces finite outputs
- Gradient flow (gradients are not None after backprop)
- Probability validity after softmax
- Parameter count
"""

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from src.gcn import GCN, GCNLayer, MLP
from src.graph_data import build_dataset
from src.normalization import normalize_adjacency_torch


@pytest.fixture
def dataset():
    return build_dataset(seed=42)


@pytest.fixture
def A_norm(dataset):
    return normalize_adjacency_torch(dataset["adjacency"])


@pytest.fixture
def X(dataset):
    return dataset["t_features"]


@pytest.fixture
def labels(dataset):
    return dataset["t_labels"]


class TestDimensionCorrectness:
    """Test 5: Verify output dimensions through the forward pass."""

    def test_gcn_layer_dimensions(self, A_norm, X):
        """GCNLayer(3, 8) should produce N × 8 output."""
        layer = GCNLayer(3, 8)
        out = layer(A_norm, X)
        N = X.shape[0]
        assert out.shape == (N, 8), f"Expected ({N}, 8), got {out.shape}"

    def test_three_layer_gcn_dimensions(self, A_norm, X):
        """
        Three-layer GCN dimensions:
            X:  N × 3
            W1: 3 × 8  →  H1: N × 8
            W2: 8 × 8  →  H2: N × 8
            W3: 8 × 2  →  H3: N × 2
        """
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        logits = model(A_norm, X)
        N = X.shape[0]
        assert logits.shape == (N, 2), f"Expected ({N}, 2), got {logits.shape}"

    def test_one_layer_gcn_dimensions(self, A_norm, X):
        """1-layer GCN: X (N×3) → logits (N×2)."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=1)
        logits = model(A_norm, X)
        N = X.shape[0]
        assert logits.shape == (N, 2)

    def test_two_layer_gcn_dimensions(self, A_norm, X):
        """2-layer GCN: X (N×3) → H1 (N×8) → logits (N×2)."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=2)
        logits = model(A_norm, X)
        N = X.shape[0]
        assert logits.shape == (N, 2)

    def test_mlp_dimensions(self, A_norm, X):
        """MLP: X (N×3) → logits (N×2)."""
        model = MLP(n_features=3, n_hidden=8, n_classes=2)
        logits = model(A_norm, X)
        N = X.shape[0]
        assert logits.shape == (N, 2)


class TestForwardPass:
    """Test 6: Ensure the model produces finite outputs."""

    def test_gcn_outputs_finite(self, A_norm, X):
        """GCN forward pass should produce no NaN or Inf values."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        logits = model(A_norm, X)
        assert torch.isfinite(logits).all(), "GCN produced NaN or Inf values!"

    def test_mlp_outputs_finite(self, A_norm, X):
        """MLP forward pass should produce no NaN or Inf values."""
        model = MLP(n_features=3, n_hidden=8, n_classes=2)
        logits = model(A_norm, X)
        assert torch.isfinite(logits).all(), "MLP produced NaN or Inf values!"


class TestGradientFlow:
    """Test 7: Verify gradients are not None after backpropagation."""

    def test_gcn_gradients_exist(self, A_norm, X, labels):
        """After backprop, all GCN parameters should have non-None gradients."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        logits = model(A_norm, X)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Gradient is None for parameter: {name}"
            assert torch.isfinite(param.grad).all(), \
                f"Gradient has NaN/Inf for parameter: {name}"

    def test_mlp_gradients_exist(self, A_norm, X, labels):
        """After backprop, all MLP parameters should have non-None gradients."""
        model = MLP(n_features=3, n_hidden=8, n_classes=2)
        logits = model(A_norm, X)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Gradient is None for parameter: {name}"


class TestProbabilityValidity:
    """Test 9: After softmax, 0 ≤ p_i ≤ 1 and Σ_c p_ic ≈ 1."""

    def test_softmax_probabilities_valid(self, A_norm, X):
        """Softmax outputs should be valid probabilities."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        logits = model(A_norm, X)
        probs = torch.softmax(logits, dim=1)

        # Check range [0, 1]
        assert (probs >= 0).all(), "Probabilities < 0 detected."
        assert (probs <= 1).all(), "Probabilities > 1 detected."

        # Check row sums ≈ 1
        row_sums = probs.sum(dim=1)
        torch.testing.assert_close(
            row_sums, torch.ones(probs.shape[0]),
            atol=1e-5, rtol=1e-5,
        )


class TestParameterCount:
    """Verify parameter counts match expectations."""

    def test_three_layer_gcn_params(self):
        """3-layer GCN: 3×8 + 8×8 + 8×2 = 24 + 64 + 16 = 104 parameters."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        expected = 3 * 8 + 8 * 8 + 8 * 2  # = 104
        actual = model.count_parameters()
        assert actual == expected, f"Expected {expected} params, got {actual}"

    def test_one_layer_gcn_params(self):
        """1-layer GCN: 3×2 = 6 parameters."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=1)
        assert model.count_parameters() == 6

    def test_two_layer_gcn_params(self):
        """2-layer GCN: 3×8 + 8×2 = 40 parameters."""
        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=2)
        assert model.count_parameters() == 40
