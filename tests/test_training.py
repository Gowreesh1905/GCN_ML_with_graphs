"""
test_training.py — Tests for Training and Reproducibility

Tests:
- Loss decreases during training (Test 8)
- Reproducibility across identical seeds (Test 10)
"""

import numpy as np
import torch
import pytest

from src.graph_data import build_dataset, set_all_seeds
from src.gcn import GCN
from src.normalization import normalize_adjacency_torch
from src.train import train_model


@pytest.fixture
def dataset():
    return build_dataset(seed=42)


class TestLossDecreases:
    """Test 8: Training loss should decrease over epochs."""

    def test_training_loss_decreases(self, dataset):
        """
        Train for a small number of epochs and verify the final training
        loss is lower than the initial loss.

        We allow reasonable tolerance — this is not a brittle exact-value test.
        """
        set_all_seeds(42)

        A_norm = normalize_adjacency_torch(dataset["adjacency"])
        X = dataset["t_features"]
        labels = dataset["t_labels"]
        train_mask = dataset["t_train_mask"]
        val_mask = dataset["t_val_mask"]

        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)

        history = train_model(
            model, A_norm, X, labels,
            train_mask, val_mask,
            lr=0.01, epochs=100, seed=42, verbose=False,
        )

        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]

        assert final_loss < initial_loss, (
            f"Training loss did not decrease: "
            f"initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_training_loss_significantly_decreases(self, dataset):
        """Final loss should be at least 20% lower than initial loss."""
        set_all_seeds(42)

        A_norm = normalize_adjacency_torch(dataset["adjacency"])
        X = dataset["t_features"]
        labels = dataset["t_labels"]
        train_mask = dataset["t_train_mask"]
        val_mask = dataset["t_val_mask"]

        model = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)

        history = train_model(
            model, A_norm, X, labels,
            train_mask, val_mask,
            lr=0.01, epochs=100, seed=42, verbose=False,
        )

        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]

        reduction = (initial_loss - final_loss) / initial_loss
        assert reduction > 0.2, (
            f"Loss only reduced by {reduction*100:.1f}%. "
            f"Expected > 20% reduction."
        )


class TestReproducibility:
    """Test 10: Results should be identical with the same seed."""

    def test_same_seed_same_results(self, dataset):
        """
        Run training twice with the same seed and verify
        that the final loss values are identical.
        """
        A_norm = normalize_adjacency_torch(dataset["adjacency"])
        X = dataset["t_features"]
        labels = dataset["t_labels"]
        train_mask = dataset["t_train_mask"]
        val_mask = dataset["t_val_mask"]

        # Run 1
        set_all_seeds(42)
        model1 = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        history1 = train_model(
            model1, A_norm, X, labels,
            train_mask, val_mask,
            lr=0.01, epochs=50, seed=42, verbose=False,
        )

        # Run 2
        set_all_seeds(42)
        model2 = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        history2 = train_model(
            model2, A_norm, X, labels,
            train_mask, val_mask,
            lr=0.01, epochs=50, seed=42, verbose=False,
        )

        # Compare final losses
        np.testing.assert_allclose(
            history1["train_loss"][-1],
            history2["train_loss"][-1],
            atol=1e-6,
            err_msg="Same seed produced different training losses!"
        )

        np.testing.assert_allclose(
            history1["val_loss"][-1],
            history2["val_loss"][-1],
            atol=1e-6,
            err_msg="Same seed produced different validation losses!"
        )

    def test_different_seed_different_results(self, dataset):
        """
        Different seeds should generally produce different results.
        This is a sanity check that the seed actually affects training.
        """
        A_norm = normalize_adjacency_torch(dataset["adjacency"])
        X = dataset["t_features"]
        labels = dataset["t_labels"]
        train_mask = dataset["t_train_mask"]
        val_mask = dataset["t_val_mask"]

        # Run with seed 42
        set_all_seeds(42)
        model1 = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        history1 = train_model(
            model1, A_norm, X, labels,
            train_mask, val_mask,
            lr=0.01, epochs=50, seed=42, verbose=False,
        )

        # Run with seed 99
        set_all_seeds(99)
        model2 = GCN(n_features=3, n_hidden=8, n_classes=2, n_layers=3)
        history2 = train_model(
            model2, A_norm, X, labels,
            train_mask, val_mask,
            lr=0.01, epochs=50, seed=99, verbose=False,
        )

        # They should be different (with very high probability)
        # Using a generous tolerance — just checking they're not identical
        assert history1["train_loss"][-1] != history2["train_loss"][-1], \
            "Different seeds produced identical results — suspicious!"
