"""
test_visualization.py — Smoke Tests for Plotting Utilities

Tests:
- Predicted-graph plot runs and saves when some predictions are wrong
- PCA plot handles 1-, 2-, and higher-dimensional embeddings
"""

import matplotlib

matplotlib.use("Agg")  # headless backend: no windows during tests

import numpy as np
import pytest

from src.graph_data import build_dataset
from src.visualization import plot_graph, plot_pca_embeddings


@pytest.fixture
def dataset():
    return build_dataset(seed=42)


def test_plot_graph_with_wrong_predictions_saves(dataset, tmp_path):
    labels = dataset["labels"]
    predictions = labels.copy()
    predictions[:3] = 1 - predictions[:3]  # three misclassified nodes

    out = tmp_path / "graph.png"
    plot_graph(dataset["graph"], labels, save_path=str(out), predictions=predictions)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("dim", [1, 2, 8])
def test_plot_pca_embeddings_any_dimension(dataset, tmp_path, dim):
    rng = np.random.RandomState(0)
    embeddings = rng.randn(len(dataset["labels"]), dim)

    out = tmp_path / f"pca_{dim}.png"
    plot_pca_embeddings(embeddings, dataset["labels"], save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
