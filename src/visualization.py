"""
visualization.py — Plotting Utilities for the GCN Toy Project

Generates publication-quality figures for:
- Graph visualization (colored by class)
- Adjacency matrix heatmaps
- Training curves (loss, accuracy)
- Confusion matrix
- Model comparison bar charts
- PCA embeddings
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import torch
from sklearn.decomposition import PCA


# Consistent color scheme
CLASS_COLORS = {0: "#E74C3C", 1: "#2ECC71"}  # Red = Fail, Green = Pass
CLASS_NAMES = {0: "Fail", 1: "Pass"}


def _finish(fig: plt.Figure, save_path: Optional[str]) -> None:
    """
    Save the figure if a path is given, otherwise display it (e.g. inline
    in a notebook). The figure is closed afterwards to free memory.
    """
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_graph(
    G: nx.Graph,
    labels: np.ndarray,
    title: str = "Student Graph — True Labels",
    save_path: Optional[str] = None,
    predictions: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Visualize the graph with nodes colored by class.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    labels : np.ndarray
        True node labels. Used for coloring when `predictions` is None.
    title : str
        Plot title.
    save_path : str or None
        If provided, save figure to this path.
    predictions : np.ndarray or None
        If provided, nodes are colored by predicted class and nodes whose
        prediction differs from `labels` are marked with a black X.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    pos = nx.spring_layout(G, seed=42, k=1.5)
    color_by = labels if predictions is None else predictions
    node_colors = [CLASS_COLORS[int(color_by[n])] for n in G.nodes()]

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS[0], label=f"Class 0: {CLASS_NAMES[0]}"),
        mpatches.Patch(color=CLASS_COLORS[1], label=f"Class 1: {CLASS_NAMES[1]}"),
    ]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.0, edge_color="#999999")

    # Mark incorrect predictions with X markers, drawn *behind* the nodes so
    # the arms stick out around each node without hiding its number.
    if predictions is not None:
        incorrect = [n for n in G.nodes() if predictions[n] != labels[n]]
        if incorrect:
            # 'x' is an unfilled marker: its stroke color comes from
            # node_color, so it must not be "none" or the X is invisible.
            nx.draw_networkx_nodes(
                G, pos, nodelist=incorrect, ax=ax,
                node_color="black", node_size=1100,
                linewidths=3.0, node_shape="x",
            )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=400,
        edgecolors="black",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    if predictions is not None:
        patches.append(Line2D(
            [], [], color="black", marker="x", linestyle="None",
            markersize=10, markeredgewidth=2.5,
            label=f"Misclassified ({len(incorrect)})",
        ))
    ax.legend(handles=patches, loc="upper left", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    _finish(fig, save_path)


def plot_adjacency_heatmap(
    matrix: np.ndarray,
    title: str = "Adjacency Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = "Blues",
) -> None:
    """
    Plot a heatmap of an adjacency matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Matrix to visualize.
    title : str
        Plot title.
    save_path : str or None
        Save path.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Node Index", fontsize=12)
    ax.set_ylabel("Node Index", fontsize=12)

    # Add tick marks
    N = matrix.shape[0]
    if N <= 30:
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.tick_params(labelsize=8)

    _finish(fig, save_path)


def plot_training_curves(
    history: Dict[str, List[float]],
    title_prefix: str = "",
    save_path_loss: Optional[str] = None,
    save_path_acc: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict
        Keys: train_loss, val_loss, train_acc, val_acc.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss Curve ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#3498DB", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#E74C3C", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(f"{title_prefix}Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path_loss)

    # --- Accuracy Curve ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ECC71", linewidth=2)
    ax.plot(epochs, history["val_acc"], label="Val Accuracy", color="#F39C12", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title_prefix}Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path_acc)


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> None:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray, shape (C, C)
        Confusion matrix.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail (0)", "Pass (1)"], fontsize=11)
    ax.set_yticklabels(["Fail (0)", "Pass (1)"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    _finish(fig, save_path)


def plot_model_comparison(
    results: Dict[str, Dict],
    metric: str = "accuracy",
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_std: bool = True,
) -> None:
    """
    Bar chart comparing models on a given metric.

    Parameters
    ----------
    results : dict
        {model_name: {"mean": float, "std": float}}
    metric : str
        Metric name for labeling.
    show_std : bool
        Whether to show error bars for standard deviation.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    model_names = list(results.keys())
    means = [results[m]["mean"] for m in model_names]
    stds = [results[m].get("std", 0) for m in model_names]

    colors = ["#E74C3C", "#3498DB", "#F39C12", "#2ECC71"]
    while len(colors) < len(model_names):
        colors.extend(colors)

    bars = ax.bar(
        model_names, means,
        yerr=stds if show_std else None,
        capsize=8,
        color=colors[:len(model_names)],
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
    )

    # Annotate bars
    for bar, mean, std in zip(bars, means, stds):
        label = f"{mean:.3f}"
        if show_std and std > 0:
            label += f"±{std:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    _finish(fig, save_path)


def plot_pca_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA of Node Embeddings",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Visualize node embeddings using PCA (first 2 principal components).

    PCA is preferred over t-SNE for this toy project because:
    - Deterministic
    - No hyperparameter sensitivity
    - Sufficient for low-dimensional embeddings

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        Node embedding matrix.
    labels : np.ndarray, shape (N,)
        Node labels.
    """
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
    elif embeddings.shape[1] == 2:
        # Already 2D, use directly
        coords = embeddings
    else:
        # 1D embeddings: plot along PC 1 with PC 2 fixed at zero
        coords = np.column_stack([embeddings[:, 0], np.zeros(len(embeddings))])

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CLASS_COLORS[int(cls)],
            label=f"{CLASS_NAMES[int(cls)]} ({int(cls)})",
            s=100, edgecolors="black", linewidths=0.8, alpha=0.8,
        )

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _finish(fig, save_path)
