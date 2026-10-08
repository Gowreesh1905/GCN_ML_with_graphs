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


def _graph_layout(G: nx.Graph, k: Optional[float] = None) -> Dict:
    """
    Spring layout of the largest connected component, with any other
    components (e.g. isolated nodes) placed in a row just below it.

    A plain spring layout pushes disconnected nodes far away from the rest,
    which shrinks the main graph into a corner of the figure.
    """
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = nx.spring_layout(G.subgraph(components[0]), seed=42, k=k)
    rest = [n for comp in components[1:] for n in sorted(comp)]
    if rest:
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        row_x = np.linspace(xs.min(), xs.max(), len(rest) + 2)[1:-1]
        row_y = ys.min() - 0.12 * (ys.max() - ys.min())
        for n, x in zip(rest, row_x):
            pos[n] = np.array([x, row_y])
    return pos


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

    # Small graphs get large, numbered nodes; large graphs get small dots
    small = G.number_of_nodes() <= 50
    node_size = 400 if small else 70

    pos = _graph_layout(G, k=1.5 if small else None)
    color_by = labels if predictions is None else predictions
    node_colors = [CLASS_COLORS[int(color_by[n])] for n in G.nodes()]

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS[0], label=f"Class 0: {CLASS_NAMES[0]}"),
        mpatches.Patch(color=CLASS_COLORS[1], label=f"Class 1: {CLASS_NAMES[1]}"),
    ]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3 if small else 0.15,
                           width=1.0, edge_color="#999999")

    # Mark incorrect predictions with X markers, drawn *behind* the nodes so
    # the arms stick out around each node without hiding its number.
    if predictions is not None:
        incorrect = [n for n in G.nodes() if predictions[n] != labels[n]]
        if incorrect:
            # 'x' is an unfilled marker: its stroke color comes from
            # node_color, so it must not be "none" or the X is invisible.
            nx.draw_networkx_nodes(
                G, pos, nodelist=incorrect, ax=ax,
                node_color="black", node_size=node_size * 2.75,
                linewidths=3.0 if small else 1.8, node_shape="x",
            )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=1.5 if small else 0.6,
    )
    if small:
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
        Keys: train_loss, val_loss, train_acc, val_acc, and optionally
        best_epoch (0-indexed), which is marked with a vertical line.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    best = history.get("best_epoch")
    # Only mark it when training ran past the kept epoch (early stopping)
    mark_best = best is not None and best + 1 < len(history["train_loss"])

    def _mark_best(ax):
        if mark_best:
            ax.axvline(best + 1, color="black", linestyle=":", linewidth=1.5,
                       label=f"Weights kept (epoch {best + 1})")

    # --- Loss Curve ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#3498DB", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#E74C3C", linewidth=2, linestyle="--")
    _mark_best(ax)
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
    _mark_best(ax)
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

    colors = ["#E74C3C", "#3498DB", "#F39C12", "#2ECC71", "#9B59B6", "#7F8C8D", "#34495E"]
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
        # Place the label above the error bar so they don't overlap
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (std if show_std else 0) + 0.02,
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


# Validated categorical palette for the sweep chart (fixed order, never cycled)
SWEEP_STYLES = {
    "MLP":            {"color": "#2a78d6", "marker": "o", "label": "MLP (features only)"},
    "Neighbour vote": {"color": "#eb6834", "marker": "s", "label": "Neighbour vote (graph only)"},
    "GCN-1":          {"color": "#1baf7a", "marker": "^", "label": "GCN-1"},
    "GCN-3":          {"color": "#eda100", "marker": "D", "label": "GCN-3"},
}


def plot_homophily_sweep(
    sweep: Dict[str, List],
    majority_rate: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Test accuracy against graph homophily for each method.

    Parameters
    ----------
    sweep : dict
        "homophily": list of realised homophily values (one per level), and
        for each method in SWEEP_STYLES a list (one per level) of per-seed
        accuracies.
    majority_rate : float or None
        Accuracy of always predicting the majority class; drawn as a
        dashed reference line.
    """
    ink, muted = "#333333", "#777777"
    h = np.array(sweep["homophily"])
    order = np.argsort(h)
    h = h[order]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if majority_rate is not None:
        ax.axhline(majority_rate, color=muted, linestyle="--", linewidth=1.2, zorder=1)
        ax.text(h[-1], majority_rate + 0.01, f"Always guess the majority class ({majority_rate:.0%})",
                color=muted, fontsize=9, ha="right", va="bottom")

    end_points = []
    for method, style in SWEEP_STYLES.items():
        if method not in sweep:
            continue
        accs = [np.asarray(a) for a in sweep[method]]
        mean = np.array([a.mean() for a in accs])[order]
        std = np.array([a.std() for a in accs])[order]
        ax.fill_between(h, mean - std, mean + std, color=style["color"], alpha=0.10,
                        linewidth=0, zorder=2)
        ax.plot(h, mean, color=style["color"], linewidth=2, marker=style["marker"],
                markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                label=style["label"], zorder=3)
        end_points.append([mean[-1], style["label"]])

    # Direct labels at the right-hand end, nudged apart so they don't collide
    end_points.sort(key=lambda e: e[0])
    min_gap, y_top = 0.035, 1.0
    for i in range(1, len(end_points)):          # push up to clear the label below
        end_points[i][0] = max(end_points[i][0], end_points[i - 1][0] + min_gap)
    end_points[-1][0] = min(end_points[-1][0], y_top)
    for i in range(len(end_points) - 2, -1, -1):  # then keep everything under the top
        end_points[i][0] = min(end_points[i][0], end_points[i + 1][0] - min_gap)
    x_label = h[-1] + 0.012
    for y, text in end_points:
        ax.text(x_label, y, text, color=ink, fontsize=9.5, va="center")

    ax.set_xlim(h[0] - 0.02, h[-1] + 0.17)
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(np.round(h, 2))
    ax.set_xlabel("Homophily (share of edges joining students with the same result)", fontsize=11, color=ink)
    ax.set_ylabel("Test accuracy", fontsize=11, color=ink)
    ax.set_title("Test Accuracy vs Graph Homophily (10 seeds, ±1 std)", fontsize=14,
                 fontweight="bold", color=ink)
    ax.grid(True, alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    _finish(fig, save_path)
