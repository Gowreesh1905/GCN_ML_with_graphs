"""
graph_data.py — Synthetic Student-Performance Graph Generator

Generates a deterministic synthetic graph representing students in a course.
Each node (student) has 3 features: study_hours, attendance, assignment_score.
Each node has a binary label: 0 = Fail, 1 = Pass.

The graph exhibits meaningful but imperfect homophily: students of the same
class are more likely to be connected (e.g., study groups), but some
cross-class edges exist.

Defaults: 200 students (110 Pass, 90 Fail) with strongly overlapping
features (feature_gap=0.25), so features alone are only moderately
informative and the graph has room to help.
"""

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_student_features(
    n_pass: int = 110,
    n_fail: int = 90,
    seed: int = 42,
    feature_gap: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic student features and labels.

    Pass students (label=1): higher study hours, attendance, assignment scores.
    Fail students (label=0): lower values, with some overlap for realism.

    Parameters
    ----------
    n_pass : int
        Number of passing students.
    n_fail : int
        Number of failing students.
    seed : int
        Random seed for reproducibility.
    feature_gap : float
        How far apart the class means are. 1.0 gives the reference means
        below (classes almost perfectly separable); smaller values pull both
        classes' means toward their midpoint, so the classes overlap more
        and features alone become less informative. Standard deviations are
        unchanged.

    Returns
    -------
    features : np.ndarray, shape (N, 3)
        Columns: [study_hours, attendance, assignment_score]
    labels : np.ndarray, shape (N,)
        Binary labels: 0 = Fail, 1 = Pass
    """
    rng = np.random.RandomState(seed)
    N = n_pass + n_fail

    # Reference class means: [study_hours, attendance, assignment_score]
    pass_ref = np.array([7.0, 80.0, 75.0])
    fail_ref = np.array([3.0, 50.0, 40.0])
    mid = (pass_ref + fail_ref) / 2
    pass_mean = mid + feature_gap * (pass_ref - mid)
    fail_mean = mid + feature_gap * (fail_ref - mid)

    # --- Pass students (label = 1) ---
    # std: study_hours 1.5, attendance 10, assignment_score 10
    pass_features = np.column_stack([
        rng.normal(loc=pass_mean[0], scale=1.5, size=n_pass),   # study_hours
        rng.normal(loc=pass_mean[1], scale=10.0, size=n_pass),  # attendance
        rng.normal(loc=pass_mean[2], scale=10.0, size=n_pass),  # assignment_score
    ])

    # --- Fail students (label = 0) ---
    # std: study_hours 1.5, attendance 12, assignment_score 12
    fail_features = np.column_stack([
        rng.normal(loc=fail_mean[0], scale=1.5, size=n_fail),   # study_hours
        rng.normal(loc=fail_mean[1], scale=12.0, size=n_fail),  # attendance
        rng.normal(loc=fail_mean[2], scale=12.0, size=n_fail),  # assignment_score
    ])

    # Clip to realistic ranges
    pass_features[:, 0] = np.clip(pass_features[:, 0], 0, 12)   # study_hours
    pass_features[:, 1] = np.clip(pass_features[:, 1], 0, 100)  # attendance
    pass_features[:, 2] = np.clip(pass_features[:, 2], 0, 100)  # assignment_score

    fail_features[:, 0] = np.clip(fail_features[:, 0], 0, 12)
    fail_features[:, 1] = np.clip(fail_features[:, 1], 0, 100)
    fail_features[:, 2] = np.clip(fail_features[:, 2], 0, 100)

    # Combine: first n_pass nodes are Pass, next n_fail are Fail
    features = np.vstack([pass_features, fail_features])
    labels = np.array([1] * n_pass + [0] * n_fail)

    # Shuffle to avoid trivial ordering
    perm = rng.permutation(N)
    features = features[perm]
    labels = labels[perm]

    return features, labels


def generate_student_graph(
    labels: np.ndarray,
    p_same: float = 0.06,
    p_cross: float = 0.015,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate an undirected graph with homophilic structure.

    Nodes with the same label are more likely to be connected (study groups),
    but cross-class edges exist (imperfect homophily).

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Node labels.
    p_same : float
        Probability of edge between two nodes of the same class.
    p_cross : float
        Probability of edge between two nodes of different classes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    G : nx.Graph
        Undirected graph with N nodes.
    """
    rng = np.random.RandomState(seed)
    N = len(labels)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] == labels[j]:
                if rng.rand() < p_same:
                    G.add_edge(i, j)
            else:
                if rng.rand() < p_cross:
                    G.add_edge(i, j)

    return G


def get_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Extract the adjacency matrix A from a NetworkX graph.

    A_ij = 1 if edge (i,j) exists, 0 otherwise.
    For an undirected graph, A = A^T.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Binary adjacency matrix.
    """
    N = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    return A


def compute_graph_statistics(
    G: nx.Graph,
    labels: np.ndarray,
) -> Dict:
    """
    Compute and return basic graph statistics.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    labels : np.ndarray
        Node labels.

    Returns
    -------
    stats : dict
        Dictionary of graph statistics.
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()
    degrees = [d for _, d in G.degree()]

    # Homophily: fraction of edges connecting same-label nodes
    same_label_edges = sum(
        1 for u, v in G.edges() if labels[u] == labels[v]
    )
    cross_label_edges = E - same_label_edges
    homophily = same_label_edges / E if E > 0 else 0.0

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))

    stats = {
        "num_nodes": N,
        "num_edges": E,
        "num_classes": len(unique),
        "feature_dimensions": 3,
        "class_distribution": class_dist,
        "graph_density": nx.density(G),
        "average_degree": np.mean(degrees),
        "min_degree": int(np.min(degrees)),
        "max_degree": int(np.max(degrees)),
        "is_directed": G.is_directed(),
        "has_self_loops": nx.number_of_selfloops(G) > 0,
        "is_connected": nx.is_connected(G),
        "num_connected_components": nx.number_connected_components(G),
        "same_label_edges": same_label_edges,
        "cross_label_edges": cross_label_edges,
        "homophily_ratio": homophily,
    }
    return stats


def create_train_val_test_masks(
    N: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create reproducible train/validation/test node masks.

    Parameters
    ----------
    N : int
        Number of nodes.
    train_ratio : float
        Fraction of nodes for training.
    val_ratio : float
        Fraction of nodes for validation.
    seed : int
        Random seed.

    Returns
    -------
    train_mask, val_mask, test_mask : np.ndarray (bool), each shape (N,)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def standardize_features(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training-set statistics to avoid data leakage.

    x' = (x - mu) / sigma

    Parameters
    ----------
    features : np.ndarray, shape (N, F)
        Raw features.
    train_mask : np.ndarray, shape (N,), dtype=bool
        Boolean mask for training nodes.

    Returns
    -------
    features_standardized : np.ndarray, shape (N, F)
        Standardized features (all nodes use train statistics).
    mu : np.ndarray, shape (F,)
        Training mean per feature.
    sigma : np.ndarray, shape (F,)
        Training std per feature.
    """
    train_features = features[train_mask]
    mu = train_features.mean(axis=0)
    sigma = train_features.std(axis=0)

    # Avoid division by zero
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    features_standardized = (features - mu) / sigma
    return features_standardized, mu, sigma


def generate_random_graph(
    N: int,
    num_edges: int,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate a random (Erdos-Renyi-like) graph with approximately
    the given number of edges, for ablation study.

    Parameters
    ----------
    N : int
        Number of nodes.
    num_edges : int
        Target number of edges.
    seed : int
        Random seed.

    Returns
    -------
    G : nx.Graph
        Random undirected graph.
    """
    max_edges = N * (N - 1) // 2
    p = min(num_edges / max_edges, 1.0) if max_edges > 0 else 0
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    return G


def build_dataset(
    n_pass: int = 110,
    n_fail: int = 90,
    p_same: float = 0.06,
    p_cross: float = 0.015,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_seed: Optional[int] = None,
    feature_gap: float = 0.25,
) -> Dict:
    """
    Build the complete dataset: features, labels, graph, masks, and tensors.

    This is the main entry point for generating the toy dataset.

    Parameters
    ----------
    feature_gap : float
        Separation of the class feature means (see generate_student_features).
    seed : int
        Seed for the features, labels, and graph.
    split_seed : int or None
        Seed for the train/val/test split. Defaults to `seed`. Passing a
        different value keeps the same graph but draws a new node split,
        which is how multi-seed experiments vary the evaluation set.

    Returns
    -------
    data : dict
        Keys: features_raw, features, labels, graph, adjacency,
              train_mask, val_mask, test_mask, mu, sigma, stats,
              and torch tensor versions prefixed with 't_'.
    """
    set_all_seeds(seed)

    # Generate features and labels
    features_raw, labels = generate_student_features(n_pass, n_fail, seed, feature_gap)

    # Generate graph
    G = generate_student_graph(labels, p_same, p_cross, seed)

    # Get adjacency matrix
    A = get_adjacency_matrix(G)

    # Create masks
    if split_seed is None:
        split_seed = seed
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        len(labels), train_ratio, val_ratio, split_seed
    )

    # Standardize features using training set statistics
    features_std, mu, sigma = standardize_features(features_raw, train_mask)

    # Compute statistics
    stats = compute_graph_statistics(G, labels)

    # Convert to PyTorch tensors
    data = {
        # NumPy arrays
        "features_raw": features_raw,
        "features": features_std,
        "labels": labels,
        "graph": G,
        "adjacency": A,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "mu": mu,
        "sigma": sigma,
        "stats": stats,
        # PyTorch tensors
        "t_features": torch.tensor(features_std, dtype=torch.float32),
        "t_labels": torch.tensor(labels, dtype=torch.long),
        "t_adjacency": torch.tensor(A, dtype=torch.float32),
        "t_train_mask": torch.tensor(train_mask, dtype=torch.bool),
        "t_val_mask": torch.tensor(val_mask, dtype=torch.bool),
        "t_test_mask": torch.tensor(test_mask, dtype=torch.bool),
    }

    return data
