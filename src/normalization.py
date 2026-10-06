"""
normalization.py — Adjacency Matrix Normalization for GCN

Implements the standard GCN normalization from Kipf & Welling (2017):
    Ã = D̂^(-1/2) Â D̂^(-1/2)
where:
    Â = A + I   (adjacency with self-loops)
    D̂_ii = Σ_j Â_ij   (degree matrix of Â)

All operations are implemented explicitly — no graph-library shortcuts.
"""

import numpy as np
import torch


def add_self_loops(A: np.ndarray) -> np.ndarray:
    """
    Add self-loops to the adjacency matrix.

    Â = A + I

    A node should aggregate information from both its neighbours and itself.
    Without self-loops, a node's own features would be lost during
    neighbourhood aggregation.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Original adjacency matrix (no self-loops).

    Returns
    -------
    A_hat : np.ndarray, shape (N, N)
        Adjacency matrix with self-loops: Â = A + I
    """
    N = A.shape[0]
    I = np.eye(N, dtype=A.dtype)
    A_hat = A + I
    return A_hat


def compute_degree_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal degree matrix.

    D_ii = Σ_j A_ij
    D_ij = 0  for i ≠ j

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Adjacency matrix (with or without self-loops).

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Diagonal degree matrix.
    """
    degrees = A.sum(axis=1)  # Row sums
    D = np.diag(degrees)
    return D


def compute_inverse_sqrt_degree(D: np.ndarray) -> np.ndarray:
    """
    Compute D^(-1/2) safely, handling zero-degree nodes.

    For a diagonal matrix D:
        D^(-1/2)_ii = 1 / sqrt(D_ii)   if D_ii > 0
        D^(-1/2)_ii = 0                if D_ii = 0

    Setting zero-degree entries to 0 rather than allowing division by zero
    or NaN is a safe default. With self-loops, this case typically doesn't
    arise because every node has degree >= 1.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
        Diagonal degree matrix.

    Returns
    -------
    D_inv_sqrt : np.ndarray, shape (N, N)
        D^(-1/2), diagonal matrix.
    """
    diag = np.diag(D).copy()
    # Safe inverse square root: avoid division by zero
    inv_sqrt = np.where(diag > 0, 1.0 / np.sqrt(diag), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt)
    return D_inv_sqrt


def symmetric_normalization(A: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric normalized adjacency matrix.

    Ã = D̂^(-1/2) Â D̂^(-1/2)

    where Â = A + I (self-loops added) and D̂ is the degree matrix of Â.

    This normalization:
    - Prevents high-degree nodes from dominating aggregation
    - Keeps feature magnitudes stable across layers
    - Preserves symmetry for undirected graphs

    The full pipeline:
    1. Â = A + I                (add self-loops)
    2. D̂_ii = Σ_j Â_ij         (compute degrees)
    3. D̂^(-1/2)                (inverse square root of degrees)
    4. Ã = D̂^(-1/2) Â D̂^(-1/2)  (symmetric normalization)

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Original adjacency matrix (no self-loops assumed).

    Returns
    -------
    A_norm : np.ndarray, shape (N, N)
        Normalized adjacency matrix Ã.
    A_hat : np.ndarray, shape (N, N)
        Adjacency with self-loops Â.
    D_hat : np.ndarray, shape (N, N)
        Degree matrix of Â.
    D_hat_inv_sqrt : np.ndarray, shape (N, N)
        D̂^(-1/2).
    """
    # Step 1: Add self-loops
    A_hat = add_self_loops(A)

    # Step 2: Compute degree matrix of Â
    D_hat = compute_degree_matrix(A_hat)

    # Step 3: Compute D̂^(-1/2)
    D_hat_inv_sqrt = compute_inverse_sqrt_degree(D_hat)

    # Step 4: Ã = D̂^(-1/2) @ Â @ D̂^(-1/2)
    #
    # Matrix dimensions:
    #   D̂^(-1/2)  : N × N
    #   Â          : N × N
    #   D̂^(-1/2)  : N × N
    #   Result Ã   : N × N
    A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

    return A_norm, A_hat, D_hat, D_hat_inv_sqrt


def normalize_adjacency_torch(A: np.ndarray) -> torch.Tensor:
    """
    Compute the normalized adjacency and return as a PyTorch float tensor.

    This is a convenience wrapper for use in training.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Original adjacency matrix.

    Returns
    -------
    A_norm_tensor : torch.Tensor, shape (N, N), dtype=float32
        Normalized adjacency matrix as a PyTorch tensor.
    """
    A_norm, _, _, _ = symmetric_normalization(A)
    return torch.tensor(A_norm, dtype=torch.float32)
