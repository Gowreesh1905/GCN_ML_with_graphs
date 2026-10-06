"""
train.py — Training Loop for GCN and MLP Models

Implements the training procedure with:
- Adam optimizer
- Cross-entropy loss on logits (numerically stable)
- Node-level train/validation masking
- Epoch-level logging of loss and accuracy
- Reproducible seeding
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.graph_data import set_all_seeds


def train_model(
    model: nn.Module,
    A_norm: torch.Tensor,
    X: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 200,
    seed: int = 42,
    verbose: bool = True,
    print_every: int = 20,
) -> Dict[str, List[float]]:
    """
    Train a GCN or MLP model using Adam and cross-entropy loss.

    Loss Function:
        L = -Σ_{i ∈ T} log P_{i, y_i}

        where T is the training-node set and P = softmax(logits).
        We use nn.CrossEntropyLoss, which expects raw logits and
        internally applies log-softmax for numerical stability.

    Parameters
    ----------
    model : nn.Module
        GCN or MLP model.
    A_norm : torch.Tensor, shape (N, N)
        Normalized adjacency matrix.
    X : torch.Tensor, shape (N, F)
        Standardized node features.
    labels : torch.Tensor, shape (N,)
        Node labels (0 or 1).
    train_mask : torch.Tensor, shape (N,), dtype=bool
        Training node mask.
    val_mask : torch.Tensor, shape (N,), dtype=bool
        Validation node mask.
    lr : float
        Learning rate for Adam optimizer.
    epochs : int
        Number of training epochs.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.
    print_every : int
        Print interval.

    Returns
    -------
    history : dict
        Keys: train_loss, val_loss, train_acc, val_acc
        Each is a list of length `epochs`.
    """
    set_all_seeds(seed)

    # CrossEntropyLoss on logits — numerically stable
    # It internally computes: -log(softmax(logits)[target_class])
    # This avoids computing softmax separately, which can cause
    # numerical overflow/underflow.
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        optimizer.zero_grad()

        # Forward pass: model outputs logits (N × C)
        logits = model(A_norm, X)

        # Compute loss only on training nodes
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        train_loss = criterion(train_logits, train_labels)

        # Backward pass and parameter update
        train_loss.backward()
        optimizer.step()

        # --- Compute metrics ---
        model.eval()
        with torch.no_grad():
            logits = model(A_norm, X)

            # Training metrics
            train_pred = logits[train_mask].argmax(dim=1)
            train_acc = (train_pred == labels[train_mask]).float().mean().item()

            # Validation metrics
            val_logits = logits[val_mask]
            val_labels = labels[val_mask]
            val_loss = criterion(val_logits, val_labels).item()
            val_pred = logits[val_mask].argmax(dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean().item()

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    return history
