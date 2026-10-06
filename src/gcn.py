"""
gcn.py — Manual Graph Convolutional Network Implementation

Implements the GCN layer and multi-layer GCN model from scratch using
only PyTorch primitives (nn.Module, nn.Parameter). No PyTorch Geometric.

GCN Layer Mathematics:
    H' = Ã H W

    where:
        Ã  : N × N  — normalized adjacency matrix (precomputed)
        H  : N × F_in  — input node features/representations
        W  : F_in × F_out  — learnable weight matrix
        H' : N × F_out  — output node representations

    Step by step:
        Ã @ H  : (N × N) @ (N × F_in) = N × F_in   — neighbourhood aggregation
        (Ã @ H) @ W : (N × F_in) @ (F_in × F_out) = N × F_out  — linear transform

Three-layer GCN Architecture:
    H^(1) = ReLU(Ã X W^(0))        — layer 1: 3 → 8
    H^(2) = ReLU(Ã H^(1) W^(1))   — layer 2: 8 → 8
    H^(3) = Ã H^(2) W^(2)          — layer 3: 8 → 2 (logits, NO activation)

Reference:
    Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
    Networks", ICLR 2017.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    A single Graph Convolutional Network layer.

    Performs: H' = Ã H W

    The weight matrix W is a learnable parameter initialized using
    Xavier uniform initialization, which helps maintain the variance
    of activations across layers.

    Parameters
    ----------
    in_features : int
        Input feature dimension F_in.
    out_features : int
        Output feature dimension F_out.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix W: F_in × F_out
        # Using nn.Parameter so PyTorch tracks it for gradient computation
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        # Initialize weights using Xavier uniform initialization
        # This sets weights from U(-a, a) where a = sqrt(6 / (F_in + F_out))
        # It aims to keep the variance of activations roughly constant
        # across layers, preventing vanishing/exploding gradients.
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weight matrix using Xavier uniform."""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A_norm: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single GCN layer.

        Computes: H' = Ã H W

        Parameters
        ----------
        A_norm : torch.Tensor, shape (N, N)
            Normalized adjacency matrix Ã.
        H : torch.Tensor, shape (N, F_in)
            Input node representations.

        Returns
        -------
        H_out : torch.Tensor, shape (N, F_out)
            Output node representations.

        Dimensions step-by-step:
            Ã @ H  : (N × N) @ (N × F_in) = (N × F_in)  — aggregate neighbours
            (Ã @ H) @ W : (N × F_in) @ (F_in × F_out) = (N × F_out)  — transform
        """
        # Step 1: Neighbourhood aggregation  — Ã @ H
        # Each node's representation becomes a weighted sum of its
        # neighbours' (and its own) representations.
        support = torch.mm(A_norm, H)  # N × F_in

        # Step 2: Linear transformation  — support @ W
        # Project aggregated features into the output space.
        H_out = torch.mm(support, self.weight)  # N × F_out

        return H_out

    def __repr__(self) -> str:
        return (
            f"GCNLayer(in_features={self.in_features}, "
            f"out_features={self.out_features})"
        )


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network for node classification.

    Architecture (default 3-layer):
        X (N × 3)  →  GCN(3→8)  →  ReLU  →  GCN(8→8)  →  ReLU  →  GCN(8→2)  →  Logits

    The final layer does NOT apply an activation function because:
    - It outputs raw logits for classification
    - PyTorch's CrossEntropyLoss expects logits (it applies LogSoftmax internally)
    - Applying softmax before CrossEntropyLoss would cause numerical instability

    Parameters
    ----------
    n_features : int
        Number of input features per node (F = 3).
    n_hidden : int
        Number of hidden dimensions (default 8).
    n_classes : int
        Number of output classes (default 2).
    n_layers : int
        Number of GCN layers (1, 2, or 3).
    dropout : float
        Dropout rate (default 0.0 for this toy project).
    """

    def __init__(
        self,
        n_features: int = 3,
        n_hidden: int = 8,
        n_classes: int = 2,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()

        if n_layers == 1:
            # Single layer: input → output directly
            # 3 → 2
            self.layers.append(GCNLayer(n_features, n_classes))

        elif n_layers == 2:
            # Two layers: input → hidden → output
            # 3 → 8 → 2
            self.layers.append(GCNLayer(n_features, n_hidden))
            self.layers.append(GCNLayer(n_hidden, n_classes))

        elif n_layers == 3:
            # Three layers: input → hidden → hidden → output
            # 3 → 8 → 8 → 2
            self.layers.append(GCNLayer(n_features, n_hidden))
            self.layers.append(GCNLayer(n_hidden, n_hidden))
            self.layers.append(GCNLayer(n_hidden, n_classes))

        else:
            raise ValueError(f"n_layers must be 1, 2, or 3, got {n_layers}")

    def forward(
        self,
        A_norm: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through all GCN layers.

        For a 3-layer GCN:
            H^(1) = ReLU(Ã X W^(0))          — 3 → 8
            H^(2) = ReLU(Ã H^(1) W^(1))      — 8 → 8
            H^(3) = Ã H^(2) W^(2)             — 8 → 2  (logits)

        Parameters
        ----------
        A_norm : torch.Tensor, shape (N, N)
            Normalized adjacency matrix Ã.
        X : torch.Tensor, shape (N, F)
            Input node feature matrix.

        Returns
        -------
        logits : torch.Tensor, shape (N, n_classes)
            Raw output logits (NOT probabilities).
        """
        H = X  # H^(0) = X

        for i, layer in enumerate(self.layers):
            H = layer(A_norm, H)

            # Apply ReLU after every layer EXCEPT the last
            # The last layer outputs logits for classification
            if i < len(self.layers) - 1:
                H = F.relu(H)
                if self.dropout > 0:
                    H = F.dropout(H, p=self.dropout, training=self.training)

        return H  # Logits: N × n_classes

    def get_embeddings(
        self,
        A_norm: torch.Tensor,
        X: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Get intermediate node embeddings from a specific layer.

        Useful for visualizing learned representations (e.g., with PCA).

        Parameters
        ----------
        A_norm : torch.Tensor, shape (N, N)
            Normalized adjacency matrix.
        X : torch.Tensor, shape (N, F)
            Input features.
        layer_idx : int
            Which layer's output to return (0-indexed, before final activation).
            Use -1 for the final logits.

        Returns
        -------
        embeddings : torch.Tensor
            Node embeddings at the specified layer.
        """
        H = X
        embeddings = []

        for i, layer in enumerate(self.layers):
            H = layer(A_norm, H)
            if i < len(self.layers) - 1:
                H = F.relu(H)
            embeddings.append(H.detach())

        if layer_idx == -1:
            return embeddings[-1]
        return embeddings[layer_idx]

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline (no graph structure).

    Architecture:
        X (N × 3) → Linear(3→8) → ReLU → Linear(8→2) → Logits

    This baseline uses only node features X, ignoring the graph structure.
    Comparing GCN vs MLP reveals whether the graph structure is helpful.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_hidden : int
        Hidden dimension.
    n_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        n_features: int = 3,
        n_hidden: int = 8,
        n_classes: int = 2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(
        self,
        A_norm: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass. A_norm is accepted but ignored (for API compatibility).

        Parameters
        ----------
        A_norm : torch.Tensor
            Ignored — MLP does not use graph structure.
        X : torch.Tensor, shape (N, F)
            Node features.

        Returns
        -------
        logits : torch.Tensor, shape (N, n_classes)
        """
        H = F.relu(self.fc1(X))  # N × n_hidden
        logits = self.fc2(H)     # N × n_classes
        return logits

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
