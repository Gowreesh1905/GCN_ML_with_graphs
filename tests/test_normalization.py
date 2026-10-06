"""
test_normalization.py — Tests for Adjacency Matrix Normalization

Tests:
- Self-loops: diag(A + I) = 1
- Degree matrix: D_ii = Σ_j A_ij, D_ij = 0 for i ≠ j
- Normalized adjacency symmetry: Ã ≈ Ã^T
- Normalization values in valid range
"""

import numpy as np
import pytest

from src.graph_data import build_dataset
from src.normalization import (
    add_self_loops,
    compute_degree_matrix,
    compute_inverse_sqrt_degree,
    symmetric_normalization,
)


@pytest.fixture
def dataset():
    return build_dataset(seed=42)


@pytest.fixture
def adjacency(dataset):
    return dataset["adjacency"]


class TestSelfLoops:
    """Test 2: Verify diag(A + I) = 1."""

    def test_self_loop_diagonal_is_one(self, adjacency):
        """After adding self-loops, every diagonal entry must be 1."""
        A_hat = add_self_loops(adjacency)
        diag = np.diag(A_hat)
        np.testing.assert_array_equal(
            diag, np.ones(len(diag)),
            err_msg="All diagonal entries of Â = A + I should be 1."
        )

    def test_self_loops_preserve_off_diagonal(self, adjacency):
        """Self-loops should not change off-diagonal entries."""
        A_hat = add_self_loops(adjacency)
        N = adjacency.shape[0]
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert A_hat[i, j] == adjacency[i, j], \
                        f"Off-diagonal entry ({i},{j}) changed after adding self-loops."

    def test_self_loops_increase_size_by_identity(self, adjacency):
        """Â = A + I means the difference should be exactly I."""
        A_hat = add_self_loops(adjacency)
        diff = A_hat - adjacency
        np.testing.assert_array_equal(
            diff, np.eye(adjacency.shape[0]),
            err_msg="Â - A should be the identity matrix."
        )


class TestDegreeMatrix:
    """Test 3: Verify D_ii = Σ_j A_ij and D_ij = 0 for i ≠ j."""

    def test_diagonal_equals_row_sum(self, adjacency):
        """Each diagonal entry of D must equal the row sum of A."""
        A_hat = add_self_loops(adjacency)
        D = compute_degree_matrix(A_hat)
        N = adjacency.shape[0]

        for i in range(N):
            expected_degree = A_hat[i, :].sum()
            assert D[i, i] == expected_degree, \
                f"Node {i}: D[{i},{i}]={D[i,i]} ≠ row_sum={expected_degree}"

    def test_off_diagonal_is_zero(self, adjacency):
        """All off-diagonal entries of D must be zero."""
        A_hat = add_self_loops(adjacency)
        D = compute_degree_matrix(A_hat)
        N = adjacency.shape[0]

        for i in range(N):
            for j in range(N):
                if i != j:
                    assert D[i, j] == 0.0, \
                        f"Off-diagonal D[{i},{j}] = {D[i,j]} should be 0."

    def test_degree_at_least_one_with_self_loops(self, adjacency):
        """With self-loops, every node degree ≥ 1."""
        A_hat = add_self_loops(adjacency)
        D = compute_degree_matrix(A_hat)
        N = adjacency.shape[0]

        for i in range(N):
            assert D[i, i] >= 1.0, \
                f"Node {i} has degree {D[i,i]} < 1 even with self-loops."


class TestNormalizedAdjacency:
    """Test 4: Verify Ã ≈ Ã^T for the undirected graph."""

    def test_normalized_adjacency_is_symmetric(self, adjacency):
        """Normalized adjacency of an undirected graph should be symmetric."""
        A_norm, _, _, _ = symmetric_normalization(adjacency)
        np.testing.assert_allclose(
            A_norm, A_norm.T, atol=1e-10,
            err_msg="Ã is not symmetric! Should be for an undirected graph."
        )

    def test_normalized_values_in_range(self, adjacency):
        """All entries of Ã should be in [0, 1]."""
        A_norm, _, _, _ = symmetric_normalization(adjacency)
        assert np.all(A_norm >= -1e-10), "Ã has negative entries."
        assert np.all(A_norm <= 1.0 + 1e-10), "Ã has entries > 1."

    def test_self_loop_entries_positive(self, adjacency):
        """Diagonal entries of Ã should be positive (from self-loops)."""
        A_norm, _, _, _ = symmetric_normalization(adjacency)
        diag = np.diag(A_norm)
        assert np.all(diag > 0), "Diagonal entries of Ã should be positive."

    def test_normalization_dimensions(self, adjacency):
        """All output matrices should be N × N."""
        N = adjacency.shape[0]
        A_norm, A_hat, D_hat, D_inv_sqrt = symmetric_normalization(adjacency)

        assert A_norm.shape == (N, N)
        assert A_hat.shape == (N, N)
        assert D_hat.shape == (N, N)
        assert D_inv_sqrt.shape == (N, N)
