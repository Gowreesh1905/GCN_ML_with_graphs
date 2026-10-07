"""
test_graph_data.py — Tests for Graph Data Generation

Tests:
- Graph symmetry (A = A^T for undirected graph)
- Class distribution
- Feature shape
- Mask validity
- Standardization correctness
"""

import numpy as np
import pytest

from src.graph_data import (
    build_dataset,
    generate_student_features,
    generate_student_graph,
    get_adjacency_matrix,
    create_train_val_test_masks,
    standardize_features,
)


@pytest.fixture
def dataset():
    """Build the standard dataset for testing."""
    return build_dataset(seed=42)


class TestGraphSymmetry:
    """Test 1: Verify A = A^T for the undirected graph."""

    def test_adjacency_is_symmetric(self, dataset):
        """The adjacency matrix of an undirected graph must be symmetric."""
        A = dataset["adjacency"]
        np.testing.assert_array_equal(
            A, A.T,
            err_msg="Adjacency matrix is not symmetric! Graph should be undirected."
        )

    def test_adjacency_binary(self, dataset):
        """Adjacency entries should be 0 or 1."""
        A = dataset["adjacency"]
        unique_vals = np.unique(A)
        assert set(unique_vals).issubset({0.0, 1.0}), \
            f"Adjacency has non-binary values: {unique_vals}"

    def test_no_initial_self_loops(self, dataset):
        """The original adjacency should have no self-loops."""
        A = dataset["adjacency"]
        diag = np.diag(A)
        np.testing.assert_array_equal(
            diag, np.zeros(len(diag)),
            err_msg="Original adjacency should have no self-loops."
        )


class TestFeatures:
    """Test feature generation."""

    def test_feature_shape(self, dataset):
        """Features should be N × 3."""
        X = dataset["features"]
        N = dataset["stats"]["num_nodes"]
        assert X.shape == (N, 3), f"Expected ({N}, 3), got {X.shape}"

    def test_label_shape(self, dataset):
        """Labels should have length N."""
        labels = dataset["labels"]
        N = dataset["stats"]["num_nodes"]
        assert len(labels) == N

    def test_label_values(self, dataset):
        """Labels should be 0 or 1."""
        labels = dataset["labels"]
        assert set(np.unique(labels)) == {0, 1}


class TestMasks:
    """Test train/validation/test mask creation."""

    def test_masks_non_overlapping(self, dataset):
        """Train, val, and test masks should not overlap."""
        train = dataset["train_mask"]
        val = dataset["val_mask"]
        test = dataset["test_mask"]

        assert not np.any(train & val), "Train and val masks overlap!"
        assert not np.any(train & test), "Train and test masks overlap!"
        assert not np.any(val & test), "Val and test masks overlap!"

    def test_masks_cover_all_nodes(self, dataset):
        """Every node should be in exactly one split."""
        train = dataset["train_mask"]
        val = dataset["val_mask"]
        test = dataset["test_mask"]

        assert np.all(train | val | test), "Some nodes are not in any split!"

    def test_approximate_split_ratios(self, dataset):
        """Split ratios should be approximately 60/20/20."""
        N = dataset["stats"]["num_nodes"]
        n_train = dataset["train_mask"].sum()
        n_val = dataset["val_mask"].sum()
        n_test = dataset["test_mask"].sum()

        assert abs(n_train / N - 0.6) < 0.15, f"Train ratio: {n_train/N:.2f}"
        assert abs(n_val / N - 0.2) < 0.15, f"Val ratio: {n_val/N:.2f}"
        assert abs(n_test / N - 0.2) < 0.15, f"Test ratio: {n_test/N:.2f}"

    def test_split_seed_defaults_to_seed(self, dataset):
        """Without split_seed, the split is drawn from `seed`."""
        explicit = build_dataset(seed=42, split_seed=42)
        np.testing.assert_array_equal(dataset["test_mask"], explicit["test_mask"])

    def test_split_seed_changes_split_not_graph(self, dataset):
        """A different split_seed draws a new split over the same graph."""
        other = build_dataset(seed=42, split_seed=43)
        assert not np.array_equal(dataset["test_mask"], other["test_mask"])
        np.testing.assert_array_equal(dataset["adjacency"], other["adjacency"])
        np.testing.assert_array_equal(dataset["labels"], other["labels"])
        np.testing.assert_array_equal(dataset["features_raw"], other["features_raw"])


class TestStandardization:
    """Test feature standardization."""

    def test_training_features_zero_mean(self, dataset):
        """Standardized training features should have ~zero mean."""
        X = dataset["features"]
        train_mask = dataset["train_mask"]
        train_means = X[train_mask].mean(axis=0)
        np.testing.assert_allclose(
            train_means, 0.0, atol=1e-6,
            err_msg="Training features should have zero mean after standardization."
        )

    def test_training_features_unit_std(self, dataset):
        """Standardized training features should have ~unit std."""
        X = dataset["features"]
        train_mask = dataset["train_mask"]
        train_stds = X[train_mask].std(axis=0)
        np.testing.assert_allclose(
            train_stds, 1.0, atol=0.1,
            err_msg="Training features should have unit std after standardization."
        )


class TestGraphStatistics:
    """Test graph statistics computation."""

    def test_graph_is_undirected(self, dataset):
        assert not dataset["stats"]["is_directed"]

    def test_node_count(self, dataset):
        assert dataset["stats"]["num_nodes"] == 200  # 110 Pass + 90 Fail


class TestFeatureGap:
    """feature_gap controls how far apart the class feature means are."""

    def test_smaller_gap_brings_class_means_closer(self):
        def class_gap(gap):
            X, y = generate_student_features(n_pass=500, n_fail=500, seed=0, feature_gap=gap)
            return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)

        wide, narrow = class_gap(1.0), class_gap(0.25)
        assert np.all(wide > 0) and np.all(narrow > 0), "Pass students should score higher"
        assert np.all(narrow < wide), "A smaller gap should bring the classes closer"

    def test_has_edges(self, dataset):
        assert dataset["stats"]["num_edges"] > 0

    def test_homophily_in_range(self, dataset):
        h = dataset["stats"]["homophily_ratio"]
        assert 0 <= h <= 1, f"Homophily out of range: {h}"
