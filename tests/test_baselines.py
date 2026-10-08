"""
test_baselines.py — Tests for the Neighbour Majority Vote Baseline
"""

import numpy as np

from src.baselines import evaluate_neighbour_vote, neighbour_majority_vote


def _path_graph(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return A


def test_majority_of_labelled_neighbours_wins():
    # Star: node 0 is linked to 1, 2, 3 (labels 1, 1, 0)
    A = np.zeros((4, 4))
    for j in (1, 2, 3):
        A[0, j] = A[j, 0] = 1
    labels = np.array([0, 1, 1, 0])
    train = np.array([False, True, True, True])
    assert neighbour_majority_vote(A, labels, train)[0] == 1


def test_unlabelled_neighbours_do_not_vote():
    # Node 0's only labelled neighbour has label 0; its unlabelled
    # neighbours (labels 1) must be ignored.
    A = np.zeros((4, 4))
    for j in (1, 2, 3):
        A[0, j] = A[j, 0] = 1
    labels = np.array([1, 0, 1, 1])
    train = np.array([False, True, False, False])
    assert neighbour_majority_vote(A, labels, train)[0] == 0


def test_isolated_or_tied_nodes_get_training_majority():
    # Path 0-1-2 plus isolated node 3. Node 1's neighbours tie (0 vs 1).
    A = _path_graph(3)
    A = np.pad(A, ((0, 1), (0, 1)))
    labels = np.array([0, 1, 1, 1])
    train = np.array([True, False, True, False])   # training labels 0 and 1
    preds = neighbour_majority_vote(A, labels, train)
    # Training majority with a 1:1 tie falls back to class 1
    assert preds[1] == 1 and preds[3] == 1


def test_perfect_homophily_gives_perfect_accuracy():
    # Two disconnected cliques, one per class
    A = np.zeros((6, 6))
    for group in ((0, 1, 2), (3, 4, 5)):
        for i in group:
            for j in group:
                if i != j:
                    A[i, j] = 1
    labels = np.array([0, 0, 0, 1, 1, 1])
    train = np.array([True, True, False, True, True, False])
    test = ~train
    assert evaluate_neighbour_vote(A, labels, train, test)["accuracy"] == 1.0
