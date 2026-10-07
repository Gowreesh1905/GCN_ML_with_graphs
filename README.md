# Node Classification Using a Three-Layer Graph Convolutional Network

**An Educational Toy Project for Machine Learning with Graphs**

This project provides a complete, manual implementation of a Graph Convolutional Network (GCN) from scratch using PyTorch. It is designed as an educational tool to demonstrate the explicit mathematical operations behind neighbourhood aggregation, symmetric normalization, and multi-layer graph learning, applied to a synthetic node classification task.

---

## 1. Project Overview

We construct a synthetic "student graph" representing 200 students (110 Pass, 90 Fail).
- **Nodes:** Each node represents a student with 3 features: `study_hours`, `attendance`, `assignment_score`. The two classes' feature distributions overlap strongly, so features alone are only moderately informative.
- **Labels:** Binary classification: `0 = Fail`, `1 = Pass`.
- **Edges:** Undirected connections representing study groups (753 edges, average degree 7.5). The graph exhibits homophily: 81% of edges join students with the same result.
- **Task:** Predict the pass/fail label of unlabelled students in a transductive setting (120 training, 40 validation and 40 test students per split).

The core of the project is the implementation of a 3-layer GCN without relying on high-level libraries like PyTorch Geometric.

---

## 2. Mathematical Model

The GCN layer performs the following update rule:

$$H^{(l+1)} = \sigma\left(\tilde{A} H^{(l)} W^{(l)}\right)$$

Where:
1. **Adjacency with self-loops:** $\hat{A} = A + I$
2. **Degree Matrix:** $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$
3. **Symmetric Normalization:** $\tilde{A} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$
4. **Neighbourhood Aggregation:** $\tilde{A} H^{(l)}$
5. **Linear Transformation:** $(\tilde{A} H^{(l)}) W^{(l)}$
6. **Activation ($\sigma$):** ReLU (omitted on the final layer which outputs logits).

---

## 3. Installation

This project requires Python 3.12+ and several standard data science libraries.

1. **Clone or download the repository.**
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include `torch`, `numpy`, `networkx`, `matplotlib`, `scikit-learn`, and `pytest`.

---

## 4. Running the Tests

The project includes 55 automated unit tests verifying mathematical correctness, dimensions, normalization bounds, evaluation metrics, early stopping, plotting, and model behaviour.

To run the test suite:
```bash
python -m pytest tests/ -v
```

---

## 5. Running the Main Experiment

The main experiment script generates the graph, applies normalization, trains an MLP baseline and 1-, 2-, and 3-layer GCNs across 10 random seeds, runs an ablation study, and produces all summary figures.

- Each seed draws a **new train/validation/test split and a new weight initialization**; the graph and node features are the same for every seed.
- All models are trained the same way: Adam (learning rate 0.01, weight decay 5e-4), dropout 0.5, and **early stopping** on validation loss (patience 30, at most 500 epochs; the best weights are restored).
- For comparison, a 3-layer GCN is also trained **without** dropout, weight decay or early stopping (fixed 200 epochs).

To run the full pipeline:
```bash
python experiments/run_experiments.py
```
*Note: This script must be run from the root directory of the project.*

---

## 6. Project Structure

```
gcn-toy-project/
│
├── src/                      # Source code
│   ├── graph_data.py         # Synthetic data & graph generation
│   ├── normalization.py      # Explicit adjacency normalization math
│   ├── gcn.py                # Manual PyTorch GCN & MLP implementation
│   ├── train.py              # Training loop with masking and early stopping
│   ├── evaluate.py           # Metrics calculation
│   └── visualization.py      # Plotting utilities
│
├── tests/                    # 55 unit tests verifying math and logic
│
├── experiments/
│   ├── run_experiments.py    # Main pipeline runner
│   └── results/              # Output CSV and JSON metrics
│
├── notebooks/
│   └── gcn_toy_project.ipynb # Interactive educational walkthrough
│
├── figures/                  # Output directory for generated plots
│
├── report/
│   ├── report.md             # Detailed report (Markdown)
│   ├── report.tex            # 5-6 page project report (LaTeX)
│   └── viva_questions.md     # Q&A for academic viva preparation
│
├── requirements.txt
└── README.md
```

---

## 7. Results

On the synthetic 200-student graph across 10 random seeds (mean ± std, 40 test students per seed):

| Model | Parameters | Train accuracy | Test accuracy |
|---|---|---|---|
| MLP (features only) | 50 | 78.8% ± 2.7% | 77.8% ± 5.2% |
| 1-Layer GCN | 6 | 90.9% ± 2.2% | 90.5% ± 5.0% |
| 2-Layer GCN | 40 | 95.5% ± 1.9% | 92.5% ± 4.3% |
| 3-Layer GCN | 104 | 95.7% ± 1.9% | **94.0% ± 4.4%** |
| 3-Layer GCN, no regularization | 104 | 96.3% ± 1.7% | 94.3% ± 4.3% |
| 3-Layer GCN on a random graph | 104 | — | 50.8% ± 10.7% |

- **The graph adds a lot.** Every GCN beats the feature-only MLP by 13–16 points, and on every one of the 10 splits. The features overlap, so a student cannot be classified reliably in isolation; averaging over their study group, most of whom share their result, supplies the missing information.
- **Depth helps a little.** Accuracy rises from 1 to 3 layers (90.5% → 92.5% → 94.0%), but the steps are smaller than the variation between splits, so this is a trend rather than a firm result. There is no sign of over-smoothing at 3 layers.
- **The graph must be meaningful.** On a random graph with the same number of edges (homophily ≈ 0.50 vs 0.81), the 3-layer GCN falls to 50.8%, below both the MLP and the 55% majority-class guess: aggregation mixes each student with unrelated students.
- **Regularization makes no measurable difference here.** With 120 labelled students the 3-layer GCN does not overfit, with or without dropout, weight decay and early stopping. Early stopping still removes the need to pick an epoch count: the 1-layer GCN keeps improving for about 440 epochs, the 3-layer GCN stops after about 120.

---

## 8. Limitations

This is an **educational toy project**, and its conclusions should not be extrapolated to real-world performance.
1. The dataset is synthetic. Its homophily (0.81) and feature overlap are chosen parameters, so the size of the GCN's advantage reflects those choices, not real student data.
2. With 200 nodes and 40 test students per split, one mistake moves accuracy by 2.5 points; differences between the GCN depths are within this noise.
3. The graph and features are fixed across seeds; only the split and initialization vary.
4. Operations use dense matrices $O(N^2)$, which is conceptually clear but does not scale to large graphs (which require sparse tensors).

---

## 9. Reproducibility

Random seeds are explicitly set for Python, NumPy, and PyTorch (`src/graph_data.py: set_all_seeds()`). Running the `run_experiments.py` script multiple times will yield identical results.
