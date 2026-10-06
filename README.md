# Node Classification Using a Three-Layer Graph Convolutional Network

**An Educational Toy Project for Machine Learning with Graphs**

This project provides a complete, manual implementation of a Graph Convolutional Network (GCN) from scratch using PyTorch. It is designed as an educational tool to demonstrate the explicit mathematical operations behind neighbourhood aggregation, symmetric normalization, and multi-layer graph learning, applied to a synthetic node classification task.

---

## 1. Project Overview

We construct a synthetic "student graph" representing 24 students. 
- **Nodes:** Each node represents a student with 3 features: `study_hours`, `attendance`, `assignment_score`.
- **Labels:** Binary classification: `0 = Fail`, `1 = Pass`.
- **Edges:** Undirected connections representing study groups. The graph exhibits homophily (students of the same class are more likely to connect).
- **Task:** Predict the pass/fail label of unlabelled students in a transductive setting.

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

The project includes 51 automated unit tests verifying mathematical correctness, dimensions, normalization bounds, evaluation metrics, plotting, and model behaviour.

To run the test suite:
```bash
python -m pytest tests/ -v
```

---

## 5. Running the Main Experiment

The main experiment script generates the graph, applies normalization, trains an MLP baseline and 1-, 2-, and 3-layer GCNs across 10 random seeds, runs an ablation study, and produces all summary figures.

Each seed draws a **new train/validation/test split and a new weight initialization**; the graph and node features are the same for every seed. This way the reported spread reflects which students end up in the 6-node test set, not just the starting weights.

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
│   ├── train.py              # Training loop with masking
│   ├── evaluate.py           # Metrics calculation
│   └── visualization.py      # Plotting utilities
│
├── tests/                    # 51 unit tests verifying math and logic
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
│   ├── report.md             # Comprehensive academic report
│   └── viva_questions.md     # Q&A for academic viva preparation
│
├── requirements.txt
└── README.md
```

---

## 7. Results

On the synthetic toy graph across 10 random seeds (mean ± std):

| Model | Parameters | Train accuracy | Test accuracy |
|---|---|---|---|
| MLP (features only) | 50 | 100.0% ± 0.0% | 100.0% ± 0.0% |
| 1-Layer GCN | 6 | 91.4% ± 4.3% | 98.3% ± 5.0% |
| 2-Layer GCN | 40 | 99.3% ± 2.1% | 98.3% ± 5.0% |
| 3-Layer GCN | 104 | 100.0% ± 0.0% | 88.3% ± 10.7% |
| 3-Layer GCN on a random graph | 104 | — | 46.7% ± 16.3% |

- **The MLP is already perfect.** The synthetic features separate Pass and Fail almost completely, so this dataset cannot show the graph *adding* accuracy over features alone.
- **The 3-layer GCN overfits.** It fits every training node (100% train accuracy, near-zero loss) but drops on test nodes. Over-smoothing would blur node representations and hurt *training* accuracy as well, which does not happen here. The more likely cause is 104 parameters fitted to 14 training nodes with no regularization or early stopping.
- **A misleading graph hurts.** On a random graph with the same number of edges (homophily ≈ 0.49 vs 0.79 for the real graph), the 3-layer GCN falls to 46.7%: aggregation then mixes in neighbours of the wrong class.

---

## 8. Limitations

This is an **educational toy project**, and its conclusions should not be extrapolated to real-world performance.
1. The graph is tiny (24 nodes). Each test set has only 6 nodes, so a single mistake moves accuracy by 16.7 percentage points.
2. The dataset is synthetic with easily separable features, so an MLP reaches 100% and the benefit of graph structure cannot be measured against it.
3. Validation accuracy is logged but not used for early stopping or model selection.
4. Operations use dense matrices $O(N^2)$, which is conceptually clear but does not scale to large graphs (which require sparse tensors).

---

## 9. Reproducibility

Random seeds are explicitly set for Python, NumPy, and PyTorch (`src/graph_data.py: set_all_seeds()`). Running the `run_experiments.py` script multiple times will yield identical results.
