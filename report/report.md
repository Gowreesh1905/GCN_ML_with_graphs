# Node Classification Using a Three-Layer Graph Convolutional Network: Implementation, Mathematical Analysis, and Experimental Evaluation

---

## 1. Abstract

This project investigates how graph structure and neighbourhood aggregation affect node classification using a Graph Convolutional Network (GCN). We implement a three-layer GCN from scratch using PyTorch, without relying on high-level graph neural network libraries, to classify 200 students in a synthetic study-group graph as Pass or Fail. The project demonstrates the complete mathematical pipeline from adjacency matrix construction through symmetric normalization to multi-layer neighbourhood aggregation. The students' features overlap strongly between classes, so a feature-only MLP reaches only 77.8% test accuracy. Across ten random seeds — each drawing a new train/validation/test split and a new weight initialization — GCNs with 1, 2 and 3 layers reach 90.5%, 92.5% and 94.0%, beating the MLP on every split. A graph-structure ablation shows that on a random graph (no homophily) the 3-layer GCN collapses to 50.8%, below the majority-class rate. All models use dropout, weight decay and early stopping; an unregularized 3-layer GCN performs the same (94.3%), showing no overfitting with 120 labelled nodes. All results are reproducible and honestly reported without overclaiming.

---

## 2. Introduction

Many real-world datasets naturally exhibit graph structure: social networks, citation networks, molecular structures, and transportation systems. In these domains, entities (nodes) are interconnected by relationships (edges), and the structure of these connections carries meaningful information.

Conventional neural networks such as Multi-Layer Perceptrons (MLPs) process each data point independently, ignoring the relational structure between them. When applied to graph-structured data, an MLP can use only the features of each node in isolation, discarding potentially valuable information about the node's neighbourhood.

**Graph Neural Networks (GNNs)** address this limitation by designing neural network architectures that explicitly operate on graph-structured inputs. Among GNN variants, the **Graph Convolutional Network (GCN)** introduced by Kipf and Welling (2017) is one of the most widely studied. A GCN performs *neighbourhood aggregation*: each node's representation is updated by combining its own features with those of its neighbours, weighted by the graph structure.

**Node classification** is a fundamental task in graph machine learning: given a partially labelled graph, predict the labels of the remaining nodes. This is a *transductive* learning setting where the entire graph structure is visible during both training and inference, but only a subset of node labels is used for training.

This project implements a three-layer GCN from scratch to perform node classification on a synthetic student-performance graph, providing a complete educational walkthrough of the mathematics, implementation, and experimental evaluation.

---

## 3. Problem Statement

Given:
- A graph $G = (V, E)$ with $N = |V|$ nodes and $|E|$ edges
- A node feature matrix $X \in \mathbb{R}^{N \times F}$ where $F = 3$
- Node labels $Y \in \{0, 1\}^N$ (binary classification: Fail/Pass)
- A train/validation/test split of nodes

**Learn** a function $f: (G, X) \rightarrow \hat{Y}$ using a three-layer GCN to predict node classes, leveraging both node features and graph structure.

---

## 4. Objectives

1. Represent a graph mathematically using adjacency and degree matrices.
2. Implement symmetric adjacency normalization explicitly.
3. Build a GCN layer that performs neighbourhood aggregation via matrix multiplication.
4. Train a three-layer GCN for node classification.
5. Compare GCN performance against an MLP baseline and GCNs of varying depth.
6. Investigate the effect of graph structure through ablation studies.
7. Analyze computational complexity.
8. Discuss limitations honestly.

---

## 5. Background

### 5.1 Graphs

A **graph** $G = (V, E)$ consists of a set of **nodes** (vertices) $V$ and **edges** $E \subseteq V \times V$. In an **undirected** graph, an edge $(i, j)$ implies $(j, i)$.

### 5.2 Adjacency Matrix

The **adjacency matrix** $A \in \mathbb{R}^{N \times N}$ encodes graph connectivity:

$$A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

For an undirected graph: $A = A^T$.

### 5.3 Degree Matrix

The **degree matrix** $D$ is a diagonal matrix where $D_{ii} = \sum_j A_{ij}$ is the number of edges connected to node $i$.

### 5.4 Node Features

Each node $i$ has an associated feature vector $x_i \in \mathbb{R}^F$. The complete feature matrix is $X \in \mathbb{R}^{N \times F}$.

### 5.5 Neighbourhood and Message Passing

The **neighbourhood** $\mathcal{N}(i)$ of node $i$ is the set of nodes connected to $i$. **Message passing** is the paradigm where nodes exchange information with their neighbours to update their representations.

### 5.6 Graph Convolution

A **graph convolution** generalizes the convolution operation from regular grids (images) to irregular graphs. Rather than sliding a fixed kernel, graph convolution aggregates information from a node's neighbourhood.

### 5.7 Graph Convolutional Network (GCN)

The GCN (Kipf & Welling, 2017) defines a layer-wise propagation rule that performs neighbourhood aggregation followed by a linear transformation. Multiple GCN layers allow information to propagate across multiple hops.

---

## 6. Mathematical Foundation

### 6.1 Adjacency Matrix

For our graph with $N = 200$ nodes:

$$A \in \mathbb{R}^{200 \times 200}, \quad A_{ij} \in \{0, 1\}, \quad A = A^T$$

### 6.2 Adding Self-Loops

$$\hat{A} = A + I$$

**Why self-loops?** Without self-loops, when a node aggregates information from its neighbours, it would lose its own features. Adding self-loops ensures that each node includes itself in its neighbourhood:

$$\mathcal{N}(i) \leftarrow \mathcal{N}(i) \cup \{i\}$$

After adding self-loops, every diagonal entry $\hat{A}_{ii} = 1$.

### 6.3 Degree Matrix of $\hat{A}$

$$\hat{D}_{ii} = \sum_j \hat{A}_{ij}$$

This is the degree of node $i$ in the augmented graph (including the self-loop). All off-diagonal entries are zero: $\hat{D}_{ij} = 0$ for $i \neq j$.

**Important distinction:** The original degree $d_i = \sum_j A_{ij}$ counts only actual neighbours. The augmented degree $\hat{d}_i = d_i + 1$ includes the self-loop.

### 6.4 Symmetric Normalization

$$\tilde{A} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$$

**Why normalize?**

- **Degree differences:** Without normalization, high-degree nodes would accumulate much larger aggregated feature values than low-degree nodes, simply because they have more neighbours.
- **Magnitude stability:** Normalization keeps the aggregated features at a comparable scale across nodes regardless of degree.
- **Scale stability across layers:** Without normalization, feature magnitudes could grow or shrink uncontrollably with each GCN layer.
- **Symmetric normalization** $\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$ considers the degrees of both the source and target nodes, providing balanced aggregation.

For an undirected graph: $\tilde{A} = \tilde{A}^T$ (verified numerically).

### 6.5 GCN Layer Propagation Rule

Starting from the input:

$$H^{(0)} = X$$

Each GCN layer computes:

$$H^{(l+1)} = \sigma\left(\tilde{A} H^{(l)} W^{(l)}\right)$$

where:
- $H^{(l)} \in \mathbb{R}^{N \times F_l}$ — node representations at layer $l$
- $\tilde{A} \in \mathbb{R}^{N \times N}$ — normalized adjacency matrix
- $W^{(l)} \in \mathbb{R}^{F_l \times F_{l+1}}$ — learnable weight matrix
- $\sigma$ — activation function (ReLU for hidden layers)

**Dimensional analysis:**

$$\underbrace{\tilde{A}}_{N \times N} \underbrace{H^{(l)}}_{N \times F_l} = \underbrace{(\tilde{A} H^{(l)})}_{N \times F_l}$$

$$\underbrace{(\tilde{A} H^{(l)})}_{N \times F_l} \underbrace{W^{(l)}}_{F_l \times F_{l+1}} = \underbrace{H^{(l+1)}}_{N \times F_{l+1}}$$

The operation $\tilde{A} H^{(l)}$ is the **neighbourhood aggregation** step: each node's representation becomes a weighted average of its neighbours' (and its own) representations. The operation $\cdot W^{(l)}$ is the **linear transformation**: projecting the aggregated features into a new feature space.

### 6.6 Loss Function

Cross-entropy loss on training nodes:

$$\mathcal{L} = -\sum_{i \in \mathcal{T}} \log P_{i, y_i}$$

where $\mathcal{T}$ is the training-node set and $P = \text{softmax}(H^{(L)})$.

In practice, we pass logits directly to `nn.CrossEntropyLoss`, which internally computes `log_softmax` for numerical stability.

---

## 7. Three-Layer Architecture

```
X (N × 3) — Input features
     │
GCN Layer 1: W⁰ ∈ ℝ^(3×8)
     │  H⁽¹⁾ = ReLU(Ã X W⁰)         — N × 8
     │
GCN Layer 2: W¹ ∈ ℝ^(8×8)
     │  H⁽²⁾ = ReLU(Ã H⁽¹⁾ W¹)     — N × 8
     │
GCN Layer 3: W² ∈ ℝ^(8×2)
     │  H⁽³⁾ = Ã H⁽²⁾ W²             — N × 2  (logits)
     │
Softmax → P ∈ [0,1]^(N×2)
     │
Classification (argmax)
```

**Why no ReLU after the final layer?**

The final layer produces **logits** — raw scores for each class. Applying ReLU would clip negative values to zero, destroying important information about class preference. The softmax function (applied implicitly by `CrossEntropyLoss`) converts logits to probabilities. Applying softmax before `CrossEntropyLoss` would be redundant and numerically unstable.

---

## 8. Dataset and Graph Construction

### 8.1 Synthetic Data

We generate a synthetic student-performance graph:

| Property | Value |
|---|---|
| Number of nodes | 200 |
| Number of edges | 753 |
| Number of classes | 2 (Fail=0, Pass=1) |
| Features per node | 3 |
| Class distribution | Fail: 90, Pass: 110 |
| Graph density | 0.038 |
| Average degree | 7.53 |
| Min/Max degree | 0 / 15 |
| Directed | No |
| Self-loops | No (before normalization) |
| Connected | No — one isolated student; the rest form one component (diameter 5) |

### 8.2 Node Features

Each node (student) has three features drawn from class-dependent normal distributions. The class means are controlled by a `feature_gap` parameter: 1.0 gives well-separated reference means (Pass: 7 h / 80% / 75; Fail: 3 h / 50% / 40), and smaller values pull both classes toward the midpoint. We use **feature_gap = 0.25**:

| Feature | Pass students | Fail students |
|---|---|---|
| study_hours | μ=5.5, σ=1.5 | μ=4.5, σ=1.5 |
| attendance | μ=68.75, σ=10 | μ=61.25, σ=12 |
| assignment_score | μ=61.9, σ=10 | μ=53.1, σ=12 |

The class means differ by less than one standard deviation, so the classes overlap strongly. This is deliberate: with well-separated features, a model that ignores the graph classifies every student correctly, leaving nothing for the graph to contribute (this happened in an earlier version of the project).

### 8.3 Graph Generation

Edges are generated stochastically for every pair of students:
- **Same-class probability:** $p_{\text{same}} = 0.06$
- **Cross-class probability:** $p_{\text{cross}} = 0.015$

This creates meaningful but imperfect homophily with an average degree of about 7.5.

### 8.4 Homophily Analysis

$$H = \frac{|\{(i,j) \in E : y_i = y_j\}|}{|E|} = \frac{608}{753} = 0.807$$

| Metric | Value |
|---|---|
| Total edges | 753 |
| Same-label edges | 608 |
| Cross-label edges | 145 |
| Homophily ratio | 0.807 |

A homophily ratio of 0.807 indicates strong homophily: students mostly connect with others who have the same result, but about one edge in five crosses between the classes.

### 8.5 Feature Preprocessing

Features are standardized using training-set statistics to prevent data leakage:

$$x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

The mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ are computed only from training nodes and applied to all nodes.

### 8.6 Train/Validation/Test Split

| Split | Nodes | Percentage |
|---|---|---|
| Training | 120 | 60% |
| Validation | 40 | 20% |
| Test | 40 | 20% |

The table shows the split for seed 42. In the multi-seed experiments (§11), every seed draws a **different** random split of the same graph.

**Transductive setting:** The entire graph structure (all edges between all 200 nodes) is visible during training. Only the *labels* of validation and test nodes are hidden. This is standard for node classification on a single graph. The model can leverage the graph connections of unlabelled nodes during message passing.

With 40 test nodes per split, one misclassification changes accuracy by 2.5 percentage points.

---

## 9. Implementation

### 9.1 GCN Layer

The core GCN layer implements $H' = \tilde{A} H W$:

```python
class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # W: F_in × F_out — learnable weight matrix
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A_norm, H):
        # Step 1: Ã @ H — neighbourhood aggregation (N×N @ N×F_in = N×F_in)
        support = torch.mm(A_norm, H)
        # Step 2: support @ W — linear transformation (N×F_in @ F_in×F_out = N×F_out)
        H_out = torch.mm(support, self.weight)
        return H_out
```

**Weight initialization:** Xavier uniform initialization sets weights from $U(-a, a)$ where $a = \sqrt{6 / (F_{\text{in}} + F_{\text{out}})}$. This aims to maintain activation variance across layers, mitigating vanishing/exploding gradients.

### 9.2 Multi-Layer GCN

```python
class GCN(nn.Module):
    def forward(self, A_norm, X):
        H = X                           # H⁰ = X
        for i, layer in enumerate(self.layers):
            H = layer(A_norm, H)         # H' = Ã H W
            if i < len(self.layers) - 1:
                H = F.relu(H)            # ReLU (not on final layer)
        return H                         # Logits
```

### 9.3 Normalization

The symmetric normalization $\tilde{A} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$ is implemented explicitly:

```python
def symmetric_normalization(A):
    A_hat = A + np.eye(N)                          # Â = A + I
    D_hat = np.diag(A_hat.sum(axis=1))             # D̂
    inv_sqrt = 1.0 / np.sqrt(np.diag(D_hat))       # D̂^(-1/2) diagonal
    D_inv_sqrt = np.diag(inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt       # Ã
    return A_norm
```

Zero-degree nodes are handled safely by setting their inverse square root to 0. (With self-loops every degree is at least 1 — including the one isolated student in our graph.)

### 9.4 Training with Early Stopping

Training is full-batch with Adam (learning rate 0.01, weight decay $5 \times 10^{-4}$) and dropout 0.5 on hidden layers. After every epoch the validation loss is computed; if it has not improved for 30 epochs, training stops and the weights from the best epoch are restored:

```python
if val_loss < best_val_loss:
    best_val_loss, best_epoch = val_loss, epoch
    best_state = copy.deepcopy(model.state_dict())
elif epoch - best_epoch >= patience:
    break                                   # no improvement for `patience` epochs
...
model.load_state_dict(best_state)           # keep the best-validation weights
```

Training runs for at most 500 epochs.

---

## 10. Testing

All 55 tests passed (`python -m pytest`).

| Test | Description | Status |
|---|---|---|
| **Test 1** — Graph symmetry | $A = A^T$ | ✅ Pass |
| **Test 2** — Self-loops | $\text{diag}(\hat{A}) = 1$ | ✅ Pass |
| **Test 3** — Degree matrix | $D_{ii} = \sum_j A_{ij}$, $D_{ij}=0$ for $i \neq j$ | ✅ Pass |
| **Test 4** — Normalized symmetry | $\tilde{A} \approx \tilde{A}^T$ | ✅ Pass |
| **Test 5** — Dimensions | H1: N×8, H2: N×8, H3: N×2 | ✅ Pass |
| **Test 6** — Forward pass | Finite outputs, no NaN/Inf | ✅ Pass |
| **Test 7** — Gradient flow | All gradients non-None | ✅ Pass |
| **Test 8** — Loss decreases | Final loss < initial loss (>20% reduction) | ✅ Pass |
| **Test 9** — Probability validity | $0 \leq p_i \leq 1$, $\sum_c p_{ic} \approx 1$ | ✅ Pass |
| **Test 10** — Reproducibility | Same seed → same results | ✅ Pass |

Additional tests verified: binary adjacency values, no initial self-loops, feature shapes, mask non-overlap, mask coverage, approximate split ratios, per-seed splits over a fixed graph, zero-mean/unit-std training features, graph statistics, normalized value ranges, positive diagonal entries, parameter counts, a 2 × 2 confusion matrix even when only one class is present, homophily consistency, the effect of `feature_gap`, early stopping (stops early, records and restores the best-validation weights), and smoke tests for the graph and PCA plots.

---

## 11. Experimental Setup

| Setting | Value |
|---|---|
| Python | 3.14.3 |
| PyTorch | 2.13.0 |
| NumPy | 2.4.3 |
| NetworkX | 3.6.1 |
| scikit-learn | 1.9.0 |
| Optimizer | Adam, learning rate 0.01, weight decay 5e-4 |
| Dropout | 0.5 (hidden layers, all models) |
| Early stopping | Validation loss, patience 30, at most 500 epochs, best weights restored |
| Hidden dimensions | 8 |
| Loss function | Cross-entropy (on logits) |
| Random seeds | 42–51 (10 seeds) |
| What a seed controls | Train/validation/test split **and** weight initialization |
| Fixed across seeds | Graph, features and labels (generated with seed 42) |
| Evaluation metrics | Accuracy, Precision, Recall, F1, Confusion Matrix; train accuracy reported alongside test accuracy |

**Models:** MLP (features only), GCN-1, GCN-2, GCN-3, all trained with the settings above; **GCN-3 (no reg)**, the same architecture with no dropout, no weight decay and no early stopping (fixed 200 epochs); and **GCN-3 on a random graph** (§12.4).

---

## 12. Results

### 12.1 Multi-Seed Results (10 Seeds)

| Model | Train Acc (mean ± std) | Test Acc (mean ± std) | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Parameters |
|---|---|---|---|---|---|---|
| MLP | 0.788 ± 0.027 | 0.778 ± 0.052 | 0.795 ± 0.076 | 0.773 ± 0.075 | 0.823 ± 0.101 | 50 |
| GCN-1 | 0.909 ± 0.022 | 0.905 ± 0.050 | 0.902 ± 0.063 | 0.938 ± 0.054 | 0.872 ± 0.086 | 6 |
| GCN-2 | 0.955 ± 0.019 | 0.925 ± 0.043 | 0.927 ± 0.051 | 0.923 ± 0.059 | 0.936 ± 0.077 | 40 |
| GCN-3 | 0.957 ± 0.019 | 0.940 ± 0.044 | 0.942 ± 0.047 | 0.932 ± 0.062 | 0.954 ± 0.055 | 104 |
| GCN-3 (no reg) | 0.963 ± 0.017 | 0.943 ± 0.043 | 0.946 ± 0.044 | 0.935 ± 0.067 | 0.962 ± 0.051 | 104 |

Mean epoch of the kept weights (early stopping): MLP 124, GCN-1 441, GCN-2 249, GCN-3 123.

### 12.2 Individual Seed Results (Test Accuracy)

Each column is a different train/validation/test split (and initialization). Every test set has 40 nodes, so accuracy moves in steps of 0.025.

| Model | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.775 | 0.750 | 0.750 | 0.850 | 0.875 | 0.750 | 0.700 | 0.825 | 0.750 | 0.750 |
| GCN-1 | 0.950 | 0.825 | 0.950 | 0.900 | 0.975 | 0.950 | 0.875 | 0.925 | 0.850 | 0.850 |
| GCN-2 | 0.975 | 0.850 | 0.900 | 0.950 | 1.000 | 0.925 | 0.925 | 0.900 | 0.950 | 0.875 |
| GCN-3 | 0.925 | 0.900 | 0.950 | 0.950 | 1.000 | 0.950 | 0.875 | 0.975 | 1.000 | 0.875 |
| GCN-3 (no reg) | 0.925 | 0.900 | 0.975 | 0.950 | 1.000 | 0.950 | 0.925 | 0.950 | 1.000 | 0.850 |

Every GCN beats the MLP on every seed.

### 12.3 Confusion Matrix (3-Layer GCN, seed=42, test set)

|  | Predicted Fail | Predicted Pass |
|---|---|---|
| **Actual Fail** | 10 | 3 |
| **Actual Pass** | 0 | 27 |

Test accuracy 0.925; training stopped after 142 epochs, keeping the weights from epoch 112.

### 12.4 Graph Ablation Study

For each seed, GCN-3 is retrained on an Erdős–Rényi random graph with the same expected number of edges, using the same split and training settings as the original-graph run for that seed.

| Configuration | Graph homophily | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---|---|
| GCN-3 (original graph) | 0.807 | 0.940 ± 0.044 | 0.942 ± 0.047 |
| GCN-3 (random graph) | 0.498 ± 0.014 | 0.508 ± 0.107 | 0.594 ± 0.207 |
| MLP (features only) | — | 0.778 ± 0.052 | 0.795 ± 0.076 |

### 12.5 Visualizations

The following figures were generated and saved to the `figures/` directory:

1. **fig1_graph_true_labels.png** — Graph coloured by true class
2. **fig2_adjacency_matrix.png** — Adjacency matrix heatmap
3. **fig3_normalized_adjacency.png** — Normalized adjacency matrix heatmap
4. **fig4_loss_curve.png** — Training and validation loss (the dotted line marks the epoch whose weights early stopping kept)
5. **fig5_accuracy_curve.png** — Training and validation accuracy
6. **fig6_graph_predicted.png** — Graph coloured by predicted class, misclassified nodes marked with a black X (seed 42: 8 of 200)
7. **fig7_confusion_matrix.png** — Confusion matrix
8. **fig8_model_comparison_accuracy.png** — Model comparison (accuracy, 10 seeds)
9. **fig8b_model_comparison_f1.png** — Model comparison (F1, 10 seeds)
10. **fig9_pca_layer1.png** — PCA of Layer 1 embeddings
11. **fig10_pca_layer2.png** — PCA of Layer 2 embeddings

---

## 13. Mathematical Walkthrough

### 4-Node Example

Consider a 4-node cycle graph with 3 features per node.

**Step 1: Adjacency matrix** $A$ (4 × 4)
```
[[0, 1, 0, 1],
 [1, 0, 1, 0],
 [0, 1, 0, 1],
 [1, 0, 1, 0]]
```

**Step 2: Self-loops** $\hat{A} = A + I$ (4 × 4)
```
[[1, 1, 0, 1],
 [1, 1, 1, 0],
 [0, 1, 1, 1],
 [1, 0, 1, 1]]
```

**Step 3: Degree matrix** $\hat{D}$ (all degrees = 3)
```
diag([3, 3, 3, 3])
```

**Step 4: Inverse sqrt** $\hat{D}^{-1/2}$
```
diag([0.5774, 0.5774, 0.5774, 0.5774])
```

**Step 5: Normalized adjacency** $\tilde{A} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$ (4 × 4)
```
[[0.3333, 0.3333, 0.0000, 0.3333],
 [0.3333, 0.3333, 0.3333, 0.0000],
 [0.0000, 0.3333, 0.3333, 0.3333],
 [0.3333, 0.0000, 0.3333, 0.3333]]
```

Each non-zero entry is $\frac{1}{\sqrt{d_i} \cdot \sqrt{d_j}} = \frac{1}{3}$ since all degrees equal 3.

**Step 6: Features** $X$ (4 × 3)
```
[[ 5, 70, 60],   ← Student 0
 [ 8, 90, 85],   ← Student 1
 [ 2, 40, 30],   ← Student 2
 [ 6, 75, 65]]   ← Student 3
```

**Step 7: Aggregation** $\tilde{A} X$ (4 × 3) — each node gets averaged neighbour features:
```
[[6.333, 78.333, 70.000],
 [5.000, 66.667, 58.333],
 [5.333, 68.333, 60.000],
 [4.333, 61.667, 51.667]]
```

**Step 8: Transform** $(\tilde{A} X) W$ (4 × 2) — linear projection:
```
[[109.28, 54.83],
 [ 91.51, 47.19],
 [ 94.17, 48.31],
 [ 82.25, 44.71]]
```

**Step 9: ReLU** — all values positive, so unchanged.

**Dimension summary:**
| Matrix | Dimensions |
|---|---|
| $A$ | $4 \times 4$ |
| $\hat{A} = A + I$ | $4 \times 4$ |
| $\hat{D}$ | $4 \times 4$ |
| $\hat{D}^{-1/2}$ | $4 \times 4$ |
| $\tilde{A}$ | $4 \times 4$ |
| $X$ | $4 \times 3$ |
| $W$ | $3 \times 2$ |
| $\tilde{A}X$ | $4 \times 3$ |
| $\tilde{A}XW$ | $4 \times 2$ |

---

## 14. Complexity Analysis

### 14.1 Time Complexity

Let $N$ = nodes, $E$ = edges, $F_l$ = feature dimension at layer $l$, $L$ = layers.

**Dense representation (used in this project):**

| Operation | Complexity |
|---|---|
| $\tilde{A} H$ (aggregation) | $O(N^2 F_l)$ |
| $H W$ (transformation) | $O(N F_l F_{l+1})$ |
| **One GCN layer** | $O(N^2 F_l + N F_l F_{l+1})$ |

**Sparse representation (for large graphs):**

| Operation | Complexity |
|---|---|
| $\tilde{A} H$ (aggregation) | $O(E F_l)$ |
| $H W$ (transformation) | $O(N F_l F_{l+1})$ |
| **One GCN layer** | $O(E F_l + N F_l F_{l+1})$ |

**Training over $T$ epochs (sparse):**

$$O\left(T \sum_{l=0}^{L-1} (E F_l + N F_l F_{l+1})\right)$$

For our project ($N=200$, $E=753$):
- Layer 1: $O(753 \times 3 + 200 \times 3 \times 8) = O(2{,}259 + 4{,}800) = O(7{,}059)$
- Layer 2: $O(753 \times 8 + 200 \times 8 \times 8) = O(6{,}024 + 12{,}800) = O(18{,}824)$
- Layer 3: $O(753 \times 8 + 200 \times 8 \times 2) = O(6{,}024 + 3{,}200) = O(9{,}224)$ (aggregation happens on the 8-dimensional input, before the projection to 2)
- Per epoch: $O(35{,}107)$
- Total: early stopping ended seed-42 training after 142 epochs, so about $O(5 \times 10^6)$ operations

### 14.2 Parameter Count

| Layer | Parameters | Count |
|---|---|---|
| GCN Layer 1 | $W^{(0)}: 3 \times 8$ | 24 |
| GCN Layer 2 | $W^{(1)}: 8 \times 8$ | 64 |
| GCN Layer 3 | $W^{(2)}: 8 \times 2$ | 16 |
| **Total** | | **104** |

No bias parameters are used in the GCN layers.

### 14.3 Memory Complexity

| Component | Dense | Sparse |
|---|---|---|
| Adjacency matrix | $O(N^2)$ | $O(N + E)$ |
| Feature matrix | $O(NF)$ | $O(NF)$ |
| Activations (forward/backward) | $O(N \sum_l F_l)$ | $O(N \sum_l F_l)$ |

For this toy project: $O(200^2) = O(40{,}000)$ entries for dense adjacency, of which only $2 \times 753 + 200 = 1{,}706$ are non-zero. For a real graph with millions of nodes, sparse representations are essential.

### 14.4 Adjacency Construction

Generating edges with the stochastic block model: $O(N^2)$ because every pair is considered.

---

## 15. Ablation Studies

### 15.1 MLP vs GCN

The MLP baseline reaches only 0.778 ± 0.052, while every GCN exceeds 0.90, and each GCN beats the MLP on every one of the 10 splits. The reason is the combination of overlapping features and a homophilic graph:

1. **Ambiguous features.** The class means differ by less than one standard deviation, so many students look like the other class when judged on their own features. The MLP cannot do better than the features allow.
2. **Informative neighbourhoods.** 81% of edges join students with the same result. Aggregation $\tilde{A}H$ averages each student with their study group, and the group average is a far more reliable signal than the student alone.

### 15.2 Layer Depth Analysis

| Model | Layers | Receptive Field | Parameters | Train Acc | Test Acc (mean ± std) |
|---|---|---|---|---|---|
| GCN-1 | 1 | 1-hop neighbours | 6 | 0.909 | 0.905 ± 0.050 |
| GCN-2 | 2 | 2-hop neighbours | 40 | 0.955 | 0.925 ± 0.043 |
| GCN-3 | 3 | 3-hop neighbours | 104 | 0.957 | 0.940 ± 0.044 |

**Observations:**

- **GCN-1 (1 layer):** A linear model of the 1-hop aggregated features. Already 0.905: one round of averaging over the study group recovers most of the information the features lack.
- **GCN-2 and GCN-3:** Each extra layer adds a hidden non-linearity and one more hop of context (friends of friends), and accuracy rises to 0.925 and 0.940.
- **Caveat:** The steps between depths (1.5–2 points) are smaller than the standard deviation across splits (about 4.5 points), so the depth trend is suggestive, not statistically established.

**No over-smoothing.** Over-smoothing means repeated aggregation makes node representations converge until classes become indistinguishable. With three layers on a 200-node graph of average degree 7.5 there is no sign of it: GCN-3 has the highest training and test accuracy of the three depths.

### 15.3 Regularization

| Model | Train Acc | Test Acc |
|---|---|---|
| GCN-3 (dropout, weight decay, early stopping) | 0.957 ± 0.019 | 0.940 ± 0.044 |
| GCN-3 (none; fixed 200 epochs) | 0.963 ± 0.017 | 0.943 ± 0.043 |

Regularization makes no measurable difference: with 120 labelled nodes and 104 parameters, the model does not overfit either way (training accuracy is only 1.5–2 points above test accuracy). Early stopping's practical benefit is choosing the training length automatically: the kept epoch ranges from about 120 (GCN-3) to about 440 (GCN-1, whose 6 parameters learn slowly), so no single fixed epoch count would suit every model.

### 15.4 Homophily

The original graph's homophily is 0.807; the random graphs' is 0.498 ± 0.014 — no label signal. On the random graph GCN-3 falls to 0.508 ± 0.107, below the MLP (0.778) and below the majority-class rate of 0.55. Aggregation over random neighbours averages each student with unrelated students, destroying the information in their own features. A GCN is only as good as its graph.

---

## 16. Discussion

### What Worked

1. **The graph adds a large, consistent gain.** GCNs improve on the feature-only MLP by 13–16 points, on every split.
2. **A meaningful graph is essential.** The same GCN on a random graph is worse than ignoring the graph altogether.
3. **Small models suffice.** GCN-1 reaches 90.5% with 6 parameters; GCN-3 reaches 94.0% with 104.
4. **Reproducibility is achieved.** Same seeds produce identical results across runs, in the script and the notebook.

### What Did Not Work as Expected

1. **Regularization did not change results.** With 120 labelled nodes the 3-layer GCN does not overfit, so dropout, weight decay and early stopping have nothing to correct. (In an earlier version with only 14 labelled nodes, GCN-3 did overfit: 100% training accuracy but 88% test accuracy.)
2. **Depth differences are within noise.** The 1 → 2 → 3 layer trend is consistent in direction but not large enough to be conclusive with 10 splits of 40 test nodes.

### Where the Errors Are

In the seed-42 predictions (fig6), the 8 misclassified students sit where the Pass and Fail clusters meet. Their neighbourhoods are mixed, so aggregation gives a weak or misleading signal — exactly where homophily, and hence the GCN's advantage, breaks down.

### Depth and Receptive Field

Each GCN layer extends the receptive field by one hop:
- 1 layer: each node "sees" itself and immediate neighbours
- 2 layers: each node "sees" nodes up to 2 hops away
- 3 layers: each node "sees" nodes up to 3 hops away

Self-loops retain each node's own information at every layer. Deeper networks face two separate risks: **over-smoothing**, where repeated averaging makes representations indistinguishable (lowering training accuracy as well), and **overfitting**, where extra parameters let the model memorize the training labels (training accuracy stays high while test accuracy falls). Comparing train and test accuracy tells them apart; at three layers on this graph neither occurs.

---

## 17. Limitations

1. **Synthetic dataset:** The graph and features are artificially generated. Results cannot be generalized to real-world datasets.
2. **Chosen difficulty:** The homophily (0.81) and the feature overlap (`feature_gap = 0.25`) are parameters. The size of the GCN's advantage over the MLP reflects these choices.
3. **Modest graph size:** 200 nodes is far smaller than real benchmarks (Cora: 2708, PubMed: 19717).
4. **Test-set noise:** 40 test nodes per split means one misclassification changes accuracy by 2.5 points; the differences between GCN depths are within this noise.
5. **Binary classification:** Two classes is the simplest possible classification task.
6. **Limited hyperparameter search:** Learning rate, dropout, weight decay and patience were fixed, not tuned.
7. **Transductive setting:** The model cannot generalize to unseen nodes not present in the original graph.
8. **Dense adjacency:** We use dense matrix operations, which would not scale to large graphs.
9. **Fixed graph:** All seeds share one graph and one set of features; only the split and initialization vary.

---

## 18. Conclusion

This project implemented a three-layer Graph Convolutional Network from scratch, demonstrating the complete mathematical pipeline from graph construction to node classification. Key lessons learned:

1. **GCN mathematics are interpretable:** The operation $\tilde{A} H W$ has a clear interpretation as neighbourhood aggregation followed by linear transformation.
2. **Normalization is essential:** Symmetric normalization $\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$ prevents scale issues and is critical for stable training.
3. **Graphs help when features are ambiguous and neighbours are similar:** With overlapping features and homophily 0.81, aggregation lifted accuracy from 77.8% (MLP) to 94.0% (GCN-3).
4. **A GCN is only as good as its graph:** On a random graph the same model fell to 50.8%, worse than ignoring the graph.
5. **Check train vs test accuracy before blaming depth:** Comparing them distinguishes over-smoothing from overfitting; here three layers showed neither.
6. **Small experiments require caution:** Differences of a couple of points between models are within the noise of 10 splits of 40 test nodes.

---

## 19. Future Work

- **Real-world benchmarks:** Evaluate on Cora, Citeseer, or PubMed to test generalization.
- **Homophily sweep:** Vary $p_{\text{same}}$ and $p_{\text{cross}}$ to find the homophily level at which the GCN stops beating the MLP.
- **Fewer labels:** Reduce the training split to see when overfitting and regularization start to matter.
- **Advanced GNN architectures:** Implement GraphSAGE (inductive learning) or GAT (attention-based aggregation).
- **Sparse operations:** Use PyTorch sparse tensors for scalability.
- **Deeper models:** Test 4+ layers to observe over-smoothing, and mitigations such as DropEdge and PairNorm.
- **Link prediction and graph classification:** Extend the task beyond node classification.

---

## 20. References

1. Kipf, T.N. and Welling, M., 2017. "Semi-Supervised Classification with Graph Convolutional Networks." *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:1609.02907.

2. Hamilton, W.L., Ying, R. and Leskovec, J., 2017. "Inductive Representation Learning on Large Graphs." *Advances in Neural Information Processing Systems (NeurIPS)*.

3. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P. and Bengio, Y., 2018. "Graph Attention Networks." *Proceedings of the International Conference on Learning Representations (ICLR)*.

4. Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T. and Weinberger, K., 2019. "Simplifying Graph Convolutional Networks." *Proceedings of the International Conference on Machine Learning (ICML)*.

5. Li, Q., Han, Z. and Wu, X.M., 2018. "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning." *Proceedings of the AAAI Conference on Artificial Intelligence*.

6. Glorot, X. and Bengio, Y., 2010. "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)*.

7. Kingma, D.P. and Ba, J., 2015. "Adam: A Method for Stochastic Optimization." *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:1412.6980.

---

## Numerical Analysis Notes

### Potential Numerical Issues

1. **Zero-degree nodes:** A node with no edges would have degree 0 after adding self-loops only if the self-loop is missing. With self-loops, every node has degree ≥ 1, avoiding division by zero in $D^{-1/2}$.

2. **Inverse square root:** The implementation safely handles zero degrees by mapping them to 0 rather than producing NaN.

3. **Floating-point precision:** The normalized adjacency $\tilde{A}$ is verified to be symmetric within tolerance $10^{-10}$.

4. **Softmax stability:** PyTorch's `CrossEntropyLoss` uses `LogSumExp` internally, which is numerically stable. Computing softmax separately and then taking the log would risk underflow.

5. **Logits vs probabilities:** The model outputs raw logits $Z \in \mathbb{R}^{N \times C}$. During training, logits are passed directly to `CrossEntropyLoss`. For reporting probabilities: `torch.softmax(logits, dim=1)`. This separation is important because `CrossEntropyLoss` expects logits, and combining `softmax + log` in a single operation (`log_softmax`) avoids the numerical instability of computing `log(exp(x) / sum(exp(x)))` in two steps.
