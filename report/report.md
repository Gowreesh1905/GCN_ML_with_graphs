# Node Classification Using a Three-Layer Graph Convolutional Network: Implementation, Mathematical Analysis, and Experimental Evaluation

---

## 1. Abstract

This project investigates how graph structure and neighbourhood aggregation affect node classification using a Graph Convolutional Network (GCN). We implement a three-layer GCN from scratch using PyTorch, without relying on high-level graph neural network libraries, to classify students in a synthetic friendship/study-group graph as Pass or Fail. The project demonstrates the complete mathematical pipeline from adjacency matrix construction through symmetric normalization to multi-layer neighbourhood aggregation. Experimental evaluation across ten random seeds — each drawing a new train/validation/test split and a new weight initialization — compares the 3-layer GCN against an MLP baseline, 1-layer GCN, and 2-layer GCN. The feature-only MLP achieves perfect test accuracy and GCN-1/GCN-2 reach 98.3%, while the 3-layer GCN achieves 88.3% ± 10.7% despite fitting every training node; this train/test gap points to overfitting rather than over-smoothing. A graph-structure ablation study shows that a random graph (no homophily) degrades the 3-layer GCN to 46.7%, i.e. aggregation over a misleading graph hurts. All results are reproducible and honestly reported without overclaiming.

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

For our graph with $N = 24$ nodes:

$$A \in \mathbb{R}^{24 \times 24}, \quad A_{ij} \in \{0, 1\}, \quad A = A^T$$

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
| Number of nodes | 24 |
| Number of edges | 63 |
| Number of classes | 2 (Fail=0, Pass=1) |
| Features per node | 3 |
| Class distribution | Fail: 10, Pass: 14 |
| Graph density | 0.228 |
| Average degree | 5.25 |
| Min/Max degree | 1 / 9 |
| Directed | No |
| Self-loops | No (before normalization) |
| Connected | Yes |

### 8.2 Node Features

Each node (student) has three features:

| Feature | Pass Students | Fail Students |
|---|---|---|
| study_hours | μ=7, σ=1.5 | μ=3, σ=1.5 |
| attendance | μ=80, σ=10 | μ=50, σ=12 |
| assignment_score | μ=75, σ=10 | μ=40, σ=12 |

### 8.3 Graph Generation

Edges are generated stochastically:
- **Same-class probability:** $p_{\text{same}} = 0.35$
- **Cross-class probability:** $p_{\text{cross}} = 0.08$

This creates meaningful but imperfect homophily.

### 8.4 Homophily Analysis

$$H = \frac{|\{(i,j) \in E : y_i = y_j\}|}{|E|} = \frac{50}{63} = 0.794$$

| Metric | Value |
|---|---|
| Total edges | 63 |
| Same-label edges | 50 |
| Cross-label edges | 13 |
| Homophily ratio | 0.794 |

A homophily ratio of 0.794 indicates moderately strong homophily: students who pass tend to be connected to other passing students (e.g., study groups), but some cross-class connections exist.

### 8.5 Feature Preprocessing

Features are standardized using training-set statistics to prevent data leakage:

$$x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

The mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ are computed only from training nodes and applied to all nodes.

### 8.6 Train/Validation/Test Split

| Split | Nodes | Percentage |
|---|---|---|
| Training | 14 | 58% |
| Validation | 4 | 17% |
| Test | 6 | 25% |

**Transductive setting:** The entire graph structure (all edges between all 24 nodes) is visible during training. Only the *labels* of validation and test nodes are hidden. This is standard for node classification on a single graph. The model can leverage the graph connections of unlabelled nodes during message passing.

The table shows the split for seed 42. In the multi-seed experiments (§11), every seed draws a **different** random split of the same graph, so results are averaged over 10 different 6-node test sets rather than one.

**Limitation:** With only 6 test nodes per split, a single misclassification changes accuracy by 16.7 percentage points; test metrics have very high variance and should not be over-interpreted.

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

Zero-degree nodes are handled safely by setting their inverse square root to 0.

---

## 10. Testing

All 51 tests passed (`python -m pytest`).

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

Additional tests verified: binary adjacency values, no initial self-loops, feature shapes, mask non-overlap, mask coverage, approximate split ratios, per-seed splits over a fixed graph, zero-mean/unit-std training features, graph statistics, normalized value ranges, positive diagonal entries, parameter counts, a 2 × 2 confusion matrix even when only one class is present, homophily consistency, and smoke tests for the graph and PCA plots.

---

## 11. Experimental Setup

| Setting | Value |
|---|---|
| Python | 3.14.3 |
| PyTorch | 2.13.0 |
| NumPy | 2.4.3 |
| NetworkX | 3.6.1 |
| scikit-learn | 1.9.0 |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Epochs | 200 (no early stopping) |
| Hidden dimensions | 8 |
| Loss function | Cross-entropy (on logits) |
| Random seeds | 42–51 (10 seeds) |
| What a seed controls | Train/validation/test split **and** weight initialization |
| Fixed across seeds | Graph, features and labels (generated with seed 42) |
| Evaluation metrics | Accuracy, Precision, Recall, F1, Confusion Matrix; train accuracy reported alongside test accuracy |

---

## 12. Results

### 12.1 Multi-Seed Results (10 Seeds)

| Model | Train Acc (mean ± std) | Test Acc (mean ± std) | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Parameters |
|---|---|---|---|---|---|---|
| MLP | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 50 |
| GCN-1 | 0.914 ± 0.043 | 0.983 ± 0.050 | 0.967 ± 0.100 | 1.000 ± 0.000 | 0.950 ± 0.150 | 6 |
| GCN-2 | 0.993 ± 0.021 | 0.983 ± 0.050 | 0.967 ± 0.100 | 0.950 ± 0.150 | 1.000 ± 0.000 | 40 |
| GCN-3 | 1.000 ± 0.000 | 0.883 ± 0.107 | 0.860 ± 0.138 | 0.867 ± 0.176 | 0.897 ± 0.172 | 104 |

### 12.2 Individual Seed Results (Test Accuracy)

Each column is a different train/validation/test split (and initialization). Every test set has 6 nodes, so accuracy moves in steps of 1/6 ≈ 0.167.

| Model | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| GCN-1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 |
| GCN-2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 |
| GCN-3 | 0.667 | 1.000 | 0.833 | 0.833 | 1.000 | 0.833 | 1.000 | 0.833 | 0.833 | 1.000 |

### 12.3 Confusion Matrix (3-Layer GCN, seed=42, test set)

|  | Predicted Fail | Predicted Pass |
|---|---|---|
| **Actual Fail** | 2 | 1 |
| **Actual Pass** | 1 | 2 |

### 12.4 Graph Ablation Study

For each seed, GCN-3 is retrained on an Erdős–Rényi random graph with the same expected number of edges, using the same split as the original-graph run for that seed.

| Configuration | Graph homophily | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---|---|
| GCN-3 (original graph) | 0.794 | 0.883 ± 0.107 | 0.860 ± 0.138 |
| GCN-3 (random graph) | 0.494 ± 0.065 | 0.467 ± 0.163 | 0.535 ± 0.157 |
| MLP (features only) | — | 1.000 ± 0.000 | 1.000 ± 0.000 |

### 12.5 Visualizations

The following figures were generated and saved to the `figures/` directory:

1. **fig1_graph_true_labels.png** — Original graph coloured by true class
2. **fig2_adjacency_matrix.png** — Adjacency matrix heatmap
3. **fig3_normalized_adjacency.png** — Normalized adjacency matrix heatmap
4. **fig4_loss_curve.png** — Training and validation loss curves
5. **fig5_accuracy_curve.png** — Training and validation accuracy curves
6. **fig6_graph_predicted.png** — Graph coloured by predicted class, with misclassified nodes marked by a black X (seed 42: nodes 7 and 10)
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

For our project ($N=24$, $E=63$, $T=200$):
- Layer 1: $O(63 \times 3 + 24 \times 3 \times 8) = O(189 + 576) = O(765)$
- Layer 2: $O(63 \times 8 + 24 \times 8 \times 8) = O(504 + 1536) = O(2040)$
- Layer 3: $O(63 \times 8 + 24 \times 8 \times 2) = O(504 + 384) = O(888)$ (aggregation happens on the 8-dimensional input, before the projection to 2)
- Per epoch: $O(3693)$
- Total: $O(200 \times 3693) \approx O(739{,}000)$

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

For this toy project: $O(24^2) = O(576)$ for dense adjacency. For a real graph with millions of nodes, sparse representations are essential.

### 14.4 Adjacency Construction

Generating edges with the stochastic block model: $O(N^2)$ because every pair is considered.

---

## 15. Ablation Studies

### 15.1 MLP vs GCN

The MLP baseline achieves perfect accuracy (1.000 ± 0.000) on all 10 splits, outperforming the 3-layer GCN (0.883 ± 0.107). This result, while perhaps counterintuitive, is explained by:

1. **Strong features:** The synthetic features (study hours, attendance, assignment scores) have clearly different distributions for Pass vs Fail students, making the classification task solvable from features alone. The MLP therefore sets a ceiling that no model can beat on this dataset, which means this experiment **cannot** show graph structure adding accuracy over features.
2. **Tiny test sets:** With only 6 test nodes per split, small differences between models (one node = 16.7 points) are within noise.
3. **Overfitting in GCN-3:** GCN-3 fits all training nodes perfectly but loses accuracy on test nodes (see §15.2).

The ablation study with a random graph shows that a *misleading* structure hurts: GCN-3 on a random graph (homophily ≈ 0.49, i.e. neighbours are no more likely to share a label than chance) achieves only 0.467 ± 0.163 accuracy, far below the same model on the original homophilic graph (0.883 ± 0.107). Because the MLP reaches 1.000 without any graph, the correct reading is that the real graph is *compatible* with the labels while a random graph actively corrupts the features through aggregation — not that the real graph is necessary.

### 15.2 Layer Depth Analysis

| Model | Layers | Receptive Field | Parameters | Train Acc | Test Acc (mean ± std) |
|---|---|---|---|---|---|
| GCN-1 | 1 | 1-hop neighbours | 6 | 0.914 | 0.983 ± 0.050 |
| GCN-2 | 2 | 2-hop neighbours | 40 | 0.993 | 0.983 ± 0.050 |
| GCN-3 | 3 | 3-hop neighbours | 104 | 1.000 | 0.883 ± 0.107 |

**Observations:**

- **GCN-1 (1 layer):** Each node aggregates from its immediate neighbours. With only 6 parameters and no hidden layer it is a linear model of the aggregated features; it does not even fit all training nodes (0.914), yet reaches 0.983 on test nodes, suggesting local information is already informative.
- **GCN-2 (2 layers):** Fits the training nodes almost perfectly and matches GCN-1 on test nodes. Two hops of aggregation plus one hidden layer is enough capacity for this task.
- **GCN-3 (3 layers):** Fits every training node (train accuracy 1.000, final training loss ≈ 0.001) but test accuracy drops to 0.883 with the largest spread across splits.

**Overfitting, not over-smoothing.** Over-smoothing means repeated aggregation makes node representations converge until classes become indistinguishable. If that were the cause here, GCN-3 would struggle to separate the *training* nodes too — but it separates them perfectly. A model that is perfect on training nodes and worse on held-out nodes is overfitting: GCN-3 has 104 parameters fitted to 14 labelled nodes, with no dropout, weight decay, or early stopping, and trains for 200 epochs regardless of validation performance. Three hops on this graph (diameter 4) also let each node draw on most of the graph, which gives the model more ways to memorize the training labels.

**Caveat:** Differences between GCN-1, GCN-2 and GCN-3 amount to a handful of test nodes across 10 splits. They indicate a trend, not a statistically established effect.

### 15.3 Homophily

The graph homophily ratio is 0.794, compared with 0.494 ± 0.065 for the random graphs. GCN-3 on the original graph (0.883) substantially outperforms GCN-3 on a random graph (0.467), so the homophilic structure is compatible with the labels while a random structure is harmful. However, the features alone are already so discriminative that the MLP doesn't need graph structure at all.

---

## 16. Discussion

### What Worked

1. **All GCN variants learn the task.** Even GCN-1 (6 parameters) achieves 98.3% test accuracy.
2. **A meaningful graph beats a random one.** The ablation study shows GCN-3 depends heavily on the graph being homophilic.
3. **GCN-1 and GCN-2 achieve the best GCN performance** (0.983 each). One or two layers are enough for this graph size and structure.
4. **Reproducibility is achieved.** Same seeds produce identical results across runs.

### What Did Not Work as Expected

1. **GCN-3 performs worse than simpler models.** It reaches 100% training accuracy but 88.3% test accuracy: it overfits the 14 training nodes (see §15.2).
2. **MLP matches or beats all GCN variants.** The synthetic features are highly discriminative, making graph structure unnecessary for this particular dataset. This is an honest finding — graph structure is not always needed — and it means this dataset cannot demonstrate a GCN *benefit* over features alone.

### Why Graph Structure Matters

The ablation study demonstrates that GCN performance depends on the graph being *meaningful*. On the original graph (homophily 0.794), GCN-3 achieves 88.3% accuracy. On a random graph with similar density (homophily 0.494), only 46.7% — below the 58% a majority-class guess would get on a 14/10 class split. Aggregation over a random graph averages each student's features with those of unrelated students, corrupting the signal the features carry.

### Depth and Receptive Field

Each GCN layer extends the receptive field by one hop:
- 1 layer: each node "sees" itself and immediate neighbours
- 2 layers: each node "sees" nodes up to 2 hops away
- 3 layers: each node "sees" nodes up to 3 hops away

However, information from previous layers is retained through self-loops, so earlier representations are not completely lost. Deeper networks face two separate risks: **over-smoothing**, where information from many different nodes gets averaged together until representations become indistinguishable (this lowers training accuracy as well), and **overfitting**, where the extra parameters and receptive field let the model memorize the training labels (training accuracy stays perfect while test accuracy falls). In this project the evidence points to the second.

---

## 17. Limitations

1. **Synthetic dataset:** The graph and features are artificially generated. Results cannot be generalized to real-world datasets.
2. **Tiny graph:** 24 nodes is far smaller than real graph datasets (Cora: 2708, PubMed: 19717). Statistical significance is limited.
3. **Small test set:** Only 6 test nodes per split means a single misclassification changes accuracy by ~17 percentage points. Averaging over 10 different splits reduces, but does not remove, this noise.
4. **Binary classification:** Two classes is the simplest possible classification task.
5. **Synthetic graph construction:** The homophily level and feature distributions are controlled parameters. Different choices would yield different results.
6. **Limited hyperparameter search:** We used a fixed learning rate (0.01) and 200 epochs without extensive tuning.
7. **Transductive setting:** The model cannot generalize to unseen nodes not present in the original graph.
8. **Dense adjacency:** We use dense matrix operations, which would not scale to large graphs.
9. **No regularization or early stopping:** No dropout or weight decay is used, and validation accuracy is logged but not used to stop training or select a checkpoint. GCN-3's train/test gap shows this matters even for a 104-parameter model when only 14 nodes are labelled.
10. **Feature separability:** The features alone are highly discriminative (the MLP scores 100%), so this dataset cannot demonstrate a benefit of graph structure over features alone.
11. **Fixed graph:** All seeds share one graph and one set of features; only the split and initialization vary.

---

## 18. Conclusion

This project implemented a three-layer Graph Convolutional Network from scratch, demonstrating the complete mathematical pipeline from graph construction to node classification. Key lessons learned:

1. **GCN mathematics are interpretable:** The operation $\tilde{A} H W$ has a clear interpretation as neighbourhood aggregation followed by linear transformation.
2. **Normalization is essential:** Symmetric normalization $\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$ prevents scale issues and is critical for stable training.
3. **Depth is a trade-off:** More layers expand the receptive field and add parameters. Here the 3-layer GCN overfit the 14 labelled nodes; on larger or deeper settings, over-smoothing is a further risk. Comparing train and test accuracy is how to tell the two apart.
4. **Graph structure can help or hurt:** A GCN depends on meaningful graph structure (homophily) — a random graph cut GCN-3's accuracy from 88% to 47% — but when features are already highly discriminative, even a good graph may not add value over an MLP.
5. **Small experiments require caution:** Results on tiny datasets have high variance and should not be used to make general claims about model architectures.

---

## 19. Future Work

- **Real-world benchmarks:** Evaluate on Cora, Citeseer, or PubMed to test generalization.
- **Advanced GNN architectures:** Implement GraphSAGE (inductive learning) or GAT (attention-based aggregation).
- **Sparse operations:** Use PyTorch sparse tensors for scalability.
- **Link prediction:** Extend the task beyond node classification.
- **Graph classification:** Apply graph-level pooling for whole-graph prediction.
- **Inductive learning:** Design models that can generalize to unseen graphs.
- **Heterogeneous graphs:** Handle multiple node and edge types.
- **Regularization and early stopping:** Use the validation split for early stopping/checkpoint selection, and add dropout or weight decay to close GCN-3's train/test gap. Investigate DropEdge and PairNorm for over-smoothing in deeper models.
- **Harder features:** Increase class overlap in the synthetic features so the MLP no longer reaches 100%, making it possible to measure whether the graph adds accuracy.

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
