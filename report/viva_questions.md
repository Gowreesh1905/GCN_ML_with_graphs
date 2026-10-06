# Viva Questions and Answers

This document contains potential questions that might be asked during a university viva regarding the GCN toy project, along with concise, technically accurate answers.

---

## Basic Concepts

**1. What is a graph?**
A graph is a mathematical structure consisting of a set of nodes (vertices) connected by edges. It represents relationships or interactions between entities.

**2. What is a Graph Neural Network (GNN)?**
A GNN is a type of neural network designed to operate directly on graph-structured data. Unlike standard neural networks that process independent samples, GNNs leverage the connections between nodes to learn representations that incorporate both node features and graph topology.

**3. What is a Graph Convolutional Network (GCN)?**
A GCN is a specific type of GNN that updates a node's representation by performing a weighted aggregation of its own features and the features of its immediate neighbours, followed by a linear transformation and a non-linear activation.

**4. What is node classification?**
Node classification is the task of predicting the labels of unlabelled nodes in a graph, given a partially labelled graph. It is a common semi-supervised learning problem on graphs.

**5. What is an adjacency matrix?**
An adjacency matrix $A$ is a square matrix used to represent a finite graph. The elements $A_{ij}$ indicate whether pairs of vertices are adjacent or not in the graph (e.g., $A_{ij} = 1$ if there is an edge between node $i$ and node $j$, otherwise $0$).

**6. What is a degree matrix?**
A degree matrix $D$ is a diagonal matrix where each diagonal entry $D_{ii}$ represents the degree of node $i$ (the number of edges connected to it). All off-diagonal entries are zero.

---

## Mathematics

**7. Why add self-loops?**
In a GCN, a node updates its representation by aggregating features from its neighbours. If self-loops are not added ($\hat{A} = A + I$), the node's own features from the previous layer would be ignored during the aggregation step. Self-loops ensure the node aggregates from its neighbours *and* itself.

**8. Why normalize the adjacency matrix?**
Normalization prevents high-degree nodes from dominating the aggregated features simply because they have more connections. It ensures the scale of the feature representations remains stable across different nodes and across multiple layers of the network.

**9. Why use symmetric normalization $D^{-1/2} A D^{-1/2}$?**
Symmetric normalization scales the edge weight between node $i$ and node $j$ by $1 / \sqrt{degree(i) \cdot degree(j)}$. This provides a balanced aggregation that considers the degrees of both the source and target nodes, reducing the influence of very high-degree neighbours while preserving the symmetry of the adjacency matrix for undirected graphs.

**10. What does $AX$ mean in matrix multiplication?**
Multiplying the adjacency matrix $A$ by the feature matrix $X$ ($AX$) computes the sum of the feature vectors of each node's neighbours. The $i$-th row of $AX$ is $\sum_j A_{ij} X_j$.

**11. What does $AXW$ mean?**
$AXW$ represents a complete graph convolution operation. $AX$ performs the neighbourhood aggregation (mixing features across the graph structure), and multiplying by $W$ performs a linear projection or transformation of those aggregated features into a new feature space.

**12. What does each dimension represent in $\tilde{A} H W$?**
- $\tilde{A}$ is $N \times N$, where $N$ is the number of nodes.
- $H$ is $N \times F_{in}$, where $F_{in}$ is the number of input features.
- $W$ is $F_{in} \times F_{out}$, where $F_{out}$ is the number of output features.
- $\tilde{A}H$ results in $N \times F_{in}$ (aggregation).
- $(\tilde{A}H)W$ results in $N \times F_{out}$ (transformation).

**13. Why use ReLU?**
ReLU (Rectified Linear Unit) introduces non-linearity into the model, allowing it to learn complex, non-linear mappings. Without non-linear activations between layers, multiple GCN layers would collapse mathematically into a single linear transformation.

**14. Why not use ReLU on the output layer?**
The output layer needs to produce raw scores (logits) for each class, which can be negative. Applying ReLU would clip negative logits to zero, destroying the relative scoring information needed by the softmax function to compute accurate class probabilities.

**15. What is a logit?**
A logit is the raw, unnormalized output of a neural network layer before a squashing function like softmax or sigmoid is applied. It maps to the domain $(-\infty, \infty)$.

**16. Why use cross-entropy loss?**
Cross-entropy measures the difference between two probability distributions. In classification, it compares the model's predicted probabilities with the true one-hot encoded labels, heavily penalizing confident but incorrect predictions. PyTorch's implementation computes `log_softmax` internally for numerical stability.

---

## Architecture

**17. Why use three layers?**
A three-layer GCN allows nodes to aggregate information from up to 3 hops away in the graph. This was chosen to study the effects of a slightly deeper network on a small graph compared to 1- or 2-layer models. In our results the extra depth did not help: GCN-3 fit every training node but generalized worse (88.3% vs 98.3% test accuracy), which is a sign of overfitting.

**18. What happens if there is only one layer?**
A 1-layer GCN only aggregates information from a node's immediate (1-hop) neighbours. It has a limited receptive field, meaning it cannot capture wider structural patterns in the graph.

**19. What is the receptive field in a GCN?**
The receptive field of a node in an $L$-layer GCN is the set of all nodes in the graph that influence its final representation. An $L$-layer GCN has an $L$-hop receptive field.

**20. What is over-smoothing?**
Over-smoothing is a phenomenon in deep GNNs where repeated neighbourhood aggregation causes the representations of all nodes in the graph to converge to similar values, making them indistinguishable and degrading classification performance.

**21. Why can too many GCN layers be problematic?**
Beyond over-smoothing, deeper GCNs are harder to train due to vanishing gradients, require more parameters (increasing the risk of overfitting, especially on small datasets), and are computationally more expensive. On small, dense graphs, the receptive field covers the whole graph very quickly, rendering extra depth detrimental.

---

## Implementation

**22. Why implement the GCN manually?**
Implementing it manually (using raw matrix multiplications in PyTorch rather than high-level libraries like PyTorch Geometric) explicitly demonstrates the underlying mathematics. It proves an understanding of how adjacency normalization, message passing, and weight transformations actually work under the hood.

**23. Why use PyTorch?**
PyTorch provides efficient tensor operations, GPU acceleration (if available), and automatic differentiation (autograd). It allows us to define the forward pass mathematically and automatically compute the gradients needed for backpropagation.

**24. Why use Adam optimizer?**
Adam (Adaptive Moment Estimation) dynamically adjusts the learning rate for each parameter based on historical gradients. It generally converges faster and requires less manual tuning of the learning rate compared to standard Stochastic Gradient Descent (SGD).

**25. Why use node-level masks?**
In transductive learning, the entire graph is processed at once. Masks (boolean arrays) are used to selectively calculate the loss and metrics only on the appropriate subsets of nodes (training, validation, or test) while ignoring the labels of the others to prevent data leakage.

**26. Why is the graph transductive?**
It is transductive because the model observes the complete graph structure (including test node connections) during training. It is learning to predict labels for specific, known unlabelled nodes in a single, fixed graph, rather than generalizing to entirely unseen new graphs (which would be inductive).

---

## Complexity

**27. What is the complexity of adjacency multiplication?**
For a dense adjacency matrix, $\tilde{A}H$ requires $O(N^2 F)$ operations. For a sparse adjacency matrix, it requires $O(E F)$ operations, where $E$ is the number of edges.

**28. What is the complexity of a single GCN layer?**
Assuming a dense implementation, one layer takes $O(N^2 F_{in} + N F_{in} F_{out})$. The first term is for aggregating neighbours, and the second is for the feature transformation.

**29. Why are sparse matrices useful for GNNs?**
Real-world graphs are typically very sparse ($E \ll N^2$). Using dense matrices requires $O(N^2)$ memory and compute, which scales poorly. Sparse matrices only store non-zero entries (edges), reducing memory to $O(N+E)$ and compute for aggregation to $O(EF)$, making large graphs tractable.

**30. What is the memory complexity of this toy project?**
Because we use dense matrices for educational clarity, the memory complexity for the adjacency matrix is $O(N^2)$. For the node activations, it is $O(N \sum_l F_l)$. Since $N=24$, this easily fits in standard memory.

---

## Experimental

**31. Why compare with an MLP?**
An MLP serves as a baseline that only uses node features and ignores graph structure completely. Comparing the GCN to the MLP reveals whether the graph connectivity actually provides useful information for the classification task.

**32. Why compare 1, 2, and 3 layers?**
Comparing different depths isolates the effect of the receptive field and model capacity, and lets us look for phenomena like over-smoothing or overfitting. It helps determine the right amount of neighbourhood aggregation for the specific graph topology.

**33. Why use multiple random seeds?**
Neural network initialization, dataset splitting, and graph generation (if stochastic) introduce randomness. On a tiny dataset, a single run might yield anomalously good or bad results by chance. Averaging over multiple seeds provides a more statistically robust estimate of the model's true expected performance. In this project each of the 10 seeds draws both a new train/validation/test split and a new initialization; varying only the initialization would re-test the same 6 test nodes every time and hide how much the result depends on which students land in the test set.

**34. Why is accuracy alone insufficient?**
If classes are imbalanced (e.g., 90% Pass, 10% Fail), a model that always predicts "Pass" gets 90% accuracy but is useless. Precision, recall, F1-score, and the confusion matrix provide a complete picture of how the model performs across both majority and minority classes.

**35. What are the limitations of the experiment?**
The main limitations are the use of a tiny, synthetic dataset (meaning results don't generalize to the real world), the reliance on dense matrix operations (not scalable), a very small test set (high variance in evaluation metrics), no early stopping or regularization, and the simplicity of the feature distributions which made the task trivially solvable by the MLP baseline.

**36. Why did the 3-layer GCN perform worse — over-smoothing or overfitting?**
The evidence points to overfitting. GCN-3 reaches 100% training accuracy with near-zero training loss, but 88.3% test accuracy. Over-smoothing makes node representations so similar that the model cannot separate classes — that would lower *training* accuracy too, which does not happen. GCN-3 has 104 parameters but only 14 labelled training nodes, and trains for 200 epochs with no dropout, weight decay or early stopping, so it can memorize the training labels.

**37. If the MLP gets 100%, what does the random-graph ablation actually show?**
It shows that a GCN depends on the graph being meaningful: on a random graph (homophily ≈ 0.49) GCN-3 drops from 88.3% to 46.7%, because aggregation mixes each student's features with those of unrelated students. It does **not** show that the graph is needed — the MLP solves the task from features alone. To show a graph benefit, the features would need to be made less separable so the MLP falls short of 100%.
