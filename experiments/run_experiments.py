"""
run_experiments.py — Main Experiment Runner

Runs the full experimental pipeline:
1. Generate synthetic graph and dataset
2. Train 3-layer GCN (main model)
3. Train MLP baseline
4. Train 1-layer and 2-layer GCN for depth comparison, and a 3-layer GCN
   without regularization (no dropout, weight decay or early stopping)
5. Graph ablation (random graph) and a graph-only baseline (neighbour
   majority vote over training labels)
5b. Homophily sweep: graphs with the same expected number of edges but
   homophily from 0.9 down to 0.5
6. Multi-seed evaluation (10 seeds; each seed draws a new train/val/test
   split *and* a new weight initialization — the graph itself is fixed)
7. Generate all figures
8. Save results to CSV and JSON

Usage:
    python experiments/run_experiments.py
"""

import json
import os
import sys

import numpy as np
import torch

# Windows consoles default to a legacy code page that cannot print the
# mathematical symbols (Â, D̂, ±) used in this script's output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.graph_data import (
    build_dataset,
    generate_random_graph,
    get_adjacency_matrix,
    set_all_seeds,
)
from src.normalization import (
    normalize_adjacency_torch,
    symmetric_normalization,
)
from src.gcn import GCN, MLP
from src.train import train_model
from src.evaluate import evaluate_model, compute_homophily
from src.baselines import evaluate_neighbour_vote
from src.visualization import (
    plot_graph,
    plot_adjacency_heatmap,
    plot_training_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pca_embeddings,
    plot_homophily_sweep,
)


# ============================================================================
# Configuration
# ============================================================================
SEEDS = list(range(42, 52))  # 10 seeds
GRAPH_SEED = 42  # features, labels and graph are fixed across all seeds
LR = 0.01
EPOCHS = 500          # upper limit; early stopping usually ends training sooner
PATIENCE = 30         # stop after 30 epochs without a lower validation loss
DROPOUT = 0.5
WEIGHT_DECAY = 5e-4
UNREG_EPOCHS = 200    # fixed training length for the unregularized GCN-3
N_PASS = 110
N_FAIL = 90
P_SAME = 0.06
P_CROSS = 0.015
FEATURE_GAP = 0.25    # class means pulled together: features overlap strongly
HOMOPHILY_LEVELS = [0.9, 0.8, 0.7, 0.6, 0.5]   # target homophily for the sweep
N_HIDDEN = 8
N_FEATURES = 3
N_CLASSES = 2

# Training settings used for every model unless stated otherwise
TRAIN_KWARGS = {"lr": LR, "epochs": EPOCHS, "patience": PATIENCE,
                "weight_decay": WEIGHT_DECAY}
UNREG_TRAIN_KWARGS = {"lr": LR, "epochs": UNREG_EPOCHS}

FIGURES_DIR = os.path.join(project_root, "figures")
RESULTS_DIR = os.path.join(project_root, "experiments", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def build_seed_dataset(seed: int) -> dict:
    """Fixed graph (GRAPH_SEED) with a train/val/test split drawn from `seed`."""
    return build_dataset(
        n_pass=N_PASS, n_fail=N_FAIL,
        p_same=P_SAME, p_cross=P_CROSS,
        feature_gap=FEATURE_GAP,
        seed=GRAPH_SEED, split_seed=seed,
    )


def edge_probabilities(target_homophily: float) -> tuple:
    """
    (p_same, p_cross) giving the requested expected homophily while keeping
    the expected number of edges equal to the main graph's.
    """
    same_pairs = N_PASS * (N_PASS - 1) / 2 + N_FAIL * (N_FAIL - 1) / 2
    cross_pairs = N_PASS * N_FAIL
    expected_edges = P_SAME * same_pairs + P_CROSS * cross_pairs
    return (target_homophily * expected_edges / same_pairs,
            (1 - target_homophily) * expected_edges / cross_pairs)


def run_single_experiment(
    model_name: str,
    model_class,
    model_kwargs: dict,
    data: dict,
    A_norm: torch.Tensor,
    seed: int,
    train_kwargs: dict = TRAIN_KWARGS,
) -> dict:
    """
    Train and evaluate a single model with a given seed.

    Returns dict with training history and test metrics.
    """
    set_all_seeds(seed)

    model = model_class(**model_kwargs)
    history = train_model(
        model, A_norm, data["t_features"], data["t_labels"],
        data["t_train_mask"], data["t_val_mask"],
        seed=seed, verbose=False, **train_kwargs,
    )

    test_metrics = evaluate_model(
        model, A_norm, data["t_features"],
        data["t_labels"], data["t_test_mask"],
    )
    train_metrics = evaluate_model(
        model, A_norm, data["t_features"],
        data["t_labels"], data["t_train_mask"],
    )

    return {
        "model_name": model_name,
        "seed": seed,
        "history": history,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "model": model,
        "n_params": model.count_parameters(),
    }


def main():
    print_section("GCN TOY PROJECT — FULL EXPERIMENTAL PIPELINE")

    # ==================================================================
    # 1. Generate Dataset
    # ==================================================================
    print_section("1. Dataset Generation")
    data = build_seed_dataset(GRAPH_SEED)

    # Print statistics
    stats = data["stats"]
    print("Graph Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    print(f"\nFeature matrix X: {data['features'].shape}")
    print(f"Labels: {data['labels']}")
    print(f"Train nodes: {data['train_mask'].sum()}")
    print(f"Val nodes:   {data['val_mask'].sum()}")
    print(f"Test nodes:  {data['test_mask'].sum()}")

    # ==================================================================
    # 2. Normalization
    # ==================================================================
    print_section("2. Adjacency Normalization")
    A = data["adjacency"]
    A_norm_np, A_hat, D_hat, D_inv_sqrt = symmetric_normalization(A)
    A_norm = torch.tensor(A_norm_np, dtype=torch.float32)

    print(f"A (original):      shape={A.shape}, symmetric={np.allclose(A, A.T)}")
    print(f"Â = A + I:          shape={A_hat.shape}, diag all 1={np.all(np.diag(A_hat)==1)}")
    print(f"D̂ (degree matrix): shape={D_hat.shape}")
    print(f"Ã (normalized):    shape={A_norm_np.shape}, symmetric={np.allclose(A_norm_np, A_norm_np.T)}")

    # Homophily
    homophily = compute_homophily(A, data["labels"])
    print(f"\nHomophily Analysis:")
    for key, val in homophily.items():
        print(f"  {key}: {val}")

    # ==================================================================
    # 3. Manual Numerical Example (4-node subgraph)
    # ==================================================================
    print_section("3. Manual Numerical Example (4-Node Subgraph)")
    manual_numerical_example()

    # ==================================================================
    # 4. Figures: Graph and Adjacency
    # ==================================================================
    print_section("4. Generating Graph Visualizations")

    plot_graph(
        data["graph"], data["labels"],
        title="Student Graph — True Labels",
        save_path=os.path.join(FIGURES_DIR, "fig1_graph_true_labels.png"),
    )

    plot_adjacency_heatmap(
        A,
        title="Original Adjacency Matrix A",
        save_path=os.path.join(FIGURES_DIR, "fig2_adjacency_matrix.png"),
    )

    plot_adjacency_heatmap(
        A_norm_np,
        title="Normalized Adjacency Matrix Ã",
        save_path=os.path.join(FIGURES_DIR, "fig3_normalized_adjacency.png"),
        cmap="YlOrRd",
    )

    # ==================================================================
    # 5. Train Main 3-Layer GCN (Single Seed for Figures)
    # ==================================================================
    print_section("5. Training Main 3-Layer GCN (seed=42)")

    main_result = run_single_experiment(
        "GCN-3", GCN,
        {"n_features": N_FEATURES, "n_hidden": N_HIDDEN,
         "n_classes": N_CLASSES, "n_layers": 3, "dropout": DROPOUT},
        data, A_norm, seed=42,
    )

    print(f"\nTest Metrics (seed=42):")
    for k, v in main_result["test_metrics"].items():
        if k not in ("predictions", "probabilities", "y_true"):
            print(f"  {k}: {v}")
    print(f"  train accuracy: {main_result['train_metrics']['accuracy']}")
    print(f"  stopped after {len(main_result['history']['train_loss'])} epochs; "
          f"best validation loss at epoch {main_result['history']['best_epoch']}")

    # Training curves
    plot_training_curves(
        main_result["history"],
        title_prefix="3-Layer GCN — ",
        save_path_loss=os.path.join(FIGURES_DIR, "fig4_loss_curve.png"),
        save_path_acc=os.path.join(FIGURES_DIR, "fig5_accuracy_curve.png"),
    )

    # Predicted graph
    all_preds_result = evaluate_model(
        main_result["model"], A_norm,
        data["t_features"], data["t_labels"],
        torch.ones(len(data["labels"]), dtype=torch.bool),  # All nodes
    )
    all_preds = np.array(all_preds_result["predictions"])

    plot_graph(
        data["graph"], data["labels"],
        title="Student Graph — Predicted Labels (3-Layer GCN)",
        save_path=os.path.join(FIGURES_DIR, "fig6_graph_predicted.png"),
        predictions=all_preds,
    )

    # Confusion matrix
    cm = np.array(main_result["test_metrics"]["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        title="Confusion Matrix — 3-Layer GCN (Test Set)",
        save_path=os.path.join(FIGURES_DIR, "fig7_confusion_matrix.png"),
    )

    # PCA embeddings
    model_for_embed = main_result["model"]
    model_for_embed.eval()
    with torch.no_grad():
        if len(model_for_embed.layers) >= 2:
            emb1 = model_for_embed.get_embeddings(A_norm, data["t_features"], layer_idx=0)
            plot_pca_embeddings(
                emb1.numpy(), data["labels"],
                title="PCA of Layer 1 Embeddings (H¹)",
                save_path=os.path.join(FIGURES_DIR, "fig9_pca_layer1.png"),
            )
        if len(model_for_embed.layers) >= 3:
            emb2 = model_for_embed.get_embeddings(A_norm, data["t_features"], layer_idx=1)
            plot_pca_embeddings(
                emb2.numpy(), data["labels"],
                title="PCA of Layer 2 Embeddings (H²)",
                save_path=os.path.join(FIGURES_DIR, "fig10_pca_layer2.png"),
            )

    # ==================================================================
    # 6. Multi-Seed Experiments
    # ==================================================================
    print_section(f"6. Multi-Seed Experiments ({len(SEEDS)} seeds)")
    print("Each seed draws a new train/val/test split and a new weight")
    print("initialization. The graph and features are the same for every seed.\n")

    seed_data = {seed: build_seed_dataset(seed) for seed in SEEDS}

    dims = {"n_features": N_FEATURES, "n_hidden": N_HIDDEN, "n_classes": N_CLASSES}
    # name: (model class, model kwargs, training kwargs)
    model_configs = {
        "MLP": (MLP, {**dims, "dropout": DROPOUT}, TRAIN_KWARGS),
        "GCN-1": (GCN, {**dims, "n_layers": 1, "dropout": DROPOUT}, TRAIN_KWARGS),
        "GCN-2": (GCN, {**dims, "n_layers": 2, "dropout": DROPOUT}, TRAIN_KWARGS),
        "GCN-3": (GCN, {**dims, "n_layers": 3, "dropout": DROPOUT}, TRAIN_KWARGS),
        # Same architecture, but no dropout, no weight decay, no early stopping
        "GCN-3 (no reg)": (GCN, {**dims, "n_layers": 3}, UNREG_TRAIN_KWARGS),
    }

    all_results = {}  # {model_name: {seed: result}}
    for model_name, (model_class, model_kwargs, train_kwargs) in model_configs.items():
        all_results[model_name] = {}
        for seed in SEEDS:
            result = run_single_experiment(
                model_name, model_class, model_kwargs,
                seed_data[seed], A_norm, seed, train_kwargs,
            )
            all_results[model_name][seed] = result

    # Aggregate results
    summary = {}
    for model_name in model_configs:
        accs = [all_results[model_name][s]["test_metrics"]["accuracy"] for s in SEEDS]
        f1s = [all_results[model_name][s]["test_metrics"]["f1"] for s in SEEDS]
        precs = [all_results[model_name][s]["test_metrics"]["precision"] for s in SEEDS]
        recs = [all_results[model_name][s]["test_metrics"]["recall"] for s in SEEDS]
        losses = [all_results[model_name][s]["history"]["train_loss"][-1] for s in SEEDS]
        train_accs = [all_results[model_name][s]["train_metrics"]["accuracy"] for s in SEEDS]
        best_epochs = [all_results[model_name][s]["history"]["best_epoch"] for s in SEEDS]

        summary[model_name] = {
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "train_accuracy_mean": np.mean(train_accs),
            "train_accuracy_std": np.std(train_accs),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "precision_mean": np.mean(precs),
            "precision_std": np.std(precs),
            "recall_mean": np.mean(recs),
            "recall_std": np.std(recs),
            "final_train_loss_mean": np.mean(losses),
            "individual_accuracies": accs,
            "individual_f1s": f1s,
            "n_params": all_results[model_name][SEEDS[0]]["n_params"],
            "best_epoch_mean": np.mean(best_epochs),
        }

    # Print summary table
    print(f"\n{'Model':<16} {'Train Acc (mean±std)':<25} {'Test Acc (mean±std)':<25} {'F1 (mean±std)':<25} {'Precision (mean±std)':<25} {'Recall (mean±std)':<25}")
    print("-" * 135)
    for model_name in model_configs:
        s = summary[model_name]
        print(
            f"{model_name:<16} "
            f"{s['train_accuracy_mean']:.4f} ± {s['train_accuracy_std']:.4f}       "
            f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}       "
            f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}       "
            f"{s['precision_mean']:.4f} ± {s['precision_std']:.4f}       "
            f"{s['recall_mean']:.4f} ± {s['recall_std']:.4f}"
        )

    # Print individual seed results
    print(f"\nIndividual Seed Results (Accuracy):")
    print(f"{'Model':<16}", end="")
    for s in SEEDS:
        print(f"  Seed {s}  ", end="")
    print()
    for model_name in model_configs:
        print(f"{model_name:<16}", end="")
        for s in SEEDS:
            acc = all_results[model_name][s]["test_metrics"]["accuracy"]
            print(f"  {acc:.4f}  ", end="")
        print()

    # Parameter counts
    print(f"\nParameter Counts:")
    for model_name in model_configs:
        print(f"  {model_name}: {summary[model_name]['n_params']} parameters, "
              f"best epoch (mean) {summary[model_name]['best_epoch_mean']:.0f}")

    # Graph-only baseline: majority vote over each node's labelled neighbours
    vote_results = {
        seed: evaluate_neighbour_vote(sd["adjacency"], sd["labels"],
                                      sd["train_mask"], sd["test_mask"])
        for seed, sd in seed_data.items()
    }
    vote_accs = [vote_results[s]["accuracy"] for s in SEEDS]
    vote_f1s = [vote_results[s]["f1"] for s in SEEDS]
    print(f"\nNeighbour majority vote (graph + training labels only, no features):")
    print(f"  Accuracy: {np.mean(vote_accs):.4f} ± {np.std(vote_accs):.4f}")
    print(f"  F1:       {np.mean(vote_f1s):.4f} ± {np.std(vote_f1s):.4f}")

    # ==================================================================
    # 7. Graph Ablation Study
    # ==================================================================
    print_section("7. Graph Ablation Study")

    # Experiment C: Random graph with same approximate number of edges
    num_edges = data["stats"]["num_edges"]
    N = data["stats"]["num_nodes"]

    ablation_results = {}
    random_homophilies = []
    for seed in SEEDS:
        sd = seed_data[seed]  # same split as the GCN-3 run for this seed

        # Generate random graph
        random_G = generate_random_graph(N, num_edges, seed=seed + 100)
        random_A = get_adjacency_matrix(random_G)
        random_A_norm = normalize_adjacency_torch(random_A)

        # Train GCN-3 on random graph
        set_all_seeds(seed)
        model = GCN(n_features=N_FEATURES, n_hidden=N_HIDDEN,
                     n_classes=N_CLASSES, n_layers=3, dropout=DROPOUT)
        history = train_model(
            model, random_A_norm, sd["t_features"], sd["t_labels"],
            sd["t_train_mask"], sd["t_val_mask"],
            seed=seed, verbose=False, **TRAIN_KWARGS,
        )
        test_metrics = evaluate_model(
            model, random_A_norm, sd["t_features"],
            sd["t_labels"], sd["t_test_mask"],
        )
        ablation_results[seed] = test_metrics

        # Homophily of the random graph (expected near 0.5 — no label signal)
        random_homophilies.append(
            compute_homophily(random_A, sd["labels"])["homophily_ratio"]
        )

    ablation_accs = [ablation_results[s]["accuracy"] for s in SEEDS]
    ablation_f1s = [ablation_results[s]["f1"] for s in SEEDS]

    print(f"Random graph homophily: {np.mean(random_homophilies):.4f} ± {np.std(random_homophilies):.4f}"
          f"  (original graph: {homophily['homophily_ratio']:.4f})\n")
    print(f"GCN-3 on Random Graph:")
    print(f"  Accuracy: {np.mean(ablation_accs):.4f} ± {np.std(ablation_accs):.4f}")
    print(f"  F1:       {np.mean(ablation_f1s):.4f} ± {np.std(ablation_f1s):.4f}")
    print(f"\nCompare with GCN-3 on Original Graph:")
    print(f"  Accuracy: {summary['GCN-3']['accuracy_mean']:.4f} ± {summary['GCN-3']['accuracy_std']:.4f}")
    print(f"  F1:       {summary['GCN-3']['f1_mean']:.4f} ± {summary['GCN-3']['f1_std']:.4f}")

    # ==================================================================
    # 7b. Homophily Sweep
    # ==================================================================
    print_section("7b. Homophily Sweep")
    print("Same students, features, splits and expected number of edges;")
    print("only the share of edges joining same-result students changes.\n")

    sweep_models = {
        "GCN-1": {**dims, "n_layers": 1, "dropout": DROPOUT},
        "GCN-3": {**dims, "n_layers": 3, "dropout": DROPOUT},
    }
    sweep = {"homophily": [], "MLP": [], "Neighbour vote": [],
             "GCN-1": [], "GCN-3": []}
    for target in HOMOPHILY_LEVELS:
        p_same, p_cross = edge_probabilities(target)
        level = {k: [] for k in ("Neighbour vote", "GCN-1", "GCN-3")}
        realised = []
        for seed in SEEDS:
            sd = build_dataset(
                n_pass=N_PASS, n_fail=N_FAIL, p_same=p_same, p_cross=p_cross,
                feature_gap=FEATURE_GAP, seed=GRAPH_SEED, split_seed=seed,
            )
            realised.append(sd["stats"]["homophily_ratio"])
            sd_A_norm = normalize_adjacency_torch(sd["adjacency"])
            level["Neighbour vote"].append(evaluate_neighbour_vote(
                sd["adjacency"], sd["labels"], sd["train_mask"], sd["test_mask"]
            )["accuracy"])
            for name, kwargs in sweep_models.items():
                r = run_single_experiment(name, GCN, kwargs, sd, sd_A_norm, seed)
                level[name].append(r["test_metrics"]["accuracy"])

        sweep["homophily"].append(float(np.mean(realised)))
        # The MLP ignores the graph, so its results are the main experiment's
        sweep["MLP"].append(summary["MLP"]["individual_accuracies"])
        for k, v in level.items():
            sweep[k].append(v)
        print(f"homophily {np.mean(realised):.3f} (p_same={p_same:.4f}, p_cross={p_cross:.4f}): "
              + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in level.items()))

    plot_homophily_sweep(
        sweep,
        majority_rate=N_PASS / (N_PASS + N_FAIL),
        save_path=os.path.join(FIGURES_DIR, "fig11_homophily_sweep.png"),
    )

    # ==================================================================
    # 8. Model Comparison Figure
    # ==================================================================
    print_section("8. Model Comparison Figure")

    # Accuracy comparison
    acc_comparison = {}
    for model_name in model_configs:
        acc_comparison[model_name.replace(" (", "\n(")] = {
            "mean": summary[model_name]["accuracy_mean"],
            "std": summary[model_name]["accuracy_std"],
        }
    acc_comparison["GCN-3\n(Random)"] = {
        "mean": np.mean(ablation_accs),
        "std": np.std(ablation_accs),
    }
    acc_comparison["Neighbour\nvote"] = {
        "mean": np.mean(vote_accs),
        "std": np.std(vote_accs),
    }

    plot_model_comparison(
        acc_comparison,
        metric="accuracy",
        title=f"Model Comparison — Test Accuracy ({len(SEEDS)} Seeds)",
        save_path=os.path.join(FIGURES_DIR, "fig8_model_comparison_accuracy.png"),
    )

    # F1 comparison
    f1_comparison = {}
    for model_name in model_configs:
        f1_comparison[model_name.replace(" (", "\n(")] = {
            "mean": summary[model_name]["f1_mean"],
            "std": summary[model_name]["f1_std"],
        }
    f1_comparison["GCN-3\n(Random)"] = {
        "mean": np.mean(ablation_f1s),
        "std": np.std(ablation_f1s),
    }
    f1_comparison["Neighbour\nvote"] = {
        "mean": np.mean(vote_f1s),
        "std": np.std(vote_f1s),
    }

    plot_model_comparison(
        f1_comparison,
        metric="f1 score",
        title=f"Model Comparison — Test F1 Score ({len(SEEDS)} Seeds)",
        save_path=os.path.join(FIGURES_DIR, "fig8b_model_comparison_f1.png"),
    )

    # ==================================================================
    # 9. Save Results
    # ==================================================================
    print_section("9. Saving Results")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, "w") as f:
        f.write("model,seed,accuracy,f1,precision,recall,train_accuracy,final_train_loss\n")
        for model_name in model_configs:
            for seed in SEEDS:
                r = all_results[model_name][seed]
                tm = r["test_metrics"]
                train_acc = r["train_metrics"]["accuracy"]
                final_loss = r["history"]["train_loss"][-1]
                f.write(
                    f"{model_name},{seed},{tm['accuracy']:.4f},"
                    f"{tm['f1']:.4f},{tm['precision']:.4f},"
                    f"{tm['recall']:.4f},{train_acc:.4f},{final_loss:.4f}\n"
                )
        # Ablation results
        for seed in SEEDS:
            tm = ablation_results[seed]
            f.write(
                f"GCN-3-Random,{seed},{tm['accuracy']:.4f},"
                f"{tm['f1']:.4f},{tm['precision']:.4f},"
                f"{tm['recall']:.4f},,\n"
            )
        # Graph-only baseline
        for seed in SEEDS:
            vr = vote_results[seed]
            f.write(f"Neighbour vote,{seed},{vr['accuracy']:.4f},{vr['f1']:.4f},,,,\n")
    print(f"Saved: {csv_path}")

    sweep_path = os.path.join(RESULTS_DIR, "homophily_sweep.csv")
    with open(sweep_path, "w") as f:
        f.write("target_homophily,homophily,method,accuracy_mean,accuracy_std\n")
        for i, target in enumerate(HOMOPHILY_LEVELS):
            for method in ("MLP", "Neighbour vote", "GCN-1", "GCN-3"):
                accs = sweep[method][i]
                f.write(f"{target},{sweep['homophily'][i]:.4f},{method},"
                        f"{np.mean(accs):.4f},{np.std(accs):.4f}\n")
    print(f"Saved: {sweep_path}")

    # Save JSON
    json_results = {
        "config": {
            "seeds": SEEDS,
            "graph_seed": GRAPH_SEED,
            "seed_controls": "train/val/test split and weight initialization",
            "max_epochs": EPOCHS,
            "patience": PATIENCE,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "unregularized_epochs": UNREG_EPOCHS,
            "feature_gap": FEATURE_GAP,
            "lr": LR,
            "n_hidden": N_HIDDEN,
            "n_pass": N_PASS,
            "n_fail": N_FAIL,
            "p_same": P_SAME,
            "p_cross": P_CROSS,
        },
        "graph_stats": stats,
        "homophily": homophily,
        "summary": {},
        "ablation": {
            "random_graph_accuracy_mean": float(np.mean(ablation_accs)),
            "random_graph_accuracy_std": float(np.std(ablation_accs)),
            "random_graph_f1_mean": float(np.mean(ablation_f1s)),
            "random_graph_f1_std": float(np.std(ablation_f1s)),
            "random_graph_homophily_mean": float(np.mean(random_homophilies)),
            "random_graph_homophily_std": float(np.std(random_homophilies)),
        },
        "neighbour_vote": {
            "accuracy_mean": float(np.mean(vote_accs)),
            "accuracy_std": float(np.std(vote_accs)),
            "f1_mean": float(np.mean(vote_f1s)),
            "f1_std": float(np.std(vote_f1s)),
            "individual_accuracies": [float(a) for a in vote_accs],
        },
        "homophily_sweep": {
            "target_homophily": HOMOPHILY_LEVELS,
            "realised_homophily": sweep["homophily"],
            **{m: {"accuracy_mean": [float(np.mean(a)) for a in sweep[m]],
                   "accuracy_std": [float(np.std(a)) for a in sweep[m]]}
               for m in ("MLP", "Neighbour vote", "GCN-1", "GCN-3")},
        },
    }
    for model_name in model_configs:
        s = summary[model_name]
        json_results["summary"][model_name] = {
            "accuracy_mean": float(s["accuracy_mean"]),
            "accuracy_std": float(s["accuracy_std"]),
            "train_accuracy_mean": float(s["train_accuracy_mean"]),
            "train_accuracy_std": float(s["train_accuracy_std"]),
            "f1_mean": float(s["f1_mean"]),
            "f1_std": float(s["f1_std"]),
            "precision_mean": float(s["precision_mean"]),
            "precision_std": float(s["precision_std"]),
            "recall_mean": float(s["recall_mean"]),
            "recall_std": float(s["recall_std"]),
            "individual_accuracies": [float(a) for a in s["individual_accuracies"]],
            "individual_f1s": [float(f) for f in s["individual_f1s"]],
            "n_params": int(s["n_params"]),
            "best_epoch_mean": float(s["best_epoch_mean"]),
        }

    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {json_path}")

    # ==================================================================
    # 10. Final Summary
    # ==================================================================
    print_section("FINAL SUMMARY")
    print(f"Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    print(f"Homophily: {homophily['homophily_ratio']:.4f}")
    print(f"Classes: {stats['class_distribution']}")
    print(f"\nAll experiments completed successfully.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")


def manual_numerical_example():
    """
    Manual numerical walkthrough with a 4-node graph.

    Demonstrates every step of the GCN computation with actual numbers.
    """
    print("--- 4-Node Manual Example ---\n")

    # Step 1: Define a small graph
    # Node 0 -- Node 1
    # Node 1 -- Node 2
    # Node 2 -- Node 3
    # Node 0 -- Node 3
    #
    # Graph: 0-1-2-3-0 (a cycle)
    A = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=np.float64)

    print("Step 1: Adjacency Matrix A (4 × 4)")
    print(A)
    print(f"  Shape: {A.shape}")
    print(f"  Symmetric: {np.allclose(A, A.T)}")

    # Step 2: Identity matrix
    I = np.eye(4)
    print(f"\nStep 2: Identity Matrix I (4 × 4)")
    print(I)

    # Step 3: Add self-loops
    A_hat = A + I
    print(f"\nStep 3: Â = A + I (4 × 4)")
    print(A_hat)

    # Step 4: Degree matrix
    D_hat = np.diag(A_hat.sum(axis=1))
    print(f"\nStep 4: Degree Matrix D̂ (4 × 4)")
    print(D_hat)
    print(f"  Degrees: {np.diag(D_hat)}")

    # Step 5: D^(-1/2)
    diag_vals = np.diag(D_hat)
    inv_sqrt_vals = 1.0 / np.sqrt(diag_vals)
    D_inv_sqrt = np.diag(inv_sqrt_vals)
    print(f"\nStep 5: D̂^(-1/2) (4 × 4)")
    print(np.round(D_inv_sqrt, 4))

    # Step 6: Normalized adjacency
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    print(f"\nStep 6: Ã = D̂^(-1/2) Â D̂^(-1/2) (4 × 4)")
    print(np.round(A_norm, 4))
    print(f"  Symmetric: {np.allclose(A_norm, A_norm.T)}")

    # Step 7: Node features (3 features per node)
    X = np.array([
        [5.0, 70.0, 60.0],   # Student 0
        [8.0, 90.0, 85.0],   # Student 1
        [2.0, 40.0, 30.0],   # Student 2
        [6.0, 75.0, 65.0],   # Student 3
    ])
    print(f"\nStep 7: Feature Matrix X (4 × 3)")
    print(X)
    print(f"  Shape: {X.shape}")
    print(f"  Columns: [study_hours, attendance, assignment_score]")

    # Step 8: Weight matrix (3 → 2 for simplicity)
    np.random.seed(0)
    W = np.random.randn(3, 2) * 0.5
    print(f"\nStep 8: Weight Matrix W (3 × 2)")
    print(np.round(W, 4))
    print(f"  Shape: {W.shape}")

    # Step 9: ÃXW computation
    AX = A_norm @ X
    print(f"\nStep 9a: ÃX = Ã @ X (4 × 3)")
    print(np.round(AX, 4))
    print(f"  Shape: {AX.shape}  (N × F)")

    AXW = AX @ W
    print(f"\nStep 9b: ÃXW = (ÃX) @ W (4 × 2)")
    print(np.round(AXW, 4))
    print(f"  Shape: {AXW.shape}  (N × F_out)")

    # Step 10: ReLU
    H1 = np.maximum(AXW, 0)
    print(f"\nStep 10: ReLU(ÃXW) = H^(1) (4 × 2)")
    print(np.round(H1, 4))
    print(f"  Shape: {H1.shape}")
    print(f"  Negative values clipped to 0")

    print(f"\n--- Dimension Summary ---")
    print(f"  A          : {A.shape[0]} × {A.shape[1]}")
    print(f"  I          : {I.shape[0]} × {I.shape[1]}")
    print(f"  Â = A + I  : {A_hat.shape[0]} × {A_hat.shape[1]}")
    print(f"  D̂          : {D_hat.shape[0]} × {D_hat.shape[1]}")
    print(f"  D̂^(-1/2)   : {D_inv_sqrt.shape[0]} × {D_inv_sqrt.shape[1]}")
    print(f"  Ã          : {A_norm.shape[0]} × {A_norm.shape[1]}")
    print(f"  X          : {X.shape[0]} × {X.shape[1]}")
    print(f"  W          : {W.shape[0]} × {W.shape[1]}")
    print(f"  ÃX         : {AX.shape[0]} × {AX.shape[1]}")
    print(f"  ÃXW        : {AXW.shape[0]} × {AXW.shape[1]}")
    print(f"  ReLU(ÃXW)  : {H1.shape[0]} × {H1.shape[1]}")


if __name__ == "__main__":
    main()
