"""
run_full_grid_and_heatmaps.py

Run a FULL grid of ANN configs:

    activation × learning_rate × architecture (hidden layer layout)

and then create 3 heatmaps (one per activation):

    - x-axis: architectures (hidden layer configs)
    - y-axis: learning rates
    - color: validation F1-score (weighted)
    - all heatmaps share the same color scale

This is a "super comparison" over both Steps 1 and 2.
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------- Ensure project root on sys.path ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_ann import run_ann_experiment  # type: ignore

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Define grid ----------

ACTIVATIONS = ["tanh"]
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
AUGMENT = True

# Architectures to test
HIDDEN_LAYER_CONFIGS: List[Tuple[int, ...]] = [
    (64,),
    (128,),
    (256,),
    (128, 64),
    (256, 128),
    (256, 128, 64),
]

LAYOUT_LABELS = ["64", "128", "256", "128-64", "256-128", "256-128-64"]

# Training settings (you can tweak these if needed)
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 7
USE_BATCHNORM = True
USE_DROPOUT = True
DROPOUT_RATE = 0.5
#AUGMENT = False  # ANN: no augmentation


def run_full_grid() -> List[Dict[str, Any]]:
    """
    Run the full grid of (activation, learning_rate, hidden_layers).

    Returns:
        List of metrics dicts (one per experiment).
    """
    all_metrics: List[Dict[str, Any]] = []

    print("=== ANN Full Grid: Activation × LR × Architecture ===")
    print(f"Activations:      {ACTIVATIONS}")
    print(f"Learning rates:   {LEARNING_RATES}")
    print(f"Architectures:    {HIDDEN_LAYER_CONFIGS}")
    print("-----------------------------------------------------\n")

    for activation in ACTIVATIONS:
        for lr in LEARNING_RATES:
            for layers in HIDDEN_LAYER_CONFIGS:
                print(f"\n>>> Running: act={activation}, lr={lr}, layers={layers}")
                metrics = run_ann_experiment(
                    name="full_grid",
                    hidden_layers=layers,
                    activation=activation,
                    learning_rate=lr,
                    batch_size=BATCH_SIZE,
                    max_epochs=MAX_EPOCHS,
                    early_stopping_patience=PATIENCE,
                    use_batchnorm=USE_BATCHNORM,
                    use_dropout=USE_DROPOUT,
                    dropout_rate=DROPOUT_RATE,
                    augment=AUGMENT,
                    save_weights=False,
                )
                all_metrics.append(metrics)
                print(
                    f"Result: val_accuracy={metrics['val_accuracy']:.4f}, "
                    f"val_f1_weighted={metrics['val_f1_weighted']:.4f}"
                )

    return all_metrics


def hidden_layers_to_tuple(v) -> Tuple[int, ...]:
    """
    Ensure hidden_layers is a tuple of ints so we can match them consistently.
    """
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    else:
        return (int(v),)


def build_f1_grid_for_activation(
    metrics_list: List[Dict[str, Any]],
    activation: str,
) -> np.ndarray:
    """
    Build a 2D grid of F1 scores for a given activation.

    Rows: LEARNING_RATES
    Cols: HIDDEN_LAYER_CONFIGS

    Cells with no matching experiment are np.nan.
    """
    grid = np.full(
        (len(LEARNING_RATES), len(HIDDEN_LAYER_CONFIGS)),
        np.nan,
        dtype=float,
    )

    for m in metrics_list:
        act = str(m.get("activation", "")).lower()
        if act != activation.lower():
            continue

        lr = float(m.get("learning_rate"))
        hidden = hidden_layers_to_tuple(m.get("hidden_layers", []))
        f1 = float(m.get("val_f1_weighted", np.nan))

        if lr in LEARNING_RATES and hidden in HIDDEN_LAYER_CONFIGS:
            i_lr = LEARNING_RATES.index(lr)
            j_layers = HIDDEN_LAYER_CONFIGS.index(hidden)
            grid[i_lr, j_layers] = f1

    return grid


def plot_heatmaps(metrics_list: List[Dict[str, Any]]):
    """
    Create 3 heatmaps (one per activation) with shared color scale.
    """
    grids = {}
    for act in ACTIVATIONS:
        grids[act] = build_f1_grid_for_activation(metrics_list, act)

    # Collect all non-NaN values to define global vmin/vmax
    non_nan_values = []
    for g in grids.values():
        if np.any(~np.isnan(g)):
            non_nan_values.append(g[~np.isnan(g)])
    if not non_nan_values:
        print("No valid F1 values found to plot.")
        return

    all_vals = np.concatenate(non_nan_values)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    n_acts = len(ACTIVATIONS)
    fig, axes = plt.subplots(1, n_acts, figsize=(5 * n_acts, 4), sharey=True)

    if n_acts == 1:
        axes = [axes]

    for idx, act in enumerate(ACTIVATIONS):
        ax = axes[idx]
        grid = grids[act]

        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="plasma_r"
        )

        # Axes labels
        ax.set_xticks(range(len(LAYOUT_LABELS)))
        ax.set_xticklabels(LAYOUT_LABELS, rotation=45, ha="right")

        ax.set_yticks(range(len(LEARNING_RATES)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in LEARNING_RATES])

        ax.set_xlabel("Architecture (hidden layers)")
        if idx == 0:
            ax.set_ylabel("Learning rate")

        ax.set_title(f"Activation: {act}")

        # Annotate cells with F1 values
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if grid[i, j] < (vmin + vmax) / 2 else "black",
                        fontsize=7,
                    )

    # Shared colorbar
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)

    cbar.set_label("Validation F1-score (weighted)")

    fig.suptitle("ANN Validation F1 across LR, Architecture, and Activation", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "ann_full_grid_heatmaps.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nHeatmaps saved to: {out_path}")


def main():
    # 1) Run full grid (this can take quite some time!)
    metrics_list = run_full_grid()

    # 2) Optionally save a CSV with all results for later analysis
    csv_path = os.path.join(LOGS_DIR, "ann_full_grid_results.csv")
    if metrics_list:
        import csv

        fieldnames = list(metrics_list[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics_list:
                writer.writerow(m)
        print(f"\nFull grid results saved to CSV: {csv_path}")

    # 3) Plot heatmaps
    plot_heatmaps(metrics_list)


if __name__ == "__main__":
    main()
