"""
evaluate_ann.py

Evaluate the FINAL chosen ANN configuration on the TEST set.

Outputs:
- Test loss
- Test accuracy
- Test F1 (weighted & macro)
- Confusion matrix (PNG)
- ROC curves + macro AUC (PNG)
- Inference time (total + per image)
- Model size (params + weights file size)
- Training time (from metrics JSON)
- Optional: mean ± std of validation metrics across multiple runs of same config
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# --------- Ensure project root on sys.path ---------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------- Local imports ---------
from config import CATEGORIES  # type: ignore
from src.shared_preprocessing import get_ann_data
from src.train_ann import normalize_to_minus1_plus1  # if you added it
from models.ann_models import build_mlp  # type: ignore


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ========= 1. DEFINE YOUR FINAL CONFIGURATION HERE =========

FINAL_CONFIG = {
    "name": "final_ann",  # experiment "name" used in architecture sweep
    "activation": "relu",
    "learning_rate": 1e-4,
    "hidden_layers": [64],               # <- your best architecture
    "use_batchnorm": True,
    "use_dropout": True,
    "dropout_rate": 0.5,
}

# ===========================================================


def load_matching_metrics(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load all metrics JSON files that match the final configuration
    (name, activation, lr, hidden_layers).

    This allows multiple runs of the same config -> mean ± std.
    If only one run exists, that's fine too.
    """
    matches: List[Dict[str, Any]] = []

    for fname in os.listdir(LOGS_DIR):
        if not fname.endswith("_metrics.json"):
            continue

        path = os.path.join(LOGS_DIR, fname)
        with open(path, "r") as f:
            m = json.load(f)

        if (
            m.get("name") == config["name"]
            and m.get("activation") == config["activation"]
            and float(m.get("learning_rate")) == float(config["learning_rate"])
            and list(m.get("hidden_layers")) == list(config["hidden_layers"])
        ):
            m["_metrics_path"] = path  # store where it came from
            matches.append(m)

    return matches


def select_best_run(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Among all matching runs, select the one with best validation F1
    (then best validation accuracy).
    """
    if not matches:
        raise RuntimeError("No matching metrics JSONs found for FINAL_CONFIG.")

    best = sorted(
        matches,
        key=lambda m: (m["val_f1_weighted"], m["val_accuracy"]),
        reverse=True,
    )[0]
    return best


def summarize_val_stats(matches: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute mean ± std over validation metrics across runs with same config.
    """
    vals_acc = [m["val_accuracy"] for m in matches]
    vals_f1 = [m["val_f1_weighted"] for m in matches]

    arr_acc = np.array(vals_acc, dtype=float)
    arr_f1 = np.array(vals_f1, dtype=float)

    return {
        "val_acc_mean": float(arr_acc.mean()),
        "val_acc_std": float(arr_acc.std(ddof=0)),
        "val_f1_mean": float(arr_f1.mean()),
        "val_f1_std": float(arr_f1.std(ddof=0)),
        "num_runs": len(matches),
    }


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    # Normalize per row (true class)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Print values
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    save_path: str,
):
    """
    Plot one-vs-rest ROC curves for each class and macro-average AUC.
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(7, 6))

    # Per-class ROC + AUC
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        aucs.append(auc_i)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_i:.3f})")

    # Macro-average AUC
    macro_auc = float(np.mean(aucs))

    # Random baseline
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (Macro AUC = {macro_auc:.3f})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return macro_auc


def main():
    # 1. Load matching runs of the final config
    matches = load_matching_metrics(FINAL_CONFIG)
    if not matches:
        raise RuntimeError("No runs found that match FINAL_CONFIG. Check your settings.")

    print(f"Found {len(matches)} run(s) for final config.")

    best_run = select_best_run(matches)
    val_stats = summarize_val_stats(matches)

    print("\n=== Validation Summary for Final Config ===")
    print(f"Config name:        {FINAL_CONFIG['name']}")
    print(f"Activation:         {FINAL_CONFIG['activation']}")
    print(f"Learning rate:      {FINAL_CONFIG['learning_rate']}")
    print(f"Hidden layers:      {FINAL_CONFIG['hidden_layers']}")
    print(f"Runs considered:    {val_stats['num_runs']}")
    print(f"Val acc (best run): {best_run['val_accuracy']:.4f}")
    print(f"Val F1  (best run): {best_run['val_f1_weighted']:.4f}")
    print(
        f"Val acc (mean±std): {val_stats['val_acc_mean']:.4f} ± {val_stats['val_acc_std']:.4f}"
    )
    print(
        f"Val F1  (mean±std): {val_stats['val_f1_mean']:.4f} ± {val_stats['val_f1_std']:.4f}"
    )

    # 2. Load data (test set)
    print("\nLoading data (test set)...")
    _, _, _, _, X_test, y_test = get_ann_data(normalize=False)
    X_test_flat = normalize_to_minus1_plus1(X_test).reshape(len(X_test), -1)


    # 3. Rebuild model and load weights from best run
    print("\nRebuilding model and loading weights from best run...")
    input_dim = X_test_flat.shape[1]
    num_classes = len(CATEGORIES)

    model = build_mlp(
        hidden_layers=best_run["hidden_layers"],
        activation=best_run["activation"],
        input_dim=input_dim,
        num_classes=num_classes,
        learning_rate=best_run["learning_rate"],
        use_batchnorm=best_run["use_batchnorm"],
        use_dropout=best_run["use_dropout"],
        dropout_rate=best_run["dropout_rate"],
    )

    weights_path = best_run["model_path"]
    model.load_weights(weights_path)

    # 4. Test evaluation: loss + accuracy + F1 + confusion matrix + ROC/AUC + inference time
    print("\nEvaluating on TEST set...")
    test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)

    # Inference timing
    t0 = time.time()
    y_proba = model.predict(X_test_flat, batch_size=32, verbose=0)
    t1 = time.time()
    total_inf_time = t1 - t0
    per_image_time = total_inf_time / len(X_test_flat)

    y_pred = np.argmax(y_proba, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
    test_f1_macro = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred)

    cm_path = os.path.join(
        PLOTS_DIR, f"{best_run['exp_id']}_confusion_matrix.png"
    )
    roc_path = os.path.join(
        PLOTS_DIR, f"{best_run['exp_id']}_roc_curves.png"
    )

    plot_confusion_matrix(cm, CATEGORIES, cm_path)
    test_macro_auc = plot_roc_curves(y_test, y_proba, CATEGORIES, roc_path)

    # 5. Model size / parameters / training time
    num_params = best_run["num_params"]
    weights_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    train_time_sec = best_run["train_time_sec"]

    print("\n=== TEST RESULTS (Final ANN) ===")
    print(f"Test loss:          {test_loss:.4f}")
    print(f"Test accuracy:      {test_accuracy:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")
    print(f"Test F1 (macro):    {test_f1_macro:.4f}")
    print(f"Test macro AUC:     {test_macro_auc:.4f}")
    print(f"Total inf time:     {total_inf_time:.4f} s")
    print(f"Per-image inf time: {per_image_time*1000:.4f} ms/image")
    print(f"Params:             {num_params}")
    print(f"Weights file size:  {weights_size_mb:.2f} MB")
    print(f"Train time (best):  {train_time_sec:.2f} s")

    print(f"\nConfusion matrix saved to: {cm_path}")
    print(f"ROC curves saved to:       {roc_path}")

    # 6. Save full test metrics to JSON
    test_metrics = {
        "config": FINAL_CONFIG,
        "best_run_exp_id": best_run["exp_id"],
        "val_summary": val_stats,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_f1_weighted": float(test_f1_weighted),
        "test_f1_macro": float(test_f1_macro),
        "test_macro_auc": float(test_macro_auc),
        "total_inference_time_sec": float(total_inf_time),
        "per_image_inference_time_sec": float(per_image_time),
        "num_params": int(num_params),
        "weights_size_mb": float(weights_size_mb),
        "train_time_sec": float(train_time_sec),
        "confusion_matrix_path": cm_path,
        "roc_curves_path": roc_path,
    }

    out_json_path = os.path.join(
        LOGS_DIR, f"{best_run['exp_id']}_TEST_METRICS.json"
    )
    with open(out_json_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"\nFull test metrics saved to: {out_json_path}")


if __name__ == "__main__":
    main()
