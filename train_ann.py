"""
train_ann.py

Train a single ANN configuration on the MRI dataset.

This script is designed to support:
- Step 0: basic setup (Adam, batch size, early stopping, val set)
- Step 1: LR + activation sweeps
- Step 2: architecture (layout) tuning

It trains ONE configuration at a time and:
- loads data (no augmentation by default for ANN)
- flattens + normalizes images
- builds ANN from models.ann_models
- trains with early stopping
- evaluates on validation set (accuracy + F1)
- saves:
    - model file          -> results/models/
    - training curves     -> results/plots/
    - metrics JSON        -> results/logs/

This file is meant to be called either:
- directly (for a quick test), or
- from run_experiments.py when looping over configs.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Sequence, Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------- Ensure project root is on sys.path -----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------- Local imports -----------
from config import IMG_SIZE, CATEGORIES  # type: ignore
from src.shared_preprocessing import get_ann_data  # if this import fails, try: from data_loading import get_data
from models.ann_models import build_mlp  # type: ignore


# ================== CONFIG-LIKE CONSTANTS ==================

# You can move these into config.py later if you prefer
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 100
EARLY_STOPPING_PATIENCE = 7

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [RESULTS_DIR, LOGS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ================== HELPER FUNCTIONS ==================

def flatten_and_normalize(
    X: np.ndarray,
) -> np.ndarray:
    """
    Flatten (N, H, W, C) -> (N, H*W*C) and scale to [0, 1].
    """
    N = X.shape[0]
    X = X.astype("float32") / 255.0
    X = X.reshape(N, -1)
    return X


def make_experiment_id(
    name: str,
    activation: str,
    learning_rate: float,
    hidden_layers: Sequence[int],
) -> str:
    """
    Construct a readable experiment ID.
    Example: "ann_relu_lr-0.001_layers-256-128_2025-11-25-1530"
    """
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    layers_str = "-".join(str(h) for h in hidden_layers)
    exp_id = f"{name}_act-{activation}_lr-{learning_rate}_layers-{layers_str}_{time_str}"
    # Make it filesystem-friendly
    exp_id = exp_id.replace(" ", "").replace(":", "-")
    return exp_id


def plot_history(history, save_path: str, title: str = "Training History"):
    """
    Plot training & validation loss/accuracy curves.
    """
    hist = history.history

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="Train loss")
    plt.plot(hist.get("val_loss", []), label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Accuracy
    if "accuracy" in hist:
        plt.subplot(1, 2, 2)
        plt.plot(hist.get("accuracy", []), label="Train acc")
        plt.plot(hist.get("val_accuracy", []), label="Val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def normalize_to_minus1_plus1(X: np.ndarray) -> np.ndarray:
    """
    Match the CNN normalization:
      ToTensor() -> [0,1]
      Normalize(mean=0.5,std=0.5) -> (x-0.5)/0.5 -> [-1,1]
    Here X is assumed to be uint8 [0..255] OR float [0..1].
    """
    X = X.astype("float32")
    if X.max() > 1.0:
        X = X / 255.0
    X = (X - 0.5) / 0.5
    return X


def make_tf_datasets_for_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    augment: bool,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create tf.data datasets that:
    - optionally apply augmentation (rotation + flips) on-the-fly
    - normalize to [-1,1]
    - flatten to vectors for MLP input
    """
    # Keras augmentation layers (roughly matches torchvision RandomRotation(20) + flips)
    rot_factor = 20.0 / 360.0  # 20 degrees
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(rot_factor),
    ])

    def preprocess(x, y, do_aug: bool):
        x = tf.cast(x, tf.float32)
        x = x / 255.0
        if do_aug:
            x = aug(x, training=True)
        # Normalize to [-1,1] like torchvision Normalize(mean=0.5,std=0.5)
        x = (x - 0.5) / 0.5
        # Flatten for MLP
        x = tf.reshape(x, [-1])
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_ds = train_ds.shuffle(min(len(X_train), 2000), seed=42, reshuffle_each_iteration=True)
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: preprocess(x, y, False), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# ================== CORE: SINGLE EXPERIMENT ==================

def run_ann_experiment(
    name: str,
    hidden_layers: Sequence[int],
    activation: str,
    learning_rate: float,
    batch_size: int = BATCH_SIZE_DEFAULT,
    max_epochs: int = MAX_EPOCHS_DEFAULT,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    use_batchnorm: bool = True,
    use_dropout: bool = True,
    dropout_rate: float = 0.5,
    augment: bool = False,
    save_weights: bool = True,
) -> Dict[str, Any]:
    """
    Train a single ANN configuration and evaluate on validation set.

    Args:
        name: experiment short name (e.g. "step1_lr_act_sweep")
        hidden_layers: tuple/list of units per hidden layer (e.g. (256, 128))
        activation: "relu", "leakyrelu", "sigmoid"
        learning_rate: Adam learning rate
        batch_size: batch size
        max_epochs: maximum number of epochs (early stopping will usually stop before this)
        early_stopping_patience: patience for EarlyStopping (val_loss)
        use_batchnorm: whether to use BatchNormalization after each hidden layer
        use_dropout: whether to use Dropout after each hidden layer
        dropout_rate: dropout probability
        augment: whether to use augmentation in get_data (for ANN we typically keep this False)

    Returns:
        metrics dict with:
            - val_accuracy
            - val_f1_weighted
            - train_time_sec
            - best_epoch
            - num_params
            - exp_id
            - and hyperparameters used
    """
    # -------- Load data (no augmentation for ANN by default) --------
    print("Loading data...")
    X_train, X_val, y_train, y_val, X_test, y_test = get_ann_data(normalize=False)

    # -------- Flatten & normalize --------
    print("Preprocessing (flatten + normalize)...")
    train_ds, val_ds = make_tf_datasets_for_mlp(
        X_train, y_train, X_val, y_val,
        batch_size=batch_size,
        augment=augment,
    )

    # input dim is fixed: IMG_SIZE * IMG_SIZE * 3
    input_dim = IMG_SIZE * IMG_SIZE * 3
    num_classes = len(CATEGORIES)

    # -------- Build model --------
    exp_id = make_experiment_id(name, activation, learning_rate, hidden_layers)
    print(f"Building model for experiment: {exp_id}")

    model = build_mlp(
        hidden_layers=hidden_layers,
        activation=activation,
        input_dim=input_dim,
        num_classes=num_classes,
        learning_rate=learning_rate,
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
    )

    # -------- Callbacks (EarlyStopping, optional checkpoint) --------
    model_path = None
    callbacks = []

    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stopping_cb)

    if save_weights:
        model_path = os.path.join(MODELS_DIR, f"{exp_id}.weights.h5")
        checkpoint_cb = ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        callbacks.append(checkpoint_cb)


    # -------- Training --------
    print("Starting training...")
    start_time = time.time()
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=max_epochs,
    callbacks=callbacks,
    verbose=1,
)
    train_time_sec = time.time() - start_time
    print(f"Training finished in {train_time_sec:.2f} seconds.")

    # Determine best epoch (last point in history after early stopping)
    best_epoch = len(history.history.get("loss", []))

    # -------- Validation metrics (NO augmentation) --------
    print("Evaluating on validation set...")

    # Normalize + flatten validation data exactly like training (but without augmentation)
    X_val_n = normalize_to_minus1_plus1(X_val).reshape(len(X_val), -1)

    y_val_pred_proba = model.predict(X_val_n, batch_size=batch_size)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted")

    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 (weighted): {val_f1_weighted:.4f}")


    # -------- Save training curves --------
    plot_path = os.path.join(PLOTS_DIR, f"{exp_id}_history.png")
    plot_history(history, plot_path, title=f"Training History - {exp_id}")

    # -------- Collect & save metrics --------
    num_params = model.count_params()

    metrics = {
        "exp_id": exp_id,
        "name": name,
        "hidden_layers": list(hidden_layers),
        "activation": activation,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "use_batchnorm": use_batchnorm,
        "use_dropout": use_dropout,
        "dropout_rate": dropout_rate,
        "augment": augment,
        "val_accuracy": float(val_accuracy),
        "val_f1_weighted": float(val_f1_weighted),
        "train_time_sec": float(train_time_sec),
        "best_epoch": int(best_epoch),
        "num_params": int(num_params),
        "model_path": model_path,
        "history_plot_path": plot_path,
    }

    metrics_path = os.path.join(LOGS_DIR, f"{exp_id}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to:   {model_path}")
    print(f"Plot saved to:    {plot_path}")

    return metrics


# ================== DEMO USAGE ==================

if __name__ == "__main__":
    """
    Quick test run:
    - ReLU
    - lr = 1e-3
    - simple layout: [256, 128]

    You can later comment this out and control everything from run_experiments.py
    """
    demo_metrics = run_ann_experiment(
        name="demo_baseline",
        hidden_layers=(128, 64),
        activation="relu",
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=50,
        early_stopping_patience=7,
        use_batchnorm=True,
        use_dropout=True,
        dropout_rate=0.5,
        augment=False,  # ANN: no augmentation by default
    )

    print("\nDemo experiment finished. Summary:")
    for k, v in demo_metrics.items():
        print(f"{k}: {v}")
