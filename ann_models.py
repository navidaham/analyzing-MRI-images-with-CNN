"""
ann_models.py

Feed-forward ANN architectures for brain MRI classification.
Designed to match the planned analysis steps:

Step 0: fix optimizer (Adam), early stopping, etc. -> handled in train_ann.py
Step 1: sweep learning rate & activation
Step 2: tune architecture (layers/units/dropout)
Step 3: sanity checks

This file provides:
- a flexible MLP builder (build_mlp)
- a small factory (build_ann_model) with named presets

Images are expected flattened: (IMG_SIZE*IMG_SIZE*3,)
"""
import tensorflow as tf
from typing import Optional, Sequence, Dict, Any

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    InputLayer,
    LeakyReLU,
)
from tensorflow.keras.optimizers import Adam

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import IMG_SIZE, CATEGORIES

# =============== UTILITIES ===============

def get_input_dim() -> int:
    """Flattened input dimension: H * W * 3."""
    return IMG_SIZE * IMG_SIZE * 3


def get_num_classes() -> int:
    """Number of classes from config."""
    return len(CATEGORIES)


# =============== FLEXIBLE MLP BUILDER ===============

def build_mlp(
    hidden_layers: Sequence[int] = (512, 256),
    activation: str = "relu",           # "relu", "leakyrelu", "sigmoid"
    input_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    learning_rate: float = 1e-3,
    use_batchnorm: bool = True,
    use_dropout: bool = True,
    dropout_rate: float = 0.5,
):
    """
    Generic MLP builder suitable for hyperparameter/architecture sweeps.

    Args:
        hidden_layers: list/tuple with units per hidden layer, e.g. (128, 64)
        activation: "relu", "leakyrelu", or "sigmoid"
        input_dim: flattened input size. If None, uses config-based size.
        num_classes: number of output classes. If None, uses config.
        learning_rate: Adam learning rate.
        use_batchnorm: whether to add BatchNorm after each hidden Dense layer.
        use_dropout: whether to add Dropout after each hidden Dense layer.
        dropout_rate: dropout probability.

    Returns:
        Compiled Keras model.
    """
    if input_dim is None:
        input_dim = get_input_dim()
    if num_classes is None:
        num_classes = get_num_classes()

    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))

    act = activation.lower()

    if act in ["relu", "leakyrelu"]:
        kernel_init = tf.keras.initializers.HeNormal()
    elif act in ["sigmoid", "tanh"]:
        kernel_init = tf.keras.initializers.GlorotUniform()
    else:
        kernel_init = tf.keras.initializers.GlorotUniform()


    for units in hidden_layers:
        if act == "leakyrelu":
            model.add(Dense(units, kernel_initializer=kernel_init))
            model.add(LeakyReLU(alpha=0.01))
        else:
            model.add(
                Dense(
                    units,
                    activation=act,
                    kernel_initializer=kernel_init,
                )
            )

        if use_batchnorm:
            model.add(BatchNormalization())
        if use_dropout:
            model.add(Dropout(dropout_rate))

    # Output layer
    model.add(
    Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )
)


    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============== PRESET MODELS FOR CONVENIENCE ===============

def build_baseline_ann(
    **kwargs: Dict[str, Any],
):
    """
    Baseline ANN:
    - 2 hidden layers: [512, 256]
    - ReLU
    - BatchNorm + Dropout ON (can be overridden via kwargs)
    """
    return build_mlp(
        hidden_layers=(512, 256),
        activation="relu",
        **kwargs,
    )


def build_deep_ann(
    **kwargs: Dict[str, Any],
):
    """
    Deeper ANN:
    - 3 hidden layers: [512, 256, 128]
    - ReLU
    - BatchNorm + Dropout ON (can be overridden)
    """
    return build_mlp(
        hidden_layers=(512, 256, 128),
        activation="relu",
        **kwargs,
    )


# =============== FACTORY ===============

def build_ann_model(
    name: str = "baseline",
    **kwargs: Dict[str, Any],
):
    """
    Factory to build a model by name OR use custom settings.

    Examples:
        # 1) Use preset baseline:
        model = build_ann_model("baseline")

        # 2) Use preset deep:
        model = build_ann_model("deep", learning_rate=5e-4)

        # 3) Custom architecture (for your sweeps):
        model = build_mlp(
            hidden_layers=(256, 128),
            activation="tanh",
            learning_rate=1e-3,
            use_dropout=False,
        )
    """
    name = name.lower()

    if name == "baseline":
        return build_baseline_ann(**kwargs)
    elif name in ["deep", "deep_ann"]:
        return build_deep_ann(**kwargs)
    elif name in ["mlp", "custom"]:
        # Direct pass-through to build_mlp
        return build_mlp(**kwargs)
    else:
        raise ValueError(f"Unknown ANN architecture name: {name}")


if __name__ == "__main__":
    # Quick sanity check
    input_dim = get_input_dim()
    num_classes = get_num_classes()

    print("Baseline ANN:")
    m1 = build_ann_model("baseline", input_dim=input_dim, num_classes=num_classes)
    m1.summary()

    print("\nDeep ANN:")
    m2 = build_ann_model("deep", input_dim=input_dim, num_classes=num_classes)
    m2.summary()

    print("\nCustom MLP (tanh, no dropout):")
    m3 = build_mlp(
        hidden_layers=(256, 128),
        activation="tanh",
        input_dim=input_dim,
        num_classes=num_classes,
        use_dropout=False,
    )
    m3.summary()
