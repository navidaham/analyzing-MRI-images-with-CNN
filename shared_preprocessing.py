"""
shared_preprocessing.py

Shared preprocessing utilities for BOTH:
- Keras MLP (ANN)
- PyTorch CNN

This matches the CNN group's preprocessing:
- load images from BASE_DATA_DIR/Training and BASE_DATA_DIR/Testing
- resize to IMG_SIZE x IMG_SIZE (e.g. 224x224)
- no cropping by default
- PyTorch augmentations for training (rotations, flips, normalization)
"""

import os
import sys
from typing import Tuple, List

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# PyTorch / torchvision (used for CNN helpers)
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

# ----------------- Ensure config is importable -----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import IMG_SIZE, CATEGORIES, BASE_DATA_DIR, VALIDATION_SPLIT  # type: ignore


# ----------------- Label mapping -----------------

# Ensure label indices are consistent everywhere
LABEL_MAP = {cat: idx for idx, cat in enumerate(CATEGORIES)}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
# 1. LOW-LEVEL IMAGE LOADING (COMMON TO ANN + CNN)
# ============================================================

def _get_image_paths_and_labels(split_dir: str) -> Tuple[List[str], List[int]]:
    """
    Get all image paths and labels from a split directory, without loading images.
    """
    image_paths: List[str] = []
    labels: List[int] = []

    for category in os.listdir(split_dir):
        category_path = os.path.join(split_dir, category)
        if not os.path.isdir(category_path):
            continue

        if category not in LABEL_MAP:
            continue

        class_idx = LABEL_MAP[category]

        for img_name in os.listdir(category_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(category_path, img_name)
            image_paths.append(img_path)
            labels.append(class_idx)

    return image_paths, labels


def _load_image(img_path: str, img_size: int) -> np.ndarray:
    """Load and resize a single image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.resize(img, (img_size, img_size))
    return img


def _load_split(split_dir: str, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all images and labels from a split directory.
    Uses preallocation to avoid memory fragmentation.
    """
    image_paths, labels = _get_image_paths_and_labels(split_dir)
    num_images = len(image_paths)
    
    if num_images == 0:
        return np.array([]).reshape(0, img_size, img_size, 3), np.array([], dtype=np.int64)
    
    # Preallocate array
    X = np.empty((num_images, img_size, img_size, 3), dtype=np.uint8)
    y = np.array(labels, dtype=np.int64)
    
    for i, path in enumerate(image_paths):
        img = _load_image(path, img_size)
        X[i] = img
    
    return X, y


def load_raw_dataset(
    base_dir: str = BASE_DATA_DIR,
    img_size: int = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train and test splits (no augmentation, no normalization),
    using the same logic as the PyTorch CNN notebook.

    Returns:
        X_train: (N_train, H, W, 3), uint8, BGR (OpenCV format)
        y_train: (N_train,), int labels
        X_test:  (N_test, H, W, 3), uint8, BGR
        y_test:  (N_test,), int labels
    """
    train_dir = os.path.join(base_dir, "Training")
    test_dir = os.path.join(base_dir, "Testing")

    X_train, y_train = _load_split(train_dir, img_size)
    X_test, y_test = _load_split(test_dir, img_size)

    print(f"[load_raw_dataset] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ============================================================
# 2. ANN-FRIENDLY DATA (NumPy, no augmentation by default)
# ============================================================

def get_ann_data(
    validation_split: float = VALIDATION_SPLIT,
    base_dir: str = BASE_DATA_DIR,
    img_size: int = IMG_SIZE,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get data for Keras ANN/MLP:

    - loads images as in the CNN preprocessing (no cropping, resized to IMG_SIZE)
    - splits training into train/val
    - returns NumPy arrays in (N, H, W, 3)
    - optionally scales to [0,1]

    Returns:
        X_train, X_val, y_train, y_val, X_test, y_test
    """
    X_train_full, y_train_full, X_test, y_test = load_raw_dataset(
        base_dir=base_dir,
        img_size=img_size,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_split,
        random_state=42,
        stratify=y_train_full,
    )

    if normalize:
        X_train = X_train.astype("float32") / 255.0
        X_val = X_val.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

    print(
        f"[get_ann_data] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )
    return X_train, X_val, y_train, y_val, X_test, y_test


# ============================================================
# 3. PYTORCH DATASETS & DATALOADERS (CNN)
# ============================================================

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor MRI images.

    Expects images as NumPy arrays in (H, W, 3), BGR (OpenCV).
    Converts to RGB + PIL, then applies torchvision transforms.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]       # (H, W, 3), BGR
        label = int(self.labels[idx])

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def get_torch_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, test_transform) that match the CNN group's code:

    - RandomRotation(20)
    - RandomHorizontalFlip
    - RandomVerticalFlip
    - ToTensor
    - Normalize(mean=[0.5]*3, std=[0.5]*3)
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    return train_transform, test_transform


def get_torch_dataloaders(
    base_dir: str = BASE_DATA_DIR,
    img_size: int = IMG_SIZE,
    batch_size: int = 32,
    num_workers: int = 2,
    augment_factor: int = 5,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for CNN, matching the notebook:

    - Load images (same as load_raw_dataset)
    - Train dataset = ConcatDataset of N copies with augmentation
    - Test dataset = no augmentation, only normalization

    Returns:
        train_loader, test_loader
    """
    X_train, y_train, X_test, y_test = load_raw_dataset(
        base_dir=base_dir,
        img_size=img_size,
    )

    train_transform, test_transform = get_torch_transforms()

    # Simulate N× augmentation by concatenating multiple augmented datasets
    train_datasets = [
        BrainTumorDataset(X_train, y_train, transform=train_transform)
        for _ in range(augment_factor)
    ]

    # Also include one non-augmented version if you want (optional):
    # train_datasets.append(BrainTumorDataset(X_train, y_train, transform=test_transform))

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = BrainTumorDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[get_torch_dataloaders] Train batches: {len(train_loader)}, "
        f"Test batches: {len(test_loader)}"
    )
    return train_loader, test_loader


# ============================================================
# 4. QUICK MANUAL TEST
# ============================================================

if __name__ == "__main__":
    # Quick check that loading works
    X_train, X_val, y_train, y_val, X_test, y_test = get_ann_data()
    print("ANN data shapes:")
    print("  Train:", X_train.shape)
    print("  Val:  ", X_val.shape)
    print("  Test: ", X_test.shape)

    # Optional: test PyTorch loaders (only if torch is installed)
    try:
        train_loader, test_loader = get_torch_dataloaders()
        print("CNN loaders OK.")
    except Exception as e:
        print("CNN loader test failed (maybe torch missing):", e)
