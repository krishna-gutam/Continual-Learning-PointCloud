# scripts/utils.py
# -*- coding: utf-8 -*-
"""
Utility functions and dataset helpers for Continual Learning on Point Clouds.

Exposed API (used by scripts):
- set_seed(seed: int, deterministic: bool = False)
- load_cfg(path: str) -> dict
- NPZPointDataset(root: str, split: str, class_list: Optional[Sequence[str]] = None, cache: bool = False)
- ece_score(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float
- get_device(pref: str = "cuda") -> torch.device
- ensure_dir(path: Union[str, Path]) -> Path
- worker_init_fn(worker_id): deterministic dataloader workers
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yaml


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU & CUDA).
    Optionally enable deterministic algorithms in cuDNN (may impact speed).

    Args:
        seed: base random seed
        deterministic: if True, set cudnn.deterministic=True and cudnn.benchmark=False
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow autotuner which can be faster on GPU
        torch.backends.cudnn.benchmark = True


def load_cfg(path: Union[str, Path]) -> Dict:
    """
    Load a YAML config as a Python dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


def get_device(pref: str = "cuda") -> torch.device:
    """
    Resolve preferred device with graceful fallback to CPU.
    """
    if pref.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref.lower() in ("mps", "metal") and torch.backends.mps.is_available():  # macOS MPS
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create directory if missing and return Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def worker_init_fn(worker_id: int) -> None:
    """
    Deterministic DataLoader worker init to vary NumPy Random Generator streams per worker.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


# ---------------------------------------------------------------------
# Dataset: NPZ point clouds
# ---------------------------------------------------------------------

class NPZPointDataset(Dataset):
    """
    Dataset over preprocessed point clouds stored as .npz:

        points: (N, 3) float32
        label : str   (class name, as written by preprocess script)
        file  : str   (original mesh path)
        meta  : object array [dataset, split, num_points]

    This dataset:
    - Scans <root>/<split>/*.npz
    - Filters by class_list if provided
    - Exposes `items` list = [(file_path, class_str), ...] for external remapping
    - Optionally caches point arrays in memory (cache=True)

    Note:
    - The dataset's internal label mapping (class_to_idx) is local.
      In continual learning, use a GLOBAL mapping wrapper (as in train.py) to align labels across tasks.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        class_list: Optional[Sequence[str]] = None,
        cache: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.cache = cache

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        # Discover files and collect (path, class_name)
        items: List[Tuple[str, str]] = []
        for p in sorted(split_dir.glob("*.npz")):
            try:
                data = np.load(p)
                lbl = data.get("label")
                if isinstance(lbl, np.ndarray):
                    if lbl.size == 1:
                        lbl = str(lbl.item())
                    else:
                        lbl = str(lbl)  # fallback
                lbl_str = str(lbl)
            except Exception:
                continue

            if (class_list is None) or (lbl_str in class_list):
                items.append((str(p), lbl_str))

        if len(items) == 0:
            raise RuntimeError(f"No NPZ files found under {split_dir} (class filter={class_list}).")

        self.items: List[Tuple[str, str]] = items
        # Build local class mapping
        classes = sorted({c for _, c in items})
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}

        # Optional in-memory cache
        self._cache_points: Dict[int, np.ndarray] = {}
        self._cache_labels: Dict[int, int] = {}
        if self.cache:
            for i, (path, cls) in enumerate(self.items):
                arr = np.load(path)
                pts = arr["points"].astype(np.float32)
                self._cache_points[i] = pts
                self._cache_labels[i] = self.class_to_idx[cls]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.cache and idx in self._cache_points:
            pts = self._cache_points[idx]
            y = self._cache_labels[idx]
        else:
            path, cls = self.items[idx]
            arr = np.load(path)
            pts = arr["points"].astype(np.float32)  # (N, 3)
            y = self.class_to_idx[cls]

        # Return tensors for training
        return torch.from_numpy(pts), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def ece_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Args:
        probs : (N, C) tensor of probabilities (softmax outputs).
        labels: (N,) tensor of true labels (int64).
        n_bins: number of confidence bins (default: 15 as in your thesis).
    Returns:
        Scalar float ECE in [0, 1].
    """
    if probs.ndim != 2:
        raise ValueError(f"ece_score expects probs with shape (N, C), got {tuple(probs.shape)}")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("labels must be shape (N,) and match probs.shape[0].")

    with torch.no_grad():
        confidences, predictions = probs.max(dim=1)  # (N,), (N,)
        accuracies = predictions.eq(labels)          # (N,)

        # Bin boundaries in (0,1]; put zero conf (rare) into first bin
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.zeros((), device=probs.device)

        for b in range(n_bins):
            # (a, b]
            in_bin = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
            prop = in_bin.float().mean()
            if prop.item() > 0:
                acc_bin = accuracies[in_bin].float().mean()
                conf_bin = confidences[in_bin].mean()
                ece += prop * (conf_bin - acc_bin).abs()

        return float(ece.item())
