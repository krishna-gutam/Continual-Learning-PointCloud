#!/usr/bin/env python3
# scripts/evaluate.py
# -*- coding: utf-8 -*-
"""
Evaluate merged LoRA adapters on ModelNet10 and OOD (ModelNet40) with calibration & uncertainty.

Outputs (in results/):
  - accuracy_overall.json
  - accuracy_per_task.csv
  - accuracy_per_class.csv
  - calibration_ood.csv        (ECE on ID; mean predictive entropy on OOD)
  - confusion_matrix.png       (if matplotlib & seaborn available)

Modes:
  1) Deterministic (default):
     Loads a merged adapter (e.g., merged_ties.pth) and evaluates once.

  2) Bayesian ( --method bayesian ):
     Recreates an ensemble by sampling S merged adapters using per-task μ (taskXX_lora.pth)
     and diagonal σ² (taskXX_var.pth). Computes Entropy of the Mean (EoM) for ID and OOD.

Notes:
  - Uses a GLOBAL class mapping derived from cfg['data']['tasks'] for label alignment.
  - Requires that you already ran preprocessing for ModelNet10 and ModelNet40 to .npz.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Local modules
from utils import load_cfg, set_seed, ece_score
from models import (
    PointNetClassifier,
    PointTransformerClassifier,
    add_lora_to_linear,
)


# ---------------------------
# Dataset wrappers
# ---------------------------

class EvaluateNPZDataset(Dataset):
    """
    Reads *.npz produced by preprocess_modelnet.py and maps labels using a GLOBAL map.

    Each .npz has:
        points (N,3) float32
        label  (str) class name
    """

    def __init__(self, root: str, split: str, global_map: Dict[str, int], include_classes: Optional[Sequence[str]] = None):
        self.items: List[Tuple[str, str]] = []  # (filepath, label_str)
        r = Path(root) / split
        if not r.exists():
            raise FileNotFoundError(f"Split folder not found: {r}")
        # Collect files
        for p in sorted(r.glob("*.npz")):
            try:
                data = np.load(p)
                lbl = data.get("label")
                if isinstance(lbl, np.ndarray):
                    # stored as np.str_ sometimes
                    lbl = str(lbl.item()) if lbl.size == 1 else str(lbl)
                lbl = str(lbl)
            except Exception:
                continue
            if include_classes is None or lbl in include_classes:
                if lbl in global_map:  # only keep classes we know in global space
                    self.items.append((str(p), lbl))

        if len(self.items) == 0:
            raise RuntimeError(f"No .npz found for split={split} (root={root}). Have you preprocessed data?")

        self.global_map = global_map

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, cls = self.items[idx]
        arr = np.load(path)
        pts = arr["points"].astype(np.float32)  # (N,3)
        y = self.global_map[cls]
        return torch.from_numpy(pts), torch.tensor(y, dtype=torch.long), cls


class OODNPZDataset(Dataset):
    """
    OOD dataset: load ModelNet40 classes that are NOT in the in-distribution set.
    Returns points and dummy label (-1). We'll compute uncertainty on predictions.
    """

    def __init__(self, root: str, split: str, excluded_classes: Sequence[str], max_classes: int = 10):
        self.items: List[str] = []
        r = Path(root) / split
        if not r.exists():
            raise FileNotFoundError(f"Split folder not found: {r}")
        # Identify classes present in this split
        class_names = set()
        for p in r.glob("*.npz"):
            try:
                data = np.load(p)
                lbl = data.get("label")
                if isinstance(lbl, np.ndarray):
                    lbl = str(lbl.item()) if lbl.size == 1 else str(lbl)
                class_names.add(str(lbl))
            except Exception:
                continue

        # Choose OOD classes
        candidates = sorted([c for c in class_names if c not in set(excluded_classes)])
        chosen = candidates[:max_classes] if max_classes > 0 else candidates

        # Collect files of chosen classes
        for p in sorted(r.glob("*.npz")):
            try:
                data = np.load(p)
                lbl = data.get("label")
                if isinstance(lbl, np.ndarray):
                    lbl = str(lbl.item()) if lbl.size == 1 else str(lbl)
                lbl = str(lbl)
                if lbl in chosen:
                    self.items.append(str(p))
            except Exception:
                continue

        if len(self.items) == 0:
            raise RuntimeError("No OOD samples found (check preprocessing of ModelNet40).")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path = self.items[idx]
        arr = np.load(path)
        pts = arr["points"].astype(np.float32)
        return torch.from_numpy(pts), torch.tensor(-1, dtype=torch.long)  # dummy label


# ---------------------------
# Utilities
# ---------------------------

def build_global_class_map(tasks: Sequence[Sequence[str]]) -> Dict[str, int]:
    order: List[str] = []
    seen = set()
    for grp in tasks:
        for c in grp:
            if c not in seen:
                seen.add(c)
                order.append(c)
    return {c: i for i, c in enumerate(order)}


def get_backbone(cfg, num_classes: int):
    """
    Try Point Transformer; fallback to PointNet baseline.
    """
    try:
        return PointTransformerClassifier(num_classes=num_classes)
    except NotImplementedError:
        print("[WARN] PointTransformer not implemented; using PointNet baseline.")
        return PointNetClassifier(num_classes=num_classes)


def load_merged_adapter(model: torch.nn.Module, merged_path: Path) -> None:
    sd = model.state_dict()
    lora_state = torch.load(merged_path, map_state_dict=True, map_location="cpu") if hasattr(torch, "load") else torch.load(merged_path, map_location="cpu")
    sd.update(lora_state)
    model.load_state_dict(sd)


def find_default_merged(results_dir: Path) -> Optional[Path]:
    """
    Prefer merged_bayesian > merged_ties > merged_fisher > merged_simple if present.
    """
    for name in ["merged_bayesian.pth", "merged_ties.pth", "merged_fisher.pth", "merged_simple.pth"]:
        p = results_dir / name
        if p.exists():
            return p
    return None


def softmax_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Entropy H(p) = -sum p log p (per-sample)
    probs: (B, C)
    returns: (B,)
    """
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=1)


def entropy_of_mean(prob_stack: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    EoM for an ensemble:
        prob_stack: (S, B, C)  →  mean over S → entropy per sample
    returns: (B,)
    """
    mean_p = prob_stack.mean(dim=0).clamp_min(eps)
    return softmax_entropy(mean_p, eps=eps)


def evaluate_single_pass(model, loader, device, num_classes: int) -> Tuple[float, torch.Tensor, torch.Tensor, List[str]]:
    """
    One forward pass evaluation.
    Returns:
        acc, probs_all (N,C), labels_all (N,), class_names_seq (per sample for per-class metrics)
    """
    model.eval()
    all_probs, all_labels, all_names = [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for pts, y, cls_name in loader:
            pts = pts.to(device)
            y = y.to(device)
            logits = model(pts)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            all_names.extend(list(cls_name))
    acc = correct / max(1, total)
    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    return acc, probs_all, labels_all, all_names


def evaluate_ood_single_pass(model, ood_loader, device) -> torch.Tensor:
    """
    Returns per-sample predictive entropies for OOD (deterministic model).
    """
    model.eval()
    entropies = []
    with torch.no_grad():
        for pts, _ in ood_loader:
            pts = pts.to(device)
            probs = F.softmax(model(pts), dim=1).cpu()
            ent = softmax_entropy(probs)  # (B,)
            entropies.append(ent)
    return torch.cat(entropies, dim=0)


def bayesian_merged_samples(adapter_dir: Path, samples: int) -> List[Dict[str, torch.Tensor]]:
    """
    Build S merged LoRA states by sampling from per-task μ & σ² and precision-weighted averaging.
    Uses files:
        adapter_dir/taskXX_lora.pth
        adapter_dir/taskXX_var.pth
    """
    lora_paths = sorted((adapter_dir).glob("task*_lora.pth"))
    var_paths = sorted((adapter_dir).glob("task*_var.pth"))
    if not lora_paths or not var_paths or len(lora_paths) != len(var_paths):
        raise FileNotFoundError("Expected per-task lora and var files under results/adapters/ for Bayesian mode.")

    mus = [torch.load(p, map_location="cpu") for p in lora_paths]
    vars_ = [torch.load(p, map_location="cpu") for p in var_paths]

    keys = sorted(set(mus[0].keys()).intersection(*[set(m.keys()) for m in mus[1:]]))
    # Verify shapes
    for k in keys:
        ref_shape = mus[0][k].shape
        for m in mus[1:]:
            if m[k].shape != ref_shape:
                raise ValueError(f"Shape mismatch for key {k} across tasks.")
        for v in vars_:
            if v[k].shape != ref_shape:
                raise ValueError(f"Variance shape mismatch for key {k}.")

    S = samples
    merged_list: List[Dict[str, torch.Tensor]] = []
    eps = 1e-12

    for s in range(S):
        # Sample adapters for each task
        samples_t = []
        for mu, var in zip(mus, vars_):
            sample = {k: mu[k] + torch.randn_like(mu[k]) * (var[k] + eps).sqrt() for k in keys}
            samples_t.append(sample)
        # Precision-weighted merge for this sample
        merged_s: Dict[str, torch.Tensor] = {}
        for k in keys:
            prec = torch.stack([1.0 / (v[k] + eps) for v in vars_], dim=0)  # (T, ...)
            W = prec / prec.sum(dim=0, keepdim=True)
            S_stack = torch.stack([st[k] for st in samples_t], dim=0)
            merged_s[k] = (W * S_stack).sum(dim=0)
        merged_list.append(merged_s)

    return merged_list


def evaluate_bayesian_ensemble(
    model,
    loaders: Dict[str, DataLoader],
    device,
    adapter_dir: Path,
    num_classes: int,
    samples: int,
) -> Dict[str, float]:
    """
    Build an ensemble of S merged adapters by sampling and report:
      - acc (ID)
      - ECE (ID) [computed from mean probabilities]
      - EoM (ID) mean
      - OOD EoM mean
    """
    merged_states = bayesian_merged_samples(adapter_dir, samples=samples)

    # In-distribution: accumulate S probability tensors (S, N, C)
    id_loader = loaders["id"]
    model.eval()
    all_probs_s = []
    with torch.no_grad():
        for s, merged in enumerate(merged_states, start=1):
            # Load LoRA state for this sample
            sd = model.state_dict()
            sd.update(merged)
            model.load_state_dict(sd)

            probs_list = []
            labels_list = []
            names = []
            for pts, y, cls_name in id_loader:
                pts = pts.to(device)
                y = y.to(device)
                probs = F.softmax(model(pts), dim=1)
                probs_list.append(probs.cpu())
                labels_list.append(y.cpu())
                names.extend(list(cls_name))
            all_probs_s.append(torch.cat(probs_list, dim=0))
            labels_all = torch.cat(labels_list, dim=0)  # same for all S

    prob_stack = torch.stack(all_probs_s, dim=0)   # (S, N, C)
    mean_probs = prob_stack.mean(dim=0)            # (N, C)
    preds = mean_probs.argmax(dim=1)
    acc = (preds == labels_all).float().mean().item()

    # ECE on mean probs
    ece = ece_score(mean_probs, labels_all, n_bins=15)

    # EoM on ID
    eom_id = entropy_of_mean(prob_stack).mean().item()

    # OOD: compute EoM
    ood_loader = loaders.get("ood", None)
    eom_ood_mean = float("nan")
    if ood_loader is not None:
        entropies_ood_s = []
        with torch.no_grad():
            for s, merged in enumerate(merged_states, start=1):
                sd = model.state_dict()
                sd.update(merged)
                model.load_state_dict(sd)
                probs_ood_batches = []
                for pts, _ in ood_loader:
                    pts = pts.to(device)
                    probs = F.softmax(model(pts), dim=1)
                    probs_ood_batches.append(probs.cpu())
                entropies_ood_s.append(torch.cat(probs_ood_batches, dim=0))  # (N_ood, C)
        # Stack over S and compute EoM per sample
        prob_ood_stack = torch.stack(entropies_ood_s, dim=0)  # (S, N_ood, C)
        eom_ood_mean = entropy_of_mean(prob_ood_stack).mean().item()

    return {
        "accuracy_id": acc,
        "ece_id": ece,
        "eom_id": eom_id,
        "eom_ood": eom_ood_mean,
    }


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate merged LoRA adapter(s) on ModelNet10 (+OOD/Calibration).")
    ap.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    ap.add_argument("--merged", type=str, default=None, help="Path to merged adapter (if not provided, auto-detect)")
    ap.add_argument("--method", type=str, default="deterministic", choices=["deterministic", "bayesian"],
                    help="Evaluation method: single merged adapter vs Bayesian ensemble sampling")
    ap.add_argument("--samples", type=int, default=None, help="Bayesian: number of samples S (default from YAML or 10)")
    ap.add_argument("--ece-bins", type=int, default=None, help="Bins for ECE (default from YAML or 15)")
    ap.add_argument("--ood-classes", type=int, default=None, help="How many OOD classes to use (default from YAML or 10)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(cfg["evaluation"]["save_dir"])
    adapter_dir = results_dir / "adapters"

    # Global mapping & paths
    tasks = [list(x) for x in cfg["data"]["tasks"]]
    global_map = build_global_class_map(tasks)
    inv_global = {v: k for k, v in global_map.items()}

    id_root = str(Path(cfg["data"]["root"]) / cfg["data"]["dataset"])
    ood_ds_name = cfg["data"].get("ood_dataset", "ModelNet40")
    ood_root = str(Path(cfg["data"]["root"]) / ood_ds_name)
    ece_bins = int(args.ece_bins or cfg["evaluation"].get("ece_bins", 15))
    ood_k = int(args.ood_classes or cfg["data"].get("ood_num_classes", 10))
    S = int(args.samples or cfg.get("uncertainty", {}).get("laplace", {}).get("samples", 10))

    # Build model + LoRA shell
    num_classes = cfg["model"].get("num_classes", len(global_map))
    model = get_backbone(cfg, num_classes=num_classes).to(device)
    for p in model.parameters():  # base frozen
        p.requires_grad = False
    lora = cfg["model"]["lora"]
    add_lora_to_linear(model, r=int(lora.get("rank", 8)), alpha=int(lora.get("alpha", 16)), dropout=float(lora.get("dropout", 0.0)))

    # Datasets
    id_test = EvaluateNPZDataset(id_root, "test", global_map)
    id_loader = DataLoader(id_test, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # OOD loader (optional—skip if data not present)
    ood_loader = None
    try:
        ood_loader = DataLoader(
            OODNPZDataset(ood_root, "test", excluded_classes=list(global_map.keys()), max_classes=ood_k),
            batch_size=64, shuffle=False, num_workers=2, pin_memory=True
        )
    except Exception as e:
        print(f"[WARN] OOD dataset not available or empty: {e}")

    # ------------------ Deterministic evaluation ------------------
    if args.method == "deterministic":
        merged_path = Path(args.merged) if args.merged else find_default_merged(results_dir)
        if merged_path is None:
            raise FileNotFoundError(
                "No merged adapter found. Provide --merged or run merge step (ties/fisher/simple/bayesian)."
            )
        print(f"Using merged adapter: {merged_path}")
        load_merged_adapter(model, merged_path)

        # Evaluate ID
        acc, probs_all, labels_all, class_names_seq = evaluate_single_pass(model, id_loader, device, num_classes=num_classes)
        ece = ece_score(probs_all, labels_all, n_bins=ece_bins)

        # Per-class accuracy
        per_class_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
        preds = probs_all.argmax(dim=1)
        for i in range(len(labels_all)):
            cls_name = class_names_seq[i]
            correct = int(preds[i].item() == labels_all[i].item())
            per_class_stats[cls_name][0] += correct
            per_class_stats[cls_name][1] += 1

        # Per-task accuracy
        task_accs = []
        for ti, task_classes in enumerate(tasks, start=1):
            idxs = [i for i, cname in enumerate(class_names_seq) if cname in task_classes]
            if len(idxs) == 0:
                task_accs.append((ti, float("nan")))
                continue
            t_correct = sum(int(preds[i].item() == labels_all[i].item()) for i in idxs)
            task_accs.append((ti, t_correct / len(idxs)))

        # OOD predictive entropy (deterministic)
        ood_entropy_mean = float("nan")
        if ood_loader is not None:
            ent = evaluate_ood_single_pass(model, ood_loader, device)
            ood_entropy_mean = ent.mean().item()

        # Save files
        results_dir.mkdir(parents=True, exist_ok=True)
        # Overall
        with open(results_dir / "accuracy_overall.json", "w") as f:
            json.dump({"accuracy": acc, "ece": ece, "ood_entropy_mean": ood_entropy_mean}, f, indent=2)

        # Per-task
        with open(results_dir / "accuracy_per_task.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["task_id", "classes", "accuracy"])
            for ti, a in task_accs:
                cw.writerow([ti, "|".join(tasks[ti - 1]), f"{a:.6f}"])

        # Per-class
        with open(results_dir / "accuracy_per_class.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["class", "correct", "total", "accuracy"])
            for cls in sorted(per_class_stats.keys()):
                c, t = per_class_stats[cls]
                cw.writerow([cls, c, t, f"{(c / t) if t else 0.0:.6f}"])

        # Calibration & OOD summary
        with open(results_dir / "calibration_ood.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["metric", "value"])
            cw.writerow(["ece_id", f"{ece:.6f}"])
            cw.writerow(["ood_entropy_mean", f"{ood_entropy_mean:.6f}"])

        # Confusion matrix plot (optional)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(labels_all.numpy(), preds.numpy(), labels=list(range(num_classes)))
            # Re-order cm rows/cols to match global_map order
            classes_order = [inv_global[i] for i in range(num_classes)]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, ax=ax,
                        xticklabels=classes_order, yticklabels=classes_order)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix (Merged)")
            fig.tight_layout()
            plt.savefig(results_dir / "confusion_matrix.png", dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Skipping confusion matrix plot: {e}")

        print(f"\nAccuracy={acc*100:.2f}%  ECE={ece:.4f}  OOD entropy={ood_entropy_mean:.4f}")
        print(f"Saved: {results_dir}/accuracy_overall.json, accuracy_per_task.csv, accuracy_per_class.csv, calibration_ood.csv")

    # ------------------ Bayesian ensemble evaluation ------------------
    else:
        print("Bayesian mode: sampling merged adapters via Laplace variances.")
        loaders = {"id": id_loader}
        if ood_loader is not None:
            loaders["ood"] = ood_loader
        metrics = evaluate_bayesian_ensemble(
            model=model,
            loaders=loaders,
            device=device,
            adapter_dir=adapter_dir,
            num_classes=num_classes,
            samples=S,
        )
        # Save
        with open(results_dir / "bayesian_eval.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nBayesian ensemble results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
