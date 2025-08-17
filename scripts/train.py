#!/usr/bin/env python3
# scripts/train.py
# -*- coding: utf-8 -*-
"""
Task-wise training of LoRA adapters for continual learning on ModelNet.

Pipeline
--------
1) Load config (YAML) describing task splits, hyperparameters, paths.
2) Build backbone (Point Transformer if available, else PointNet baseline).
3) Freeze base weights, inject LoRA (rank/alpha/dropout from cfg).
4) For each task:
   - Reset LoRA params (fresh adapters).
   - Train ONLY LoRA params on the task's classes.
   - (Optional) Estimate diagonal Fisher and Laplace variances.
   - Save per-task artifacts:
        results/adapters/taskXX_lora.pth
        results/adapters/taskXX_fisher.pth        (if enabled)
        results/adapters/taskXX_var.pth           (if enabled)
   - Log metrics to results/log_taskXX.json

Notes
-----
- Labels are mapped to a GLOBAL index (built from the union of all classes in cfg['data']['tasks']),
  ensuring the classifier head (num_classes) is used consistently across tasks.
- The base model remains frozen throughout; only LoRA (A/B) trains task-wise.
- To merge adapters later, use scripts/merge_models.py
- To evaluate accuracy/ECE/OOD on the merged model, use scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Local imports
from utils import load_cfg, set_seed, NPZPointDataset
from models import (
    PointNetClassifier,
    PointTransformerClassifier,
    add_lora_to_linear,
    mark_only_lora_as_trainable,
    extract_lora_state,
    estimate_fisher_diagonal,
    laplace_variance_from_fisher,
    count_lora_params,
)

# -------------------------
# Data utilities
# -------------------------

class GlobalLabelProxy(Dataset):
    """
    Wrap an NPZPointDataset but remap labels using a provided global mapping
    (string class name -> global index).

    This ensures that, across tasks, labels always point to the same output
    indices in the classifier head (e.g., 0..9 for ModelNet10).
    """
    def __init__(self, base_ds: NPZPointDataset, global_map: Dict[str, int]):
        self.base = base_ds
        self.global_map = global_map
        # sanity: all labels in dataset must exist in global map
        ds_classes = sorted({cls for _, cls in self.base.items})
        missing = [c for c in ds_classes if c not in self.global_map]
        if missing:
            raise ValueError(f"Classes missing in global_map: {missing}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        p, cls = self.base.items[idx]  # (path, class_name)
        data = np.load(p)
        pts = torch.from_numpy(data["points"].astype(np.float32))  # (N,3)
        y = self.global_map[cls]  # global label index
        return pts, torch.tensor(y, dtype=torch.long)


def build_global_class_map(tasks: Sequence[Sequence[str]]) -> Dict[str, int]:
    """
    Build a deterministic class->index mapping from tasks list-of-lists.
    Keeps first appearance order.
    """
    order: List[str] = []
    seen = set()
    for group in tasks:
        for c in group:
            if c not in seen:
                seen.add(c)
                order.append(c)
    return {c: i for i, c in enumerate(order)}


# -------------------------
# Training helpers
# -------------------------

def reset_lora_params(module: nn.Module) -> None:
    """
    Re-initialize all LoRA A/B parameters so each task starts from a fresh adapter.
    """
    for name, p in module.named_parameters():
        if "lora_A" in name:
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)
        elif "lora_B" in name:
            nn.init.zeros_(p)


@dataclass
class TrainStats:
    task_id: int
    classes: List[str]
    epochs: int
    lr: float
    batch_size: int
    final_loss: float
    final_acc: float
    lora_params: int
    total_trainable: int
    base_frozen: int

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_one_task(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip: float = 0.0,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Train only LoRA params for a single task.
    Returns: (final_loss, final_acc)
    """
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    ce = nn.CrossEntropyLoss()

    final_loss, final_acc = 0.0, 0.0
    for ep in range(1, epochs + 1):
        running_loss, running_correct, running_total = 0.0, 0, 0
        for pts, y in loader:
            pts = pts.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
                logits = model(pts)
                loss = ce(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += y.size(0)

        final_loss = running_loss / max(1, running_total)
        final_acc = running_correct / max(1, running_total)
        print(f"  Epoch {ep:02d}/{epochs:02d}  loss={final_loss:.4f}  acc={final_acc*100:.2f}%")

    return final_loss, final_acc


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Train LoRA adapters task-wise for continual learning.")
    ap.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    ap.add_argument("--no-fisher", action="store_true", help="Disable Fisher estimation per task")
    ap.add_argument("--no-laplace", action="store_true", help="Disable Laplace variance saving per task")
    ap.add_argument("--fisher-max-batches", type=int, default=4, help="Batches to estimate Fisher (speed/quality)")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max norm (0 = off)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed", 42))

    device = torch.device(
        cfg["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )

    # Build GLOBAL class mapping (order matters to be reproducible)
    tasks: List[List[str]] = [list(x) for x in cfg["data"]["tasks"]]
    global_map = build_global_class_map(tasks)
    num_classes = cfg["model"].get("num_classes", len(global_map))
    assert num_classes >= len(global_map), "num_classes must cover all classes in tasks."

    data_root = Path(cfg["data"]["root"]) / cfg["data"]["dataset"]
    save_dir = Path(cfg["evaluation"]["save_dir"])
    (save_dir / "adapters").mkdir(parents=True, exist_ok=True)

    # Build backbone
    def get_backbone():
        try:
            # Try Point Transformer first
            return PointTransformerClassifier(num_classes=num_classes)
        except NotImplementedError:
            print("[WARN] PointTransformer not implemented; using PointNet baseline.")
            return PointNetClassifier(num_classes=num_classes)

    model = get_backbone().to(device)

    # Freeze base weights
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA
    lora_cfg = cfg["model"]["lora"]
    add_lora_to_linear(
        model,
        r=int(lora_cfg.get("rank", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    # Only LoRA trainable
    mark_only_lora_as_trainable(model, train_classifier=False)

    # Print LoRA param counts (sanity)
    counts = count_lora_params(model)
    print(
        f"[LoRA] trainable={counts.total_trainable} (lora={counts.lora_trainable})  base_frozen={counts.base_frozen}"
    )

    # Training settings
    epochs = int(cfg["training"]["epochs_per_task"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    use_amp = bool(args.amp)

    # Task loop
    per_task_logs: List[Dict] = []
    for task_id, classes in enumerate(tasks, start=1):
        print(f"\n===> Task {task_id} / {len(tasks)}   classes = {classes}")

        # Reset LoRA parameters (fresh adapter for this task)
        reset_lora_params(model)

        # Dataset for this task (train split), but remap to GLOBAL labels
        base_ds = NPZPointDataset(str(data_root), "train", class_list=classes)
        # Replace the dataset's internal mapping by wrapping with global-map proxy
        train_ds = GlobalLabelProxy(base_ds, global_map)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )

        # Train one task (LoRA-only)
        final_loss, final_acc = train_one_task(
            model,
            train_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            grad_clip=float(args.grad_clip),
            use_amp=use_amp,
        )

        # Save LoRA adapter (A/B only)
        lora_state = extract_lora_state(model, to_cpu=True)
        lora_path = save_dir / "adapters" / f"task{task_id:02d}_lora.pth"
        torch.save(lora_state, lora_path)
        print(f"✔ Saved LoRA adapter -> {lora_path}")

        # Optional: Fisher diagonal for this task (LoRA params only)
        fisher_path = save_dir / "adapters" / f"task{task_id:02d}_fisher.pth"
        var_path = save_dir / "adapters" / f"task{task_id:02d}_var.pth"

        if not args.no_fisher or not args.no_laplace:
            # Build a small loader for estimation (same dataset; fewer batches)
            fisher_loader = DataLoader(
                train_ds, batch_size=min(batch_size, 32), shuffle=True, num_workers=2, pin_memory=True
            )
            loss_fn = nn.CrossEntropyLoss(reduction="mean")

        if not args.no_fisher:
            print("  Estimating diagonal Fisher (LoRA params only)...")
            fisher = estimate_fisher_diagonal(
                model, fisher_loader, loss_fn, device=device, max_batches=int(args.fisher_max_batches)
            )
            # Move to CPU & save
            fisher_cpu = {k: v.detach().cpu() for k, v in fisher.items()}
            torch.save(fisher_cpu, fisher_path)
            print(f"  ✔ Saved Fisher -> {fisher_path}")

        if not args.no_laplace:
            # If Fisher not computed in this run, try to load it back (supports --no-fisher)
            if args.no_fisher:
                if fisher_path.exists():
                    fisher_cpu = torch.load(fisher_path, map_location="cpu")
                else:
                    print("  [WARN] Fisher not available; computing once for Laplace variances.")
                    fisher = estimate_fisher_diagonal(
                        model, fisher_loader, loss_fn, device=device, max_batches=int(args.fisher_max_batches)
                    )
                    fisher_cpu = {k: v.detach().cpu() for k, v in fisher.items()}
                    torch.save(fisher_cpu, fisher_path)
                    print(f"  ✔ Saved Fisher -> {fisher_path}")
            # Laplace variances Σ ≈ (F + λI)^(-1)
            variances = laplace_variance_from_fisher(fisher_cpu, damping=1e-3)
            variances_cpu = {k: v.detach().cpu() for k, v in variances.items()}
            torch.save(variances_cpu, var_path)
            print(f"  ✔ Saved Laplace variances -> {var_path}")

        # Log per-task stats
        c = count_lora_params(model)
        stats = TrainStats(
            task_id=task_id,
            classes=list(classes),
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            final_loss=final_loss,
            final_acc=final_acc,
            lora_params=c.lora_trainable,
            total_trainable=c.total_trainable,
            base_frozen=c.base_frozen,
        )
        with open(save_dir / f"log_task{task_id:02d}.json", "w") as f:
            json.dump(asdict(stats), f, indent=2)
        per_task_logs.append(asdict(stats))

    # Overall log
    with open(save_dir / "log_all_tasks.json", "w") as f:
        json.dump(per_task_logs, f, indent=2)

    print("\nAll tasks completed.")
    print("Next:")
    print("  • Merge adapters:   python scripts/merge_models.py --cfg config/experiment_config.yaml --method ties")
    print("  • Evaluate results: python scripts/evaluate.py --cfg config/experiment_config.yaml")


if __name__ == "__main__":
    main()

