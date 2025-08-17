#!/usr/bin/env python3
# scripts/merge_model.py
# -*- coding: utf-8 -*-
"""
Merge per-task LoRA adapters into a single adapter file.

Supported methods:
  - simple   : element-wise mean of LoRA parameters
  - fisher   : element-wise Fisher-weighted mean (requires taskXX_fisher.pth)
  - ties     : TIES-Merging (trim -> elect sign -> disjoint merge)
  - bayesian : sample adapters from Laplace diag N(mu, sigma^2), precision-weighted merge, average over S samples
               (requires taskXX_var.pth; optionally fisher for diagnostics)

Inputs (expected):
  results/adapters/task01_lora.pth
  results/adapters/task02_lora.pth
  ...
  [fisher only]  results/adapters/task01_fisher.pth ...
  [bayesian only]results/adapters/task01_var.pth    ...

Outputs:
  results/merged_<method>.pth   # LoRA-only state_dict

Examples:
  python scripts/merge_model.py --cfg config/experiment_config.yaml --method ties
  python scripts/merge_model.py --cfg config/experiment_config.yaml --method fisher
  python scripts/merge_model.py --cfg config/experiment_config.yaml --method bayesian --samples 10

Notes:
  - This script merges ONLY LoRA parameters (keys matching '*lora_A*' or '*lora_B*').
  - You don't need to load the backbone to merge; evaluate.py will apply the merged LoRA onto the model at runtime.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Local utils
from utils import load_cfg


# ---------------------------
# Helpers
# ---------------------------

def _load_per_task_states(adapter_dir: Path, pattern: str) -> List[Dict[str, torch.Tensor]]:
    paths = sorted(glob.glob(str(adapter_dir / pattern)))
    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {adapter_dir / pattern}")
    states = [torch.load(p, map_location="cpu") for p in paths]
    return states


def _sorted_adapter_paths(adapter_dir: Path) -> List[str]:
    paths = sorted(glob.glob(str(adapter_dir / "task*_lora.pth")))
    if not paths:
        raise FileNotFoundError(f"No LoRA adapters found in {adapter_dir}")
    return paths


def _intersect_keys(states: List[Dict[str, torch.Tensor]]) -> List[str]:
    key_sets = [set(s.keys()) for s in states]
    common = set.intersection(*key_sets)
    if not common:
        raise RuntimeError("No common LoRA keys across tasks; cannot merge.")
    # Keep a stable order (sorted)
    return sorted(common)


def _assert_same_shapes(states: List[Dict[str, torch.Tensor]], keys: List[str]) -> None:
    ref = {k: states[0][k].shape for k in keys}
    for s in states[1:]:
        for k in keys:
            if s[k].shape != ref[k]:
                raise ValueError(f"Shape mismatch for key '{k}': {s[k].shape} vs {ref[k]}")


# ---------------------------
# Merging methods
# ---------------------------

@torch.no_grad()
def merge_simple(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = _intersect_keys(states)
    _assert_same_shapes(states, keys)
    merged = {}
    for k in keys:
        stack = torch.stack([s[k] for s in states], dim=0)
        merged[k] = stack.mean(dim=0)
    return merged


@torch.no_grad()
def merge_fisher(states: List[Dict[str, torch.Tensor]], fishers: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(states) != len(fishers):
        raise ValueError("Number of LoRA states and Fisher states must match.")
    keys = _intersect_keys(states)
    _assert_same_shapes(states, keys)
    # Ensure fishers have same keys
    fisher_keys = _intersect_keys(fishers)
    if set(keys) - set(fisher_keys):
        missing = sorted(set(keys) - set(fisher_keys))
        raise ValueError(f"Fisher dicts missing keys for LoRA params: {missing}")

    merged = {}
    eps = 1e-12
    for k in keys:
        # Stack fishers and normalize along task dimension
        F_stack = torch.stack([f[k] for f in fishers], dim=0)  # (T, ...)
        F_norm = F_stack + eps
        F_norm = F_norm / F_norm.sum(dim=0, keepdim=True)

        S_stack = torch.stack([s[k] for s in states], dim=0)  # (T, ...)
        merged[k] = (F_norm * S_stack).sum(dim=0)
    return merged


@torch.no_grad()
def merge_ties(states: List[Dict[str, torch.Tensor]], trim_percentile: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    TIES-Merging: Trim -> Elect Sign -> Disjoint Merge
    - Trim: zero out small magnitudes (per-key threshold from global stack)
    - Elect: majority sign (ties → 0; fall back to sign of max-abs contributor)
    - Disjoint: sum values that match elected sign, average by count

    Returns merged LoRA dict.
    """
    assert 0.0 <= trim_percentile < 1.0, "trim_percentile must be in [0,1)"
    keys = _intersect_keys(states)
    _assert_same_shapes(states, keys)
    T = len(states)

    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        stack = torch.stack([s[k] for s in states], dim=0)  # (T, ...)
        # Trim
        thresh = torch.quantile(stack.abs().reshape(T, -1), trim_percentile, dim=0).view_as(stack[0])
        trimmed = torch.where(stack.abs() < thresh, torch.zeros_like(stack), stack)

        # Elect sign
        sign_votes = torch.sign(trimmed)          # {-1, 0, +1}
        vote_sum = sign_votes.sum(dim=0)          # element-wise sum
        elected = torch.sign(vote_sum)            # majority sign; 0 if tie

        # If tie (0), use sign of argmax |value| across tasks
        ties_mask = (elected == 0)
        if ties_mask.any():
            max_abs_idx = trimmed.abs().argmax(dim=0)  # indices of max |val| along task dim
            elected = torch.where(
                ties_mask,
                torch.gather(torch.sign(trimmed), dim=0, index=max_abs_idx.unsqueeze(0)).squeeze(0),
                elected,
            )

        # Disjoint merge: keep only values with same sign as elected
        same_sign_mask = (torch.sign(trimmed) == elected.unsqueeze(0)).float()
        num = same_sign_mask.sum(dim=0).clamp_min(1.0)
        merged_val = (trimmed * same_sign_mask).sum(dim=0) / num
        merged[k] = torch.nan_to_num(merged_val, nan=0.0)
    return merged


@torch.no_grad()
def merge_bayesian(
    states: List[Dict[str, torch.Tensor]],
    variances: List[Dict[str, torch.Tensor]],
    samples: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Bayesian LoRA merge:
      For s in 1..S:
        - Sample A_i^{(s)} ~ N(mu_i, sigma_i^2) for each task i and param key
        - Precision-weighted average across tasks:  sum( (1/σ_i^2) * A_i^{(s)} ) / sum(1/σ_i^2)
      Final = average over s.

    Returns merged LoRA dict.
    """
    if len(states) != len(variances):
        raise ValueError("Number of LoRA states and variance dicts must match for Bayesian merge.")

    keys = _intersect_keys(states)
    _assert_same_shapes(states, keys)
    var_keys = _intersect_keys(variances)
    if set(keys) - set(var_keys):
        missing = sorted(set(keys) - set(var_keys))
        raise ValueError(f"Variance dicts missing keys for LoRA params: {missing}")

    T = len(states)
    eps = 1e-12
    totals = {k: torch.zeros_like(states[0][k]) for k in keys}

    for s in range(samples):
        # Sample adapters for each task
        sampled = []
        for t in range(T):
            sample_t = {}
            for k in keys:
                mu = states[t][k]
                var = variances[t][k]
                std = (var + eps).sqrt()
                sample_t[k] = mu + torch.randn_like(mu) * std
            sampled.append(sample_t)

        # Precision-weighted merge for this sample
        for k in keys:
            prec = torch.stack([1.0 / (variances[t][k] + eps) for t in range(T)], dim=0)  # (T, ...)
            W = prec / prec.sum(dim=0, keepdim=True)
            S_stack = torch.stack([sampled[t][k] for t in range(T)], dim=0)
            merged_k = (W * S_stack).sum(dim=0)
            totals[k] += merged_k

    # Average over samples
    merged = {k: totals[k] / float(samples) for k in keys}
    return merged


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge task-specific LoRA adapters into one.")
    ap.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    ap.add_argument(
        "--method",
        type=str,
        default="ties",
        choices=["simple", "fisher", "ties", "bayesian"],
        help="Merging strategy (default: ties)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save merged adapter (defaults to results/merged_<method>.pth)",
    )
    ap.add_argument("--trim-percentile", type=float, default=None, help="TIES trim percentile (default from YAML or 0.2)")
    ap.add_argument("--samples", type=int, default=None, help="Bayesian: #samples S (default from YAML or 10)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    save_root = Path(cfg["evaluation"]["save_dir"])
    adapter_dir = save_root / "adapters"

    # Load LoRA states
    adapter_paths = _sorted_adapter_paths(adapter_dir)
    states = [torch.load(p, map_location="cpu") for p in adapter_paths]
    print(f"Found {len(states)} task adapters:")
    for p in adapter_paths:
        print(f"  - {p}")

    # Merge
    method = args.method.lower()
    if method == "simple":
        merged = merge_simple(states)
        out_path = args.output or (save_root / "merged_simple.pth")

    elif method == "fisher":
        fishers = _load_per_task_states(adapter_dir, "task*_fisher.pth")
        merged = merge_fisher(states, fishers)
        out_path = args.output or (save_root / "merged_fisher.pth")

    elif method == "ties":
        trim = args.trim_percentile
        if trim is None:
            trim = float(cfg.get("merging", {}).get("ties", {}).get("trim_percentile", 0.2))
        merged = merge_ties(states, trim_percentile=trim)
        out_path = args.output or (save_root / "merged_ties.pth")

    elif method == "bayesian":
        vars_ = _load_per_task_states(adapter_dir, "task*_var.pth")
        S = args.samples if args.samples is not None else int(cfg.get("uncertainty", {}).get("laplace", {}).get("samples", 10))
        merged = merge_bayesian(states, vars_, samples=S)
        out_path = args.output or (save_root / "merged_bayesian.pth")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Save merged adapter
    torch.save(merged, out_path)
    print(f"\n✔ Saved merged adapter -> {out_path}")


if __name__ == "__main__":
    main()
