# models/lora.py
# -*- coding: utf-8 -*-
"""
Lightweight LoRA utilities for PyTorch models.

Features
--------
- LoRALinear: wraps a frozen nn.Linear with trainable low-rank adapters (A, B)
- add_lora_to_linear(model, r, alpha, dropout): recursively inject LoRA
- mark_only_lora_as_trainable(model): convenience for CL training loops
- extract_lora_state(model) / load_lora_state(model, state): save/load adapters only
- merge_lora_weights(model) / unmerge_lora_weights(model): fuse/unfuse deltaW
- estimate_fisher_diagonal(...): diagonal Fisher for LoRA params
- laplace_variance_from_fisher(...): variance estimate Σ ≈ (F + λI)^(-1)

Notes
-----
- By default we do NOT merge LoRA into base weights; forward() adds the low-rank delta on-the-fly.
  For inference speed, you can call merge_lora_weights(model) once.
- Designed to target Linear layers (LoRA on attention/MLP is most common); extend to conv if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Core module: LoRA for Linear layers
# ---------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    LoRA adapter for a frozen nn.Linear.

    W_eff = W + scaling * (B @ A)   where A in R^{r x in}, B in R^{out x r}

    Args:
        base (nn.Linear): The base (frozen) Linear layer to augment.
        r (int): Rank of the adapter (r << min(in, out)). Use 0 to disable.
        alpha (int): LoRA alpha; effective scaling = alpha / r (if r > 0).
        dropout (float): Dropout on the input to LoRA branch.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear as base.")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 1.0
        self.merged = False  # whether deltaW has been merged into base weight

        # Keep the original base layer's params as the main weight/bias
        # (These are the ones used by nn.functional.linear)
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)

        # Trainable LoRA factors
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros((self.r, self.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r)))
            # He/Kaiming init for A; zeros for B is common for stability
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            # still register buffers for API symmetry
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # Retain dtype/device of base
        self.to(dtype=base.weight.dtype, device=base.weight.device)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}")

    @torch.no_grad()
    def _delta_weight(self) -> Optional[torch.Tensor]:
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return None
        # (out, r) @ (r, in) -> (out, in)
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """
        Permanently add LoRA deltaW into base weight for faster inference.
        Call unmerge() to revert.
        """
        if self.merged or self.r <= 0:
            self.merged = True
            return
        delta = self._delta_weight()
        if delta is not None:
            self.weight.add_(delta)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """
        Revert merge(): subtract deltaW from base weight.
        """
        if not self.merged or self.r <= 0:
            self.merged = False
            return
        delta = self._delta_weight()
        if delta is not None:
            self.weight.sub_(delta)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard linear forward + (optional) low-rank delta.

        Shapes:
            x: (B, *, in_features)
            out: (B, *, out_features)
        """
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0 and not self.merged:
            # Efficient: (x @ A^T) @ B^T
            lora_out = self.dropout(x) @ self.lora_A.t()   # (B, *, r)
            lora_out = lora_out @ self.lora_B.t()          # (B, *, out)
            out = out + self.scaling * lora_out
        return out


# ---------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------

def add_lora_to_linear(
    module: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_module_names: Optional[Iterable[str]] = None,
) -> None:
    """
    Recursively replace nn.Linear children with LoRALinear.

    Args:
        module: root module to modify in-place.
        r, alpha, dropout: LoRA hyperparameters.
        target_module_names: optional set/list of names to target; if None, all Linear.
                             Names are matched against immediate child attribute names.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if (target_module_names is None) or (name in set(target_module_names)):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            add_lora_to_linear(child, r=r, alpha=alpha, dropout=dropout, target_module_names=target_module_names)


def mark_only_lora_as_trainable(model: nn.Module, train_classifier: bool = False) -> None:
    """
    Set requires_grad=True only for LoRA params (and optionally final classifier).
    Assumes the rest of the model is frozen (good for CL).

    Args:
        train_classifier: If True, keep classifier (usually 'fc*' or 'classifier') trainable too.
    """
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad = True
        elif train_classifier and (".fc3." in n or n.endswith(".fc3.weight") or n.endswith(".fc3.bias") or "classifier" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False


def extract_lora_state(model: nn.Module, to_cpu: bool = False) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict containing ONLY LoRA parameters (A/B) from the model.

    Use this to save per-task adapters:
        torch.save(extract_lora_state(model), "task01_lora.pth")
    """
    state = {}
    for k, v in model.state_dict().items():
        if ("lora_A" in k) or ("lora_B" in k):
            state[k] = v.cpu() if to_cpu else v.clone()
    return state


@torch.no_grad()
def load_lora_state(model: nn.Module, lora_state: Dict[str, torch.Tensor], strict: bool = False) -> None:
    """
    Load a LoRA-only state dict into the corresponding modules of `model`.

    Args:
        lora_state: mapping from param name to tensor (must match model keys).
        strict: if True, raises if a key is missing or unexpected.
    """
    missing = []
    for k, v in lora_state.items():
        if k in dict(model.named_parameters()) or k in dict(model.named_buffers()):
            dst = model
            # set via state_dict update
        else:
            if strict:
                missing.append(k)
    sd = model.state_dict()
    sd.update(lora_state)
    model.load_state_dict(sd)
    if strict and len(missing) > 0:
        raise KeyError(f"LoRA keys not found in model: {missing}")


def merge_lora_weights(model: nn.Module) -> None:
    """
    Merge all LoRALinear deltas into base weights for inference speed.
    """
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    """
    Unmerge previously merged LoRA deltas (restores original base weights).
    """
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


# ---------------------------------------------------------------------
# Fisher & Laplace helpers (optional; for merging & Bayesian UQ)
# ---------------------------------------------------------------------

@torch.no_grad()
def _zero_like_lora(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Build a zero dict with the same shapes as LoRA parameters in `model`.
    """
    z: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        if p.requires_grad and (("lora_A" in n) or ("lora_B" in n)):
            z[n] = torch.zeros_like(p, device=p.device, dtype=p.dtype)
    return z


def estimate_fisher_diagonal(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Fisher Information for *LoRA parameters only*.
    F(θ) ≈ E[(∂log p(y|x)/∂θ)^2], implemented via squared gradients of loss.

    Args:
        model: your model with LoRA injected; ensure only LoRA params require grad
        dataloader: yields (inputs, labels)
        loss_fn: e.g., nn.CrossEntropyLoss(reduction='mean')
        device: torch.device, defaults to cuda if available
        max_batches: if set, limit number of batches for speed

    Returns:
        dict mapping param_name -> diagonal Fisher tensor (same shape as param)
    """
    model.eval()
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(dev)

    # Ensure only LoRA params require grads
    for n, p in model.named_parameters():
        p.grad = None
        p.requires_grad = ("lora_A" in n) or ("lora_B" in n)

    fisher = _zero_like_lora(model)
    count = 0

    for b, (x, y) in enumerate(dataloader):
        x = x.to(dev)
        y = y.to(dev)
        logits = model(x)
        loss = loss_fn(logits, y)

        # backward on LoRA params only
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        loss.backward()

        # accumulate grad^2
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and (("lora_A" in n) or ("lora_B" in n)):
                fisher[n] += (p.grad.detach() ** 2)

        count += 1
        if (max_batches is not None) and (b + 1 >= max_batches):
            break

    # average over batches (simple empirical estimate)
    if count > 0:
        for n in fisher:
            fisher[n] /= float(count)

    return fisher


def laplace_variance_from_fisher(
    fisher: Dict[str, torch.Tensor],
    damping: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Diagonal Laplace approximation: Σ ≈ (F + λI)^(-1)
    Args:
        fisher: dict of diagonal Fisher tensors for LoRA params
        damping: λ (ridge) to stabilize inversion
    Returns:
        dict of variances with same keys/shapes as fisher
    """
    variances: Dict[str, torch.Tensor] = {}
    for k, f in fisher.items():
        variances[k] = 1.0 / (f + damping)
    return variances


# ---------------------------------------------------------------------
# Convenience: count params
# ---------------------------------------------------------------------

@dataclass
class LoRAParamCounts:
    total_trainable: int
    lora_trainable: int
    base_frozen: int


def count_lora_params(model: nn.Module) -> LoRAParamCounts:
    """
    Count trainable LoRA params vs frozen base params for quick sanity checks.
    """
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and (("lora_A" in n) or ("lora_B" in n))
    )
    base_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return LoRAParamCounts(total_trainable, lora_trainable, base_frozen)


# ---------------------------------------------------------------------
# End
# ---------------------------------------------------------------------
