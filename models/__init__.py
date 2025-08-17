# models/__init__.py
# -*- coding: utf-8 -*-
"""
Models package for Continual Learning on Point Clouds.

Exports:
    - PointNetClassifier: Minimal PointNet baseline (runs out-of-the-box)
    - PointTransformerClassifier: Placeholder for Point Transformer backbone
    - LoRA utilities: add_lora_to_linear, extract_lora_state, load_lora_state,
                      merge_lora_weights, unmerge_lora_weights,
                      estimate_fisher_diagonal, laplace_variance_from_fisher
"""

from .pointnet_baseline import PointNetClassifier
from .point_transformer import PointTransformerClassifier
from .lora import (
    add_lora_to_linear,
    mark_only_lora_as_trainable,
    extract_lora_state,
    load_lora_state,
    merge_lora_weights,
    unmerge_lora_weights,
    estimate_fisher_diagonal,
    laplace_variance_from_fisher,
    count_lora_params,
)

__all__ = [
    "PointNetClassifier",
    "PointTransformerClassifier",
    "add_lora_to_linear",
    "mark_only_lora_as_trainable",
    "extract_lora_state",
    "load_lora_state",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "estimate_fisher_diagonal",
    "laplace_variance_from_fisher",
    "count_lora_params",
]
