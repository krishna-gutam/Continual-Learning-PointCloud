# models/pointnet_baseline.py
# -*- coding: utf-8 -*-
"""
PointNet baseline classifier (minimal, runnable out-of-the-box).

- Input:  B x N x 3  point clouds (float32)
- Output: B x C      class logits

Components
----------
TNet(k=3):
    Learns an affine 3x3 transform on the input points (as in PointNet),
    improving invariance to rotations/perturbations.
PointNetClassifier:
    Shared MLP over points via 1x1 convs: 3->64->128->1024
    Global max pool over N points -> 1024-d global feature
    MLP head: 1024->512->256->num_classes with dropout

Notes
-----
- Kept intentionally compact and dependency-light for smoke tests.
- Works with your LoRA injection (targets nn.Linear in the MLP head).
- If you later swap to Point Transformer, keep the forward API the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Input transform network (k x k), implemented with 1D convolutions + MLP.

    Args:
        k (int): dimension of the input features per point (3 for xyz)
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        # Shared MLP over points (via 1x1 conv)
        self.conv1 = nn.Conv1d(k, 64, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=True)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # MLP on pooled feature
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, k * k, bias=True)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize the last layer to predict (near) zero so that
        # the transform starts close to identity.
        nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, k, N) point/features tensor
        Returns:
            transform: (B, k, k) learned transform matrix
        """
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))        # (B, 1024, N)
        x = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)                    # (B, k*k)

        # Add identity bias to help stability (classic PointNet trick)
        ident = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k)
        x = x + ident
        return x.view(-1, self.k, self.k)


class PointNetClassifier(nn.Module):
    """
    Minimal PointNet classifier.

    Args:
        num_classes (int): number of output classes
        dropout_p (float): dropout probability in the MLP head
        use_input_transform (bool): enable the input T-Net (recommended True)
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_p: float = 0.3,
        use_input_transform: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_input_transform = use_input_transform

        if self.use_input_transform:
            self.input_tnet = TNet(k=3)

        # Shared MLP on per-point features (via 1x1 conv)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=True)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Global feature -> classifier head (nn.Linear so LoRA can hook)
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, num_classes, bias=True)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(dropout_p)
        self.dp2 = nn.Dropout(dropout_p)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: (B, N, 3) float32 tensor of points
        Returns:
            logits: (B, num_classes)
        """
        # Convert to (B, 3, N) for Conv1d
        x = pts.transpose(1, 2).contiguous()  # (B, 3, N)

        # Input transform (optional)
        if self.use_input_transform:
            trans = self.input_tnet(x)        # (B, 3, 3)
            x = torch.bmm(trans, x)           # (B, 3, N)

        # Shared MLP (per-point)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = self.bn3(self.conv3(x))           # (B, 1024, N)

        # Symmetric function: global max pool
        x = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)

        # MLP head
        x = F.relu(self.bn4(self.fc1(x)))     # (B, 512)
        x = self.dp1(x)
        x = F.relu(self.bn5(self.fc2(x)))     # (B, 256)
        x = self.dp2(x)
        logits = self.fc3(x)                  # (B, C)
        return logits

