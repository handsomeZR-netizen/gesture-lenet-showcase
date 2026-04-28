"""Tiny 3-layer MLP for static gesture classification."""

from __future__ import annotations

import torch
from torch import nn

from . import FEATURE_DIM, NUM_CLASSES


class GestureMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
