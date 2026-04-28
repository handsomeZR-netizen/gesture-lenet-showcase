"""Sanity tests for GestureMLP."""

from __future__ import annotations

import torch

from gesture_mlp import FEATURE_DIM, NUM_CLASSES
from gesture_mlp.model import GestureMLP, parameter_count


def test_forward_shape():
    model = GestureMLP()
    out = model(torch.zeros(4, FEATURE_DIM))
    assert out.shape == (4, NUM_CLASSES)


def test_parameter_count_under_50k():
    """The model must stay tiny so the browser can load it instantly."""
    model = GestureMLP()
    n = parameter_count(model)
    assert 5_000 < n < 50_000


def test_dropout_only_in_train_mode():
    """Dropout should only zero activations in train(), not in eval()."""
    model = GestureMLP(dropout=0.5)
    x = torch.ones(1, FEATURE_DIM)
    model.eval()
    a = model(x)
    b = model(x)
    assert torch.allclose(a, b)
