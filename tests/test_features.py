"""Unit tests for the 21-keypoint feature pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_mlp import FEATURE_DIM, NUM_CLASSES
from gesture_mlp.features import (
    NUM_LANDMARKS,
    landmarks_from_mediapipe,
    landmarks_to_feature,
    normalize_landmarks,
)


@pytest.fixture
def random_landmarks():
    rng = np.random.default_rng(0)
    return rng.random((NUM_LANDMARKS, 3)).astype(np.float32)


def test_feature_shape_and_dtype(random_landmarks):
    feat = landmarks_to_feature(random_landmarks)
    assert feat.shape == (FEATURE_DIM,)
    assert feat.dtype == np.float32


def test_feature_translation_invariance(random_landmarks):
    """Adding a constant offset to all landmarks should not change features."""
    feat_a = landmarks_to_feature(random_landmarks)
    shifted = random_landmarks + np.array([0.3, -0.2, 0.05], dtype=np.float32)
    feat_b = landmarks_to_feature(shifted)
    np.testing.assert_allclose(feat_a, feat_b, atol=1e-5)


def test_feature_scale_invariance(random_landmarks):
    """Uniform scaling around the wrist must not change features."""
    wrist = random_landmarks[0:1].copy()
    scaled = (random_landmarks - wrist) * 1.7 + wrist
    feat_a = landmarks_to_feature(random_landmarks)
    feat_b = landmarks_to_feature(scaled.astype(np.float32))
    np.testing.assert_allclose(feat_a, feat_b, atol=1e-4)


def test_left_hand_mirrors_right(random_landmarks):
    """Same physical landmarks classified as left vs right should differ only by x flip."""
    feat_right = landmarks_to_feature(random_landmarks, "Right")
    feat_left = landmarks_to_feature(random_landmarks, "Left")
    feat_left_flipped = feat_left.reshape(NUM_LANDMARKS, 3).copy()
    feat_left_flipped[:, 0] *= -1
    np.testing.assert_allclose(
        feat_right, feat_left_flipped.reshape(-1), atol=1e-6
    )


def test_wrist_is_origin(random_landmarks):
    """After normalization landmark 0 is exactly at the origin."""
    canonical = normalize_landmarks(random_landmarks)
    np.testing.assert_allclose(canonical[0], np.zeros(3), atol=1e-6)


def test_normalize_rejects_bad_shape():
    with pytest.raises(ValueError):
        landmarks_to_feature(np.zeros((20, 3), dtype=np.float32))


def test_landmarks_from_mediapipe_shape():
    class FakeLM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    fakes = [FakeLM(i / 21.0, i / 42.0, i / 100.0) for i in range(NUM_LANDMARKS)]
    arr = landmarks_from_mediapipe(fakes)
    assert arr.shape == (NUM_LANDMARKS, 3)
    assert arr.dtype == np.float32


def test_label_set_is_ten():
    assert NUM_CLASSES == 10
