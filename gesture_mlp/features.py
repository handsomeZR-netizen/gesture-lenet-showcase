"""Normalize 21 hand landmarks into a 63-d feature vector.

The same transform is implemented in JS at
``web_control_demo/modules/features.js`` — both must agree to within 1e-5,
otherwise the ONNX model trained in Python will see different inputs in the
browser. A small fixture-based cross-language test guards this.

Pipeline:
    1. translate so wrist (landmark 0) is at the origin
    2. divide by palm scale = max(|wrist - middleMcp|, |indexMcp - pinkyMcp|)
    3. mirror left hands across x so all hands look like a right hand
    4. flatten to 63 floats
"""

from __future__ import annotations

import numpy as np

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3
PALM_SCALE_FLOOR = 1e-4


def _palm_scale(points: np.ndarray) -> float:
    span_a = float(np.linalg.norm(points[WRIST, :2] - points[MIDDLE_MCP, :2]))
    span_b = float(np.linalg.norm(points[INDEX_MCP, :2] - points[PINKY_MCP, :2]))
    return max(span_a, span_b, PALM_SCALE_FLOOR)


def normalize_landmarks(
    landmarks: np.ndarray,
    handedness: str = "Right",
) -> np.ndarray:
    """Return a (21, 3) float32 array in the canonical right-hand frame."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {points.shape}")

    centered = points - points[WRIST]
    scale = _palm_scale(points)
    scaled = centered / scale

    if handedness.lower().startswith("l"):
        scaled = scaled.copy()
        scaled[:, 0] = -scaled[:, 0]

    return scaled.astype(np.float32)


def landmarks_to_feature(
    landmarks: np.ndarray,
    handedness: str = "Right",
) -> np.ndarray:
    """Return a (63,) float32 feature vector for the MLP."""
    canonical = normalize_landmarks(landmarks, handedness=handedness)
    return canonical.reshape(-1).astype(np.float32)


def landmarks_from_mediapipe(hand_landmarks: object) -> np.ndarray:
    """Convert a MediaPipe HandLandmarkerResult.hand_landmarks[0] to (21, 3)."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )


def _self_check() -> None:
    rng = np.random.default_rng(0)
    landmarks = rng.random((NUM_LANDMARKS, 3)).astype(np.float32)
    feat = landmarks_to_feature(landmarks)
    assert feat.shape == (FEATURE_DIM,)
    assert feat.dtype == np.float32
    feat_left = landmarks_to_feature(landmarks, handedness="Left")
    flipped = landmarks_to_feature(landmarks)
    flipped = flipped.copy()
    flipped.reshape(NUM_LANDMARKS, 3)[:, 0] *= -1
    np.testing.assert_allclose(feat_left, flipped, atol=1e-6)
    print("features self-check OK", feat.shape)


if __name__ == "__main__":
    _self_check()
