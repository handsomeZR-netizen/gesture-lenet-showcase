"""Cross-language fixture: Python features must equal JS features within float32 epsilon.

Regenerates the fixture on demand so the JS test (run separately under node)
can read the same numbers we computed here. The actual JS check lives in
tests/test_features_js.mjs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gesture_mlp.features import landmarks_to_feature

FIXTURE_PATH = Path("web_control_demo/modules/features.fixture.json")


def test_fixture_can_be_regenerated(tmp_path):
    rng = np.random.default_rng(7)
    landmarks = (rng.random((21, 3)) * 0.5 + 0.25).astype(np.float32)
    feat_right = landmarks_to_feature(landmarks, "Right")
    feat_left = landmarks_to_feature(landmarks, "Left")
    fixture = {
        "landmarks": landmarks.tolist(),
        "feature_right": feat_right.tolist(),
        "feature_left": feat_left.tolist(),
    }
    out = tmp_path / "fixture.json"
    out.write_text(json.dumps(fixture), encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["landmarks"][0][0] == landmarks[0, 0]


def test_published_fixture_matches_python_if_present():
    """If a fixture exists at the canonical path, Python features must match."""
    if not FIXTURE_PATH.exists():
        return  # fixture is gitignored; skipping is fine
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    landmarks = np.array(fixture["landmarks"], dtype=np.float32)
    feat_right = landmarks_to_feature(landmarks, "Right")
    feat_left = landmarks_to_feature(landmarks, "Left")
    np.testing.assert_allclose(
        feat_right, np.array(fixture["feature_right"], dtype=np.float32), atol=1e-5
    )
    np.testing.assert_allclose(
        feat_left, np.array(fixture["feature_left"], dtype=np.float32), atol=1e-5
    )
