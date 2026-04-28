"""Synthesize a cold-start keypoint dataset from hand-crafted templates.

This is a fallback for users who do not want to record their own samples. The
resulting model will land around 75–85% accuracy in real conditions — enough
to run the demo, but real recordings via record.html are recommended.

Each gesture has a base template (21 landmarks in normalized image space).
We apply random rotation, translation, scale, mirror and per-landmark noise
to produce N samples per class.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from . import GESTURE_LABELS

# Coordinate system: x to the right, y down (MediaPipe convention).
# Each template assumes a right hand held upright with fingers up.

WRIST = (0.50, 0.90, 0.00)


def _finger_chain(
    base: tuple[float, float, float],
    direction: tuple[float, float, float],
    segments: int = 4,
    length: float = 0.10,
    curl: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Place a finger as a poly-line of segments + 1 joints.

    curl bends each subsequent joint by `curl` radians toward the palm (down).
    """
    points = [base]
    px, py, pz = base
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    for _ in range(segments):
        px += dx * length
        py += dy * length
        pz += dz * length
        points.append((px, py, pz))
        # rotate direction in xy plane to simulate curl toward palm (positive y)
        cos_c = math.cos(curl)
        sin_c = math.sin(curl)
        dx_new = dx * cos_c - dy * sin_c
        dy_new = dx * sin_c + dy * cos_c
        dx, dy = dx_new, dy_new
    return points


def _palm_anchors() -> dict[int, tuple[float, float, float]]:
    """The non-finger anchor joints (wrist + finger MCPs)."""
    return {
        0: WRIST,
        1: (0.43, 0.86, 0.00),  # thumb cmc
        5: (0.45, 0.70, 0.00),  # index mcp
        9: (0.50, 0.68, 0.00),  # middle mcp
        13: (0.55, 0.70, 0.00),  # ring mcp
        17: (0.60, 0.74, 0.00),  # pinky mcp
    }


def _build_open_palm() -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float] | None] = [None] * 21
    palm = _palm_anchors()
    for idx, value in palm.items():
        points[idx] = value
    # thumb (1..4)
    thumb = _finger_chain(palm[1], (-0.45, -0.55, 0.0), segments=3, length=0.07)
    for i, p in enumerate(thumb[1:], start=2):
        points[i] = p
    # index, middle, ring, pinky: extended upward
    finger_specs = [
        (5, palm[5], (-0.05, -1.0, 0.0)),
        (9, palm[9], (0.0, -1.0, 0.0)),
        (13, palm[13], (0.05, -1.0, 0.0)),
        (17, palm[17], (0.10, -0.95, 0.0)),
    ]
    for mcp_idx, mcp_pos, direction in finger_specs:
        chain = _finger_chain(mcp_pos, direction, segments=3, length=0.085)
        for offset, p in enumerate(chain[1:], start=1):
            points[mcp_idx + offset] = p
    assert all(p is not None for p in points)
    return list(points)  # type: ignore[return-value]


def _curl_finger(
    points: list[tuple[float, float, float]],
    mcp_idx: int,
    extended: bool = False,
) -> None:
    """Replace a finger's three distal joints with either extended or curled positions."""
    palm = _palm_anchors()
    base = palm.get(mcp_idx, points[mcp_idx])
    if extended:
        chain = _finger_chain(base, (0.0, -1.0, 0.0), segments=3, length=0.085)
    else:
        # curl tips back toward palm (down)
        chain = _finger_chain(base, (0.0, -0.4, 0.0), segments=3, length=0.06, curl=0.55)
    for offset, p in enumerate(chain[1:], start=1):
        points[mcp_idx + offset] = p


def _curl_thumb(points: list[tuple[float, float, float]], style: str) -> None:
    palm = _palm_anchors()
    base = palm[1]
    if style == "out":
        chain = _finger_chain(base, (-0.5, -0.5, 0.0), segments=3, length=0.07)
    elif style == "up":
        chain = _finger_chain(base, (-0.05, -1.0, 0.0), segments=3, length=0.075)
    elif style == "down":
        chain = _finger_chain(base, (-0.05, 1.0, 0.0), segments=3, length=0.075)
    elif style == "in":
        chain = _finger_chain(base, (0.4, -0.2, 0.0), segments=3, length=0.05)
    elif style == "pinch":
        # tip near index tip
        chain = _finger_chain(base, (-0.1, -0.55, 0.0), segments=3, length=0.07)
    else:
        chain = _finger_chain(base, (-0.5, -0.5, 0.0), segments=3, length=0.07)
    for offset, p in enumerate(chain[1:], start=2):
        points[offset] = p


def _build_template(label: str) -> list[tuple[float, float, float]]:
    points = _build_open_palm()
    if label == "open_palm":
        return points
    if label == "fist":
        _curl_thumb(points, "in")
        for mcp in (5, 9, 13, 17):
            _curl_finger(points, mcp, extended=False)
        return points
    if label == "point":
        _curl_thumb(points, "in")
        for mcp in (9, 13, 17):
            _curl_finger(points, mcp, extended=False)
        return points
    if label == "victory":
        _curl_thumb(points, "in")
        _curl_finger(points, 13, extended=False)
        _curl_finger(points, 17, extended=False)
        return points
    if label == "three":
        _curl_thumb(points, "in")
        _curl_finger(points, 17, extended=False)
        return points
    if label == "thumbs_up":
        _curl_thumb(points, "up")
        for mcp in (5, 9, 13, 17):
            _curl_finger(points, mcp, extended=False)
        return points
    if label == "thumbs_down":
        _curl_thumb(points, "down")
        for mcp in (5, 9, 13, 17):
            _curl_finger(points, mcp, extended=False)
        # flip vertically so hand points down
        flipped = []
        for x, y, z in points:
            flipped.append((x, 1.0 - y + 0.6, z))
        return flipped
    if label == "call":
        # thumb out, pinky out, middle three curled
        _curl_thumb(points, "out")
        _curl_finger(points, 5, extended=False)
        _curl_finger(points, 9, extended=False)
        _curl_finger(points, 13, extended=False)
        # pinky extended
        chain = _finger_chain(_palm_anchors()[17], (0.18, -0.95, 0.0), segments=3, length=0.085)
        for offset, p in enumerate(chain[1:], start=1):
            points[17 + offset] = p
        return points
    if label == "ok":
        # thumb tip touches index tip; middle/ring/pinky extended
        _curl_thumb(points, "pinch")
        # index curls so its tip meets thumb tip
        index_chain = _finger_chain(
            _palm_anchors()[5], (-0.15, -0.65, 0.0), segments=3, length=0.07, curl=0.4
        )
        for offset, p in enumerate(index_chain[1:], start=1):
            points[5 + offset] = p
        for mcp in (9, 13, 17):
            _curl_finger(points, mcp, extended=True)
        return points
    if label == "pinch":
        # thumb tip and index tip almost touching, others curled
        _curl_thumb(points, "pinch")
        index_chain = _finger_chain(
            _palm_anchors()[5], (-0.15, -0.65, 0.0), segments=3, length=0.07, curl=0.4
        )
        for offset, p in enumerate(index_chain[1:], start=1):
            points[5 + offset] = p
        for mcp in (9, 13, 17):
            _curl_finger(points, mcp, extended=False)
        return points
    raise ValueError(f"unknown gesture label: {label}")


def synthesize(
    output_dir: Path,
    *,
    samples_per_class: int = 250,
    seed: int = 42,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for label in GESTURE_LABELS:
        template = _build_template(label)
        path = output_dir / f"{label}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for _ in range(samples_per_class):
                # we mirror inside _augment; track separately so handedness label matches
                mirror_flag = rng.random() < 0.5
                # re-seeded augment that knows whether mirror happened
                landmarks = _augment_with_flag(template, rng, mirror=mirror_flag)
                record = {
                    "landmarks": landmarks,
                    "handedness": "Left" if mirror_flag else "Right",
                    "ts": 0,
                    "synthetic": True,
                }
                fh.write(json.dumps(record) + "\n")
        counts[label] = samples_per_class
    return counts


def _augment_with_flag(
    template: list[tuple[float, float, float]],
    rng: random.Random,
    *,
    mirror: bool,
    angle_std: float = 0.18,
    scale_std: float = 0.10,
    translation_std: float = 0.05,
    landmark_jitter: float = 0.012,
) -> list[list[float]]:
    angle = rng.gauss(0.0, angle_std)
    scale = max(0.4, 1.0 + rng.gauss(0.0, scale_std))
    tx = rng.gauss(0.0, translation_std)
    ty = rng.gauss(0.0, translation_std)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    out = []
    cx, cy = 0.5, 0.7
    for x, y, z in template:
        if mirror:
            x = 1.0 - x
        ox = (x - cx) * scale
        oy = (y - cy) * scale
        rx = ox * cos_a - oy * sin_a
        ry = ox * sin_a + oy * cos_a
        nx = cx + rx + tx + rng.gauss(0.0, landmark_jitter)
        ny = cy + ry + ty + rng.gauss(0.0, landmark_jitter)
        nz = z * scale + rng.gauss(0.0, landmark_jitter * 0.5)
        out.append([nx, ny, nz])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "gesture_keypoints_seed"),
    )
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    counts = synthesize(
        output_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(f"wrote synthetic dataset to {output_dir}")
    for label, n in counts.items():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
