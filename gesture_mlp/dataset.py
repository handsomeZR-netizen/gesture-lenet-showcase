"""Read JSONL keypoint files into a torch dataset.

Each line in ``data/gesture_keypoints/<label>.jsonl`` is::

    {"landmarks": [[x,y,z], ... 21 entries], "handedness": "Right", "ts": 1700000000}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from . import GESTURE_LABELS, LABEL_TO_INDEX
from .features import FEATURE_DIM, landmarks_to_feature

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gesture_keypoints"


@dataclass
class KeypointSample:
    feature: np.ndarray
    label_index: int


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_samples(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    augment_jitter: float = 0.0,
    rng: random.Random | None = None,
) -> list[KeypointSample]:
    data_dir = Path(data_dir)
    samples: list[KeypointSample] = []
    for label in GESTURE_LABELS:
        path = data_dir / f"{label}.jsonl"
        if not path.exists():
            continue
        label_idx = LABEL_TO_INDEX[label]
        for record in _iter_jsonl(path):
            try:
                landmarks = np.array(record["landmarks"], dtype=np.float32)
                handedness = record.get("handedness", "Right")
                feature = landmarks_to_feature(landmarks, handedness=handedness)
                samples.append(KeypointSample(feature=feature, label_index=label_idx))
            except Exception:
                continue
    if augment_jitter > 0 and samples:
        rng = rng or random.Random(0)
        augmented = []
        for sample in samples:
            for _ in range(2):
                noise = np.array(
                    [rng.gauss(0.0, augment_jitter) for _ in range(FEATURE_DIM)],
                    dtype=np.float32,
                )
                augmented.append(
                    KeypointSample(
                        feature=sample.feature + noise,
                        label_index=sample.label_index,
                    )
                )
        samples.extend(augmented)
    return samples


def split_samples(
    samples: list[KeypointSample],
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[list[KeypointSample], list[KeypointSample]]:
    by_class: dict[int, list[KeypointSample]] = {}
    for sample in samples:
        by_class.setdefault(sample.label_index, []).append(sample)

    rng = random.Random(seed)
    train: list[KeypointSample] = []
    val: list[KeypointSample] = []
    for label_idx, group in by_class.items():
        rng.shuffle(group)
        cut = max(1, int(round(len(group) * val_ratio)))
        val.extend(group[:cut])
        train.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class KeypointDataset(Dataset):
    def __init__(self, samples: list[KeypointSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        feature = torch.from_numpy(sample.feature)
        label = torch.tensor(sample.label_index, dtype=torch.long)
        return feature, label


def class_distribution(samples: Iterable[KeypointSample]) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in GESTURE_LABELS}
    for sample in samples:
        counts[GESTURE_LABELS[sample.label_index]] += 1
    return counts
