"""Label helpers for the Sign Language MNIST dataset."""

from __future__ import annotations

from typing import Iterable

# Common raw-label mapping used by the Kaggle Sign Language MNIST dataset.
# J and Z are omitted because they require motion and are not part of the static set.
RAW_LABEL_TO_LETTER = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
}


def raw_label_to_display(raw_label: int) -> str:
    """Map a raw dataset label to a human-readable class name."""
    return RAW_LABEL_TO_LETTER.get(raw_label, f"class_{raw_label}")


def build_display_labels(raw_labels: Iterable[int]) -> list[str]:
    """Build display labels for a sorted iterable of raw labels."""
    return [raw_label_to_display(int(label)) for label in raw_labels]
