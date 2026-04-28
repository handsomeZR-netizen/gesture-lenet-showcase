"""Dataset loading utilities for Sign Language MNIST CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from gesture_lenet.labels import build_display_labels


EXPECTED_PIXEL_COLUMNS = 28 * 28


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata required to reconstruct label mappings and model heads."""

    raw_labels: list[int]
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.raw_labels)

    @property
    def raw_label_to_index(self) -> dict[int, int]:
        return {raw_label: index for index, raw_label in enumerate(self.raw_labels)}

    @property
    def index_to_raw_label(self) -> dict[int, int]:
        return {index: raw_label for index, raw_label in enumerate(self.raw_labels)}


class SignLanguageCSVDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset backed by CSV-extracted numpy arrays."""

    def __init__(
        self,
        images: np.ndarray,
        raw_labels: np.ndarray,
        raw_label_to_index: dict[int, int],
        augment: bool = False,
    ) -> None:
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError("Expected images with shape (N, 28, 28).")

        dense_labels = [raw_label_to_index[int(raw_label)] for raw_label in raw_labels]

        self.images = torch.from_numpy(images).unsqueeze(1).float() / 255.0
        self.labels = torch.tensor(dense_labels, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.augment:
            image = augment_sign_image(image)
        return image, self.labels[index]


def augment_sign_image(image: torch.Tensor) -> torch.Tensor:
    """Apply small CPU-friendly perturbations to a normalized 1x28x28 image."""
    augmented = image.clone()

    if torch.rand(()) < 0.75:
        shift_y = int(torch.randint(-2, 3, ()).item())
        shift_x = int(torch.randint(-2, 3, ()).item())
        augmented = torch.roll(augmented, shifts=(shift_y, shift_x), dims=(1, 2))
        if shift_y > 0:
            augmented[:, :shift_y, :] = 0
        elif shift_y < 0:
            augmented[:, shift_y:, :] = 0
        if shift_x > 0:
            augmented[:, :, :shift_x] = 0
        elif shift_x < 0:
            augmented[:, :, shift_x:] = 0

    if torch.rand(()) < 0.60:
        scale = 0.88 + float(torch.rand(())) * 0.24
        bias = (float(torch.rand(())) - 0.5) * 0.12
        augmented = augmented * scale + bias

    if torch.rand(()) < 0.35:
        augmented = augmented + torch.randn_like(augmented) * 0.025

    return torch.clamp(augmented, 0.0, 1.0)


def load_sign_mnist_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Sign Language MNIST CSV into raw labels and image arrays."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "label" not in frame.columns:
        raise ValueError(f"CSV is missing the 'label' column: {csv_path}")

    pixel_columns = [column for column in frame.columns if column != "label"]
    if len(pixel_columns) != EXPECTED_PIXEL_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_PIXEL_COLUMNS} pixel columns, got {len(pixel_columns)} in {csv_path}"
        )

    raw_labels = frame["label"].to_numpy(dtype=np.int64)
    images = frame[pixel_columns].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
    return raw_labels, images


def build_metadata(raw_labels: Iterable[int]) -> DatasetMetadata:
    unique_raw_labels = sorted({int(label) for label in raw_labels})
    class_names = build_display_labels(unique_raw_labels)
    return DatasetMetadata(raw_labels=unique_raw_labels, class_names=class_names)


def create_dataloaders(
    train_csv: str | Path,
    test_csv: str | Path,
    batch_size: int,
    val_size: float,
    num_workers: int,
    seed: int,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    """Create train/validation/test dataloaders from CSV files."""
    train_raw_labels, train_images = load_sign_mnist_csv(train_csv)
    metadata = build_metadata(train_raw_labels)

    train_indices, val_indices = train_test_split(
        np.arange(len(train_raw_labels)),
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=train_raw_labels,
    )

    train_dataset = SignLanguageCSVDataset(
        images=train_images[train_indices],
        raw_labels=train_raw_labels[train_indices],
        raw_label_to_index=metadata.raw_label_to_index,
        augment=augment,
    )
    val_dataset = SignLanguageCSVDataset(
        images=train_images[val_indices],
        raw_labels=train_raw_labels[val_indices],
        raw_label_to_index=metadata.raw_label_to_index,
    )

    test_raw_labels, test_images = load_sign_mnist_csv(test_csv)
    unknown_labels = sorted(set(int(label) for label in test_raw_labels) - set(metadata.raw_labels))
    if unknown_labels:
        raise ValueError(
            "Test set contains labels not present in the training set: "
            + ", ".join(str(label) for label in unknown_labels)
        )

    test_dataset = SignLanguageCSVDataset(
        images=test_images,
        raw_labels=test_raw_labels,
        raw_label_to_index=metadata.raw_label_to_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader, metadata
