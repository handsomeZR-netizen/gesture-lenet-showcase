"""Shared helpers for training, evaluation, and inference."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from gesture_lenet.model import build_model

matplotlib.use("Agg")
import cv2
from matplotlib import pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "predictions": predictions,
        "targets": targets,
    }


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)))
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_checkpoint(
    output_path: str | Path,
    model: nn.Module,
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            **metadata,
        },
        output_path,
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    architecture = str(checkpoint.get("architecture", "lenet"))
    model = build_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def preprocess_grayscale_image(
    image: np.ndarray,
    threshold: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    side = min(gray.shape[:2])
    start_y = (gray.shape[0] - side) // 2
    start_x = (gray.shape[1] - side) // 2
    square = gray[start_y : start_y + side, start_x : start_x + side]

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    if threshold:
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        _, resized = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    return tensor, resized


def analyze_hand_presence(image: np.ndarray) -> dict[str, Any]:
    """Heuristically decide whether an ROI likely contains a hand.

    The live classifier is trained on forced 24-way classification, so it will
    otherwise assign a confident label even to background-only frames. This gate
    is intentionally simple and tuned for webcam demo use with a bare hand.
    """

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    roi_area = float(gray.shape[0] * gray.shape[1])

    # Skin-color prior in YCrCb space. This is not universal, but it is a
    # practical way to suppress obvious background frames in the current demo.
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 183, 127], dtype=np.uint8),
    )
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    skin_ratio = float(np.count_nonzero(skin_mask)) / max(roi_area, 1.0)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_skin_area = max((cv2.contourArea(contour) for contour in contours), default=0.0)
    largest_skin_ratio = float(largest_skin_area) / max(roi_area, 1.0)

    center = skin_mask[
        gray.shape[0] // 4 : gray.shape[0] * 3 // 4,
        gray.shape[1] // 4 : gray.shape[1] * 3 // 4,
    ]
    center_skin_ratio = float(np.count_nonzero(center)) / max(center.size, 1)

    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / max(roi_area, 1.0)
    gray_std = float(np.std(blurred))

    is_present = (
        (center_skin_ratio >= 0.05 and largest_skin_ratio >= 0.03)
        or (center_skin_ratio >= 0.08 and edge_ratio >= 0.02 and gray_std >= 18.0)
        or (
            center_skin_ratio >= 0.05
            and skin_ratio >= 0.08
            and edge_ratio >= 0.015
            and gray_std >= 20.0
        )
    )

    return {
        "is_present": bool(is_present),
        "skin_ratio": skin_ratio,
        "largest_skin_ratio": largest_skin_ratio,
        "center_skin_ratio": center_skin_ratio,
        "edge_ratio": edge_ratio,
        "gray_std": gray_std,
        "mask": skin_mask,
    }


def topk_predictions(
    model: nn.Module,
    tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    k: int = 3,
) -> list[dict[str, Any]]:
    tensor = tensor.to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        scores, indices = torch.topk(probabilities, k=min(k, probabilities.shape[1]), dim=1)

    results = []
    for score, index in zip(scores[0].cpu().tolist(), indices[0].cpu().tolist()):
        results.append(
            {
                "index": int(index),
                "label": class_names[int(index)],
                "confidence": float(score),
            }
        )
    return results


def detect_system_camera_nodes() -> list[str]:
    return sorted(str(path) for path in Path("/dev").glob("video*"))


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run_command_capture(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 127, ""
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def fps_tracker(previous_time: float) -> tuple[float, float]:
    current_time = time.time()
    fps = 0.0 if previous_time == 0 else 1.0 / max(current_time - previous_time, 1e-6)
    return current_time, fps
