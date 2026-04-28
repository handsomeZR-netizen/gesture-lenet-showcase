#!/usr/bin/env python3
"""Train a LeNet sign classifier on Sign Language MNIST CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from gesture_lenet.data import create_dataloaders
from gesture_lenet.model import build_model
from gesture_lenet.utils import (
    evaluate_model,
    plot_training_curves,
    save_checkpoint,
    save_json,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/raw/sign_mnist_train.csv")
    parser.add_argument("--test-csv", default="data/raw/sign_mnist_test.csv")
    parser.add_argument("--output-dir", default="outputs/train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--architecture", choices=["lenet", "improved"], default="lenet")
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment,
    )

    model = build_model(architecture=args.architecture, num_classes=metadata.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_accuracy = -1.0
    best_checkpoint = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for inputs, labels in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            running_correct += (predictions == labels).sum().item()
            running_total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_total, 1)
        train_accuracy = running_correct / max(running_total, 1)

        val_metrics = evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                best_checkpoint,
                model,
                metadata={
                    "num_classes": metadata.num_classes,
                    "raw_labels": metadata.raw_labels,
                    "class_names": metadata.class_names,
                    "train_csv": str(Path(args.train_csv).resolve()),
                    "test_csv": str(Path(args.test_csv).resolve()),
                    "architecture": args.architecture,
                    "augment": args.augment,
                },
            )

    plot_training_curves(history, output_dir / "training_curves.png")
    save_json(history, output_dir / "history.json")

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    save_json(
        {
            "best_val_accuracy": best_val_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "num_classes": metadata.num_classes,
            "class_names": metadata.class_names,
            "architecture": args.architecture,
            "augment": args.augment,
        },
        output_dir / "summary.json",
    )

    print(
        f"Training complete. best_val_accuracy={best_val_accuracy:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
