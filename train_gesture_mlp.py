#!/usr/bin/env python3
"""Train the keypoint MLP on collected JSONL data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from gesture_mlp import GESTURE_LABELS, NUM_CLASSES
from gesture_mlp.dataset import (
    DEFAULT_DATA_DIR,
    KeypointDataset,
    class_distribution,
    load_samples,
    split_samples,
)
from gesture_mlp.model import GestureMLP, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="default",
        help="模型名称，会作为 outputs/gesture_mlp/<name>/ 子目录使用，便于训练/对比多个模型",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="覆盖输出目录；不指定时为 outputs/gesture_mlp/<name>/",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--augment-jitter", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss_sum += float(criterion(logits, labels))
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum())
            total += int(labels.size(0))
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    samples = load_samples(args.data_dir, augment_jitter=args.augment_jitter)
    if not samples:
        raise SystemExit(
            f"no samples found under {args.data_dir}. Run record.html or seed_dataset.py first."
        )
    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
    print("class distribution (train):", class_distribution(train_samples))
    print("class distribution (val):  ", class_distribution(val_samples))

    train_loader = DataLoader(
        KeypointDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        KeypointDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = GestureMLP().to(device)
    print(f"model parameters: {parameter_count(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/gesture_mlp") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing model to {output_dir}")
    best_acc = 0.0
    best_path = output_dir / "best.pth"
    history = []

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * labels.size(0)
            running_correct += int((logits.argmax(1) == labels).sum())
            running_total += int(labels.size(0))
        scheduler.step()
        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": GESTURE_LABELS,
                    "num_classes": NUM_CLASSES,
                },
                best_path,
            )
        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:>3}/{args.epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                f"best={best_acc:.4f}"
            )

    elapsed = time.time() - started
    summary = {
        "name": args.name,
        "best_val_acc": best_acc,
        "epochs": args.epochs,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "elapsed_seconds": elapsed,
        "labels": GESTURE_LABELS,
        "data_dir": str(args.data_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        f"done in {elapsed:.1f}s. best val acc {best_acc:.4f}. "
        f"weights saved to {best_path}"
    )


if __name__ == "__main__":
    main()
