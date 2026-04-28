#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the Sign Language MNIST test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report
from torch import nn

from gesture_lenet.data import SignLanguageCSVDataset, load_sign_mnist_csv
from gesture_lenet.reporting import save_per_class_metrics_csv, summarize_confusions
from gesture_lenet.utils import (
    evaluate_model,
    load_model_from_checkpoint,
    plot_confusion_matrix,
    save_json,
    select_device,
)
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/train/best_model.pth")
    parser.add_argument("--test-csv", default="data/raw/sign_mnist_test.csv")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    raw_labels, images = load_sign_mnist_csv(args.test_csv)
    raw_label_to_index = {
        int(raw_label): index for index, raw_label in enumerate(checkpoint["raw_labels"])
    }
    dataset = SignLanguageCSVDataset(images, raw_labels, raw_label_to_index)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, dataloader, criterion, device)
    class_names = [str(name) for name in checkpoint["class_names"]]

    report = classification_report(
        metrics["targets"],
        metrics["predictions"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    top_confusions = summarize_confusions(
        targets=metrics["targets"],
        predictions=metrics["predictions"],
        class_names=class_names,
    )

    save_json(
        {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "classification_report": report,
            "top_confusions": top_confusions,
        },
        output_dir / "metrics.json",
    )
    save_json({"pairs": top_confusions}, output_dir / "confusion_pairs.json")
    save_per_class_metrics_csv(
        {"classification_report": report},
        output_dir / "per_class_metrics.csv",
    )
    plot_confusion_matrix(
        targets=metrics["targets"],
        predictions=metrics["predictions"],
        class_names=class_names,
        output_path=output_dir / "confusion_matrix.png",
    )

    print(f"Evaluation complete. accuracy={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
