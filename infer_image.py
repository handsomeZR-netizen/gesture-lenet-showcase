#!/usr/bin/env python3
"""Run single-image inference with a trained LeNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from gesture_lenet.utils import (
    load_model_from_checkpoint,
    preprocess_grayscale_image,
    save_json,
    select_device,
    topk_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/improved_train/best_model.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", default="outputs/predict/image_prediction.json")
    parser.add_argument("--save-preview", default="outputs/predict/preprocessed_input.png")
    parser.add_argument("--threshold", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    tensor, preview = preprocess_grayscale_image(image=image, threshold=args.threshold)
    predictions = topk_predictions(
        model=model,
        tensor=tensor,
        class_names=list(checkpoint["class_names"]),
        device=device,
    )

    Path(args.save_preview).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.save_preview, preview)
    save_json(
        {
            "image": str(Path(args.image).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "predictions": predictions,
            "threshold": args.threshold,
        },
        args.output_json,
    )

    top1 = predictions[0]
    print(
        f"Top-1 prediction: {top1['label']} "
        f"(index={top1['index']}, confidence={top1['confidence']:.4f})"
    )


if __name__ == "__main__":
    main()
