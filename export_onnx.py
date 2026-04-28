#!/usr/bin/env python3
"""Export the trained gesture MLP to ONNX for in-browser inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gesture_mlp import GESTURE_LABELS, FEATURE_DIM
from gesture_mlp.model import GestureMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="default",
        help="模型名称；不指定 --checkpoint/--output 时会从 outputs/gesture_mlp/<name>/best.pth 读，导出到 web_control_demo/models/<name>/",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument(
        "--display-name",
        default=None,
        help="可选的人类可读名（写入 meta.json），不指定就用 --name",
    )
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name
    checkpoint_path = Path(
        args.checkpoint or f"outputs/gesture_mlp/{name}/best.pth"
    )
    if not checkpoint_path.exists():
        # 兼容老的扁平路径
        legacy = Path("outputs/gesture_mlp/best.pth")
        if name == "default" and legacy.exists():
            checkpoint_path = legacy
        else:
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    labels = (
        payload["labels"]
        if isinstance(payload, dict) and "labels" in payload
        else GESTURE_LABELS
    )

    model = GestureMLP(num_classes=len(labels))
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"web_control_demo/models/{name}/gesture_mlp.onnx")
    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        metadata_path = output_path.parent / "gesture_mlp.meta.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False keeps weights inside the .onnx file (no .onnx.data sidecar),
    # which is what we want for a simple fetch from the browser.
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "name": name,
                "display_name": args.display_name or name,
                "labels": list(labels),
                "feature_dim": FEATURE_DIM,
                "input_name": "features",
                "output_name": "logits",
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"exported ONNX to {output_path}")
    print(f"wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
