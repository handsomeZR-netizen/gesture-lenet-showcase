#!/usr/bin/env python3
"""Probe webcam throughput and MediaPipe hand detection readiness."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from gesture_lenet.utils import save_json
from infer_camera import create_hand_landmarker, detect_hand, open_camera, resolve_camera_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=["/dev/video0", "/dev/video1", "0", "1"])
    parser.add_argument("--camera", default="")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.55)
    parser.add_argument("--min-hand-tracking-confidence", type=float, default=0.50)
    parser.add_argument("--mediapipe-sample-rate", type=int, default=5)
    parser.add_argument("--output-json", default="outputs/demo/camera_benchmark.json")
    return parser.parse_args()


def probe_camera(candidate: str, args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate": candidate,
        "resolved_source": str(resolve_camera_source(candidate)),
        "opened": False,
        "frames_read": 0,
        "read_fps": 0.0,
        "shape": None,
        "backend": "unknown",
        "error": "",
    }

    capture = None
    try:
        capture = open_camera(candidate)
        result["opened"] = True
        try:
            result["backend"] = capture.getBackendName()
        except Exception:
            result["backend"] = "unknown"
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)

        started = time.time()
        last_shape = None
        for _ in range(args.frames):
            ok, frame = capture.read()
            if not ok:
                continue
            result["frames_read"] += 1
            last_shape = list(frame.shape)
        elapsed = max(time.time() - started, 1e-6)
        result["read_fps"] = result["frames_read"] / elapsed
        result["shape"] = last_shape
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if capture is not None:
            capture.release()

    return result


def probe_mediapipe(camera: str, args: argparse.Namespace) -> dict[str, Any]:
    capture = open_camera(camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    hand_landmarker = create_hand_landmarker(args)
    sampled_frames = 0
    hand_frames = 0
    started = time.time()
    try:
        for frame_index in range(args.frames):
            ok, frame = capture.read()
            if not ok:
                continue
            if frame_index % max(args.mediapipe_sample_rate, 1) != 0:
                continue
            sampled_frames += 1
            detection = detect_hand(frame, hand_landmarker, time.monotonic_ns() // 1_000_000)
            if detection is not None:
                hand_frames += 1
    finally:
        hand_landmarker.close()
        capture.release()

    elapsed = max(time.time() - started, 1e-6)
    return {
        "camera": camera,
        "sampled_frames": sampled_frames,
        "hand_frames": hand_frames,
        "hand_detection_rate": hand_frames / max(sampled_frames, 1),
        "elapsed_seconds": elapsed,
        "sampled_fps": sampled_frames / elapsed,
    }


def main() -> None:
    args = parse_args()
    candidates = [args.camera] if args.camera else args.candidates
    camera_results = [probe_camera(candidate, args) for candidate in candidates]
    usable = [item for item in camera_results if item["opened"] and item["frames_read"] > 0]
    best_camera = args.camera or (usable[0]["candidate"] if usable else "")

    mediapipe_result = None
    if best_camera:
        mediapipe_result = probe_mediapipe(best_camera, args)

    payload = {
        "camera_results": camera_results,
        "best_camera": best_camera,
        "mediapipe": mediapipe_result,
    }
    save_json(payload, args.output_json)

    for item in camera_results:
        print(
            f"{item['candidate']}: opened={item['opened']} "
            f"frames={item['frames_read']} fps={item['read_fps']:.2f} "
            f"shape={item['shape']} error={item['error']}"
        )
    if mediapipe_result:
        print(
            "MediaPipe: "
            f"sampled={mediapipe_result['sampled_frames']} "
            f"hand_rate={mediapipe_result['hand_detection_rate']:.2%} "
            f"sampled_fps={mediapipe_result['sampled_fps']:.2f}"
        )
    print(f"Benchmark written to: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
