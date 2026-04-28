#!/usr/bin/env python3
"""Run realtime webcam gesture demos with MediaPipe and optional LeNet letters."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from gesture_lenet.utils import (
    detect_system_camera_nodes,
    load_model_from_checkpoint,
    preprocess_grayscale_image,
    save_json,
    select_device,
    topk_predictions,
)

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass(frozen=True)
class HandDetection:
    landmarks: object
    handedness: str = "unknown"
    score: float = 0.0


@dataclass(frozen=True)
class GestureResult:
    label: str
    confidence: float
    details: dict[str, Any]


@dataclass
class RuntimeStats:
    mode: str
    camera: str
    started_at: float = field(default_factory=time.time)
    total_frames: int = 0
    processed_frames: int = 0
    hand_frames: int = 0
    low_confidence_frames: int = 0
    gesture_counts: Counter[str] = field(default_factory=Counter)
    fps_values: deque[float] = field(default_factory=lambda: deque(maxlen=360))
    latency_ms_values: deque[float] = field(default_factory=lambda: deque(maxlen=360))

    def observe(
        self,
        *,
        hand_present: bool,
        label: str,
        confidence: float,
        min_confidence: float,
        fps: float,
        latency_ms: float,
        processed: bool,
    ) -> None:
        self.total_frames += 1
        if processed:
            self.processed_frames += 1
        if hand_present:
            self.hand_frames += 1
        if hand_present and confidence < min_confidence:
            self.low_confidence_frames += 1
        self.gesture_counts[label] += 1
        if fps > 0:
            self.fps_values.append(float(fps))
        self.latency_ms_values.append(float(latency_ms))

    def to_dict(self) -> dict[str, Any]:
        elapsed = max(time.time() - self.started_at, 1e-6)
        average_fps = float(np.mean(self.fps_values)) if self.fps_values else 0.0
        p95_latency = (
            float(np.percentile(self.latency_ms_values, 95))
            if self.latency_ms_values
            else 0.0
        )
        return {
            "mode": self.mode,
            "camera": self.camera,
            "elapsed_seconds": elapsed,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "hand_frames": self.hand_frames,
            "detection_success_rate": self.hand_frames / max(self.total_frames, 1),
            "low_confidence_frames": self.low_confidence_frames,
            "average_fps": average_fps,
            "min_fps": min(self.fps_values) if self.fps_values else 0.0,
            "max_fps": max(self.fps_values) if self.fps_values else 0.0,
            "average_latency_ms": (
                float(np.mean(self.latency_ms_values)) if self.latency_ms_values else 0.0
            ),
            "p95_latency_ms": p95_latency,
            "gesture_counts": dict(self.gesture_counts.most_common()),
        }


class LabelSmoother:
    def __init__(self, window_size: int) -> None:
        self.labels: deque[str] = deque(maxlen=max(1, window_size))

    def update(self, label: str) -> str:
        self.labels.append(label)
        return Counter(self.labels).most_common(1)[0][0]


@dataclass(frozen=True)
class ControlSnapshot:
    mode: str
    status: str
    action: str
    available: bool
    enabled: bool
    manual_paused: bool
    active: bool
    screen_size: tuple[int, int] = (0, 0)
    cursor: tuple[int, int] | None = None
    warning: str = ""


class MouseKeyboardController:
    """Translate stable gesture labels into guarded desktop input events."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.mode = args.control_mode
        self.sensitivity = max(float(args.control_sensitivity), 0.1)
        self.safe_start = bool(args.safe_start)
        self.available = False
        self.enabled = self.mode == "mouse" and not self.safe_start
        self.manual_paused = False
        self.status = "off" if self.mode == "off" else "locked"
        self.last_action = "系统控制已关闭" if self.mode == "off" else "按 C 键开启系统控制"
        self.warning = ""
        self.pyautogui: Any = None
        self.screen_size = (0, 0)
        self.cursor: tuple[int, int] | None = None
        self.smoothed_cursor: tuple[float, float] | None = None
        self.pinch_started_at: float | None = None
        self.drag_active = False
        self.last_click_at = 0.0
        self.last_hotkey_at = 0.0
        self.last_escape_at = 0.0

        if self.mode != "mouse":
            return

        try:
            self.pyautogui = importlib.import_module("pyautogui")
            self.pyautogui.PAUSE = 0
            self.pyautogui.MINIMUM_DURATION = 0
            self.pyautogui.FAILSAFE = True
            size = self.pyautogui.size()
            if hasattr(size, "width") and hasattr(size, "height"):
                width = size.width
                height = size.height
            else:
                width, height = size
            self.screen_size = (int(width), int(height))
            self.available = True
            self.status = "locked" if self.safe_start else "active"
            if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
                self.warning = "Wayland 可能拦截鼠标注入；如果指针不动，建议切到 X11 会话。"
        except Exception as error:
            self.available = False
            self.enabled = False
            self.status = "fallback"
            self.last_action = "系统控制不可用，当前仅运行可视化演示"
            self.warning = f"pyautogui 不可用：{type(error).__name__}: {error}"

    def snapshot(self) -> ControlSnapshot:
        active = (
            self.mode == "mouse"
            and self.available
            and self.enabled
            and not self.manual_paused
            and self.status == "active"
        )
        return ControlSnapshot(
            mode=self.mode,
            status=self.status,
            action=self.last_action,
            available=self.available,
            enabled=self.enabled,
            manual_paused=self.manual_paused,
            active=active,
            screen_size=self.screen_size,
            cursor=self.cursor,
            warning=self.warning,
        )

    def enable(self) -> None:
        if self.mode != "mouse":
            self.last_action = "当前未启用系统控制模式"
            return
        if not self.available:
            self.status = "fallback"
            return
        self.enabled = True
        self.manual_paused = False
        self.status = "active"
        self.last_action = "已开启系统控制"

    def toggle_pause(self) -> None:
        if self.mode != "mouse" or not self.available:
            return
        if not self.enabled:
            self.enable()
            return
        self.manual_paused = not self.manual_paused
        if self.manual_paused:
            self.release_drag()
            self.status = "paused"
            self.last_action = "已暂停系统控制"
        else:
            self.status = "active"
            self.last_action = "已恢复系统控制"

    def release_drag(self) -> None:
        if not self.drag_active or self.pyautogui is None:
            self.drag_active = False
            return
        try:
            self.pyautogui.mouseUp()
            self.last_action = "已释放拖拽"
        except Exception as error:
            self._deactivate_after_error(error)
        self.drag_active = False

    def close(self) -> None:
        self.release_drag()

    def _deactivate_after_error(self, error: Exception) -> None:
        self.drag_active = False
        self.manual_paused = True
        self.status = "paused"
        self.warning = f"系统控制异常：{type(error).__name__}: {error}"
        self.last_action = "检测到控制异常，已自动暂停"

    def _call(self, method_name: str, *args: Any) -> bool:
        if self.pyautogui is None:
            return False
        try:
            getattr(self.pyautogui, method_name)(*args)
            return True
        except Exception as error:
            self._deactivate_after_error(error)
            return False

    def _hotkey(self, *keys: str) -> bool:
        if self.pyautogui is None:
            return False
        try:
            self.pyautogui.hotkey(*keys)
            return True
        except Exception as error:
            self._deactivate_after_error(error)
            return False

    def _map_to_screen(self, point: np.ndarray) -> tuple[int, int]:
        screen_w, screen_h = self.screen_size
        x = float(np.clip(0.5 + (float(point[0]) - 0.5) * self.sensitivity, 0.0, 1.0))
        y = float(np.clip(0.5 + (float(point[1]) - 0.5) * self.sensitivity, 0.0, 1.0))
        target_x = x * max(screen_w - 1, 1)
        target_y = y * max(screen_h - 1, 1)
        if self.smoothed_cursor is None:
            smooth_x, smooth_y = target_x, target_y
        else:
            smooth_x = self.smoothed_cursor[0] * 0.72 + target_x * 0.28
            smooth_y = self.smoothed_cursor[1] * 0.72 + target_y * 0.28
        self.smoothed_cursor = (smooth_x, smooth_y)
        self.cursor = (int(round(smooth_x)), int(round(smooth_y)))
        return self.cursor

    def _move_cursor(self, detection: HandDetection, label: str) -> None:
        points = landmark_points(detection.landmarks)
        if label == "Pinch":
            control_point = (points[4] + points[8]) * 0.5
        else:
            control_point = points[8]
        cursor = self._map_to_screen(control_point)
        self._call("moveTo", cursor[0], cursor[1], 0)

    def _finish_pinch(self, now: float, *, allow_click: bool) -> None:
        if self.pinch_started_at is None:
            return
        duration = now - self.pinch_started_at
        if self.drag_active:
            self.release_drag()
        elif allow_click and 0.06 <= duration < 0.42 and now - self.last_click_at > 0.35:
            if self._call("click"):
                self.last_click_at = now
                self.last_action = "已执行单击"
        self.pinch_started_at = None

    def update(
        self,
        *,
        label: str,
        detection: HandDetection | None,
        confidence: float,
        min_confidence: float,
    ) -> ControlSnapshot:
        now = time.time()
        if self.mode != "mouse":
            return self.snapshot()
        if not self.available:
            self.status = "fallback"
            return self.snapshot()
        if not self.enabled:
            self.status = "locked"
            self._finish_pinch(now, allow_click=False)
            return self.snapshot()
        if self.manual_paused:
            self.status = "paused"
            self._finish_pinch(now, allow_click=False)
            return self.snapshot()
        if detection is None:
            self.status = "waiting"
            self.last_action = "等待清晰手势进入画面"
            self._finish_pinch(now, allow_click=False)
            self.smoothed_cursor = None
            return self.snapshot()
        if confidence < min_confidence:
            self.status = "low-confidence"
            self.last_action = "识别置信度不足，暂不触发控制"
            self._finish_pinch(now, allow_click=False)
            return self.snapshot()

        self.status = "active"

        if label != "Pinch":
            self._finish_pinch(now, allow_click=True)

        if label == "Open Palm":
            self.release_drag()
            self.last_action = "准备手势，当前不触发系统操作"
        elif label == "Point":
            self._move_cursor(detection, label)
            self.last_action = "移动鼠标指针"
        elif label == "Pinch":
            self._move_cursor(detection, label)
            if self.pinch_started_at is None:
                self.pinch_started_at = now
                self.last_action = "检测到捏合，快速松开将执行单击"
            elif now - self.pinch_started_at >= 0.42:
                if not self.drag_active and self._call("mouseDown"):
                    self.drag_active = True
                self.last_action = "拖拽中"
            else:
                self.last_action = "单击准备中"
        elif label == "Victory":
            self.release_drag()
            if now - self.last_hotkey_at > 1.25:
                if self._hotkey("alt", "tab"):
                    self.last_hotkey_at = now
                    self.last_action = "Alt+Tab"
            else:
                self.last_action = "窗口切换冷却中"
        elif label == "Fist":
            self.release_drag()
            if now - self.last_escape_at > 1.1:
                if self._call("press", "esc"):
                    self.last_escape_at = now
                    self.last_action = "已发送 Esc / 重置"
            else:
                self.last_action = "重置冷却中"
        else:
            self.release_drag()
            self.last_action = "仅跟踪，不触发控制"

        return self.snapshot()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["showcase", "letter"], default="showcase")
    parser.add_argument("--checkpoint", default="outputs/improved_train/best_model.pth")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--process-fps", type=float, default=20.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.55)
    parser.add_argument("--min-hand-tracking-confidence", type=float, default=0.50)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--threshold", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--save-video", default="")
    parser.add_argument("--save-metrics", default="")
    parser.add_argument("--window-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-mode", choices=["off", "mouse"], default="mouse")
    parser.add_argument("--control-sensitivity", type=float, default=1.0)
    parser.add_argument("--safe-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_camera_source(camera_arg: str) -> str | int:
    if camera_arg.isdigit():
        if platform.system() == "Linux":
            return f"/dev/video{camera_arg}"
        return int(camera_arg)
    return camera_arg


def open_camera(camera_arg: str) -> cv2.VideoCapture:
    source = resolve_camera_source(camera_arg)
    attempts: list[tuple[str | int, int | None]] = [(source, cv2.CAP_V4L2)]
    if isinstance(source, str) and source.startswith("/dev/video"):
        suffix = source.removeprefix("/dev/video")
        if suffix.isdigit():
            attempts.append((int(suffix), cv2.CAP_V4L2))
    attempts.append((source, None))

    for candidate, backend in attempts:
        capture = (
            cv2.VideoCapture(candidate)
            if backend is None
            else cv2.VideoCapture(candidate, backend)
        )
        if capture.isOpened():
            return capture
        capture.release()

    cameras = detect_system_camera_nodes()
    raise RuntimeError(
        f"无法打开摄像头 {camera_arg}。"
        f"解析后的输入源：{source}。"
        f"当前检测到的系统摄像头节点：{', '.join(cameras) if cameras else 'none'}"
    )


def ensure_hand_landmarker_model(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 MediaPipe 手部模型到 {model_path} ...")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
    return model_path


def create_hand_landmarker(args: argparse.Namespace) -> mp.tasks.vision.HandLandmarker:
    model_path = ensure_hand_landmarker_model(args.hand_model)
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_hand_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hand(
    frame: cv2.typing.MatLike,
    hand_landmarker: mp.tasks.vision.HandLandmarker,
    timestamp_ms: int,
) -> HandDetection | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None

    handedness = "unknown"
    score = 0.0
    if result.handedness and result.handedness[0]:
        category = result.handedness[0][0]
        handedness = category.category_name or "unknown"
        score = float(category.score or 0.0)

    return HandDetection(
        landmarks=result.hand_landmarks[0],
        handedness=handedness,
        score=score,
    )


def landmark_points(hand_landmarks: object) -> np.ndarray:
    return np.array([[point.x, point.y, point.z] for point in hand_landmarks], dtype=np.float32)


def distance(points: np.ndarray, start: int, end: int) -> float:
    return float(np.linalg.norm(points[start, :2] - points[end, :2]))


def palm_size(points: np.ndarray) -> float:
    return max(distance(points, 0, 9), distance(points, 5, 17), 1e-4)


def finger_states(points: np.ndarray) -> dict[str, bool]:
    scale = palm_size(points)
    vertical_margin = 0.08 * scale
    states = {
        "index": bool(points[8, 1] < points[6, 1] - vertical_margin),
        "middle": bool(points[12, 1] < points[10, 1] - vertical_margin),
        "ring": bool(points[16, 1] < points[14, 1] - vertical_margin),
        "pinky": bool(points[20, 1] < points[18, 1] - vertical_margin),
    }
    states["thumb"] = bool(
        distance(points, 4, 9) > distance(points, 3, 9) * 1.12
        and distance(points, 4, 5) > 0.45 * scale
    )
    return states


def classify_showcase_gesture(hand_landmarks: object) -> GestureResult:
    points = landmark_points(hand_landmarks)
    scale = palm_size(points)
    states = finger_states(points)
    extended = sum(1 for is_extended in states.values() if is_extended)
    pinch_distance = distance(points, 4, 8) / scale
    tips_to_palm = np.mean([distance(points, tip, 0) / scale for tip in (8, 12, 16, 20)])

    if pinch_distance < 0.34:
        label = "Pinch"
        confidence = float(np.clip(1.0 - pinch_distance / 0.34, 0.55, 0.99))
    elif states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        label = "Victory"
        confidence = 0.88
    elif states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
        label = "Point"
        confidence = 0.86
    elif extended >= 4:
        label = "Open Palm"
        confidence = 0.92
    elif extended <= 1 and tips_to_palm < 1.25:
        label = "Fist"
        confidence = 0.88
    else:
        label = "Gesture"
        confidence = 0.62

    return GestureResult(
        label=label,
        confidence=confidence,
        details={
            "finger_states": states,
            "pinch_distance": pinch_distance,
            "extended_fingers": extended,
        },
    )


def landmarks_bbox(
    hand_landmarks: object,
    frame_width: int,
    frame_height: int,
    margin: float = 0.25,
) -> tuple[int, int, int, int]:
    points = landmark_points(hand_landmarks)
    xs = points[:, 0] * frame_width
    ys = points[:, 1] * frame_height
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    side = max(max_x - min_x, max_y - min_y) * (1.0 + margin * 2.0)
    side = max(side, 96.0)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    x2 = int(round(center_x + side / 2.0))
    y2 = int(round(center_y + side / 2.0))
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    return x1, y1, x2, y2


def draw_landmarks(
    frame: cv2.typing.MatLike,
    hand_landmarks: object,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    points: list[tuple[int, int]] = []
    for landmark in hand_landmarks:
        x = max(0, min(width - 1, int(landmark.x * width)))
        y = max(0, min(height - 1, int(landmark.y * height)))
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, color, -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (255, 255, 0), 2)


def put_label(
    frame: cv2.typing.MatLike,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.72,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_text(
    frame: cv2.typing.MatLike,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (235, 238, 242),
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 12, 16), thickness + 2)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_card(
    frame: cv2.typing.MatLike,
    rect: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] = (35, 39, 48),
    border: tuple[int, int, int] = (68, 76, 90),
) -> None:
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1)


def draw_progress(
    frame: cv2.typing.MatLike,
    rect: tuple[int, int, int, int],
    value: float,
    *,
    color: tuple[int, int, int],
    background: tuple[int, int, int] = (52, 57, 68),
) -> None:
    x, y, w, h = rect
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), background, -1)
    cv2.rectangle(frame, (x, y), (x + int(w * value), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (82, 92, 108), 1)


def draw_metric(
    frame: cv2.typing.MatLike,
    label: str,
    value: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (111, 220, 205),
) -> None:
    x, y = origin
    draw_text(frame, label.upper(), (x, y), scale=0.42, color=(154, 166, 184), thickness=1)
    draw_text(frame, value, (x, y + 30), scale=0.74, color=color, thickness=2)


def wrap_text(text: str, max_chars: int) -> list[str]:
    if " " not in text:
        return [text[index:index + max_chars] for index in range(0, len(text), max_chars)] or [""]
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if len(candidate) <= max_chars:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word[:max_chars]
    if line:
        lines.append(line)
    return lines or [""]


def should_show_window(args: argparse.Namespace) -> bool:
    if args.no_window or args.window_mode == "off":
        return False
    if args.window_mode == "on":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_letter_prediction(
    frame: cv2.typing.MatLike,
    detection: HandDetection,
    model: Any,
    checkpoint: dict[str, Any],
    device: Any,
    args: argparse.Namespace,
) -> tuple[GestureResult, cv2.typing.MatLike, tuple[int, int, int, int]]:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = landmarks_bbox(detection.landmarks, frame_width, frame_height)
    roi = frame[y1:y2, x1:x2]
    tensor, preview = preprocess_grayscale_image(image=roi, threshold=args.threshold)
    predictions = topk_predictions(
        model=model,
        tensor=tensor,
        class_names=list(checkpoint["class_names"]),
        device=device,
        k=3,
    )
    top1 = predictions[0]
    label = str(top1["label"])
    confidence = float(top1["confidence"])
    if confidence < args.min_confidence:
        label = "Unknown"

    return (
        GestureResult(
            label=label,
            confidence=confidence,
            details={"top3": predictions},
        ),
        cv2.resize(preview, (140, 140), interpolation=cv2.INTER_NEAREST),
        (x1, y1, x2, y2),
    )


def render_overlay(
    frame: cv2.typing.MatLike,
    *,
    mode: str,
    label: str,
    raw_label: str,
    confidence: float,
    fps: float,
    latency_ms: float,
    hand_present: bool,
    handedness: str,
    elapsed: float,
    color: tuple[int, int, int],
    control: ControlSnapshot,
    stats: RuntimeStats,
    processed: bool,
    process_fps: float,
    preview: cv2.typing.MatLike | None = None,
    top3: list[dict[str, Any]] | None = None,
) -> cv2.typing.MatLike:
    display = np.full((720, 1280, 3), (20, 23, 29), dtype=np.uint8)
    cv2.rectangle(display, (0, 0), (1280, 720), (22, 27, 34), -1)
    cv2.line(display, (0, 70), (1280, 70), (58, 66, 78), 1)

    status_labels = {
        "active": "运行中",
        "locked": "已锁定",
        "paused": "已暂停",
        "waiting": "等待手势",
        "low-confidence": "低置信度",
        "fallback": "演示模式",
        "off": "已关闭",
    }
    mode_labels = {
        "showcase": "展示模式",
        "letter": "字母识别",
        "mouse": "系统控制",
        "off": "仅展示",
    }
    gesture_labels = {
        "No hand": "未检测到手",
        "Open Palm": "张开手掌",
        "Point": "单指指向",
        "Pinch": "捏合",
        "Victory": "剪刀手",
        "Fist": "握拳",
        "Gesture": "普通手势",
        "Unknown": "未知",
    }

    draw_text(display, "手势控制计算机演示系统", (24, 38), scale=0.88, color=(245, 247, 250), thickness=2)
    draw_text(
        display,
        "摄像头 -> MediaPipe 手部关键点 -> 规则引擎 -> 鼠标 / 键盘控制",
        (24, 62),
        scale=0.45,
        color=(162, 174, 190),
    )

    status_colors = {
        "active": (88, 214, 141),
        "locked": (64, 156, 255),
        "paused": (95, 177, 255),
        "waiting": (230, 195, 84),
        "low-confidence": (69, 183, 255),
        "fallback": (83, 101, 230),
        "off": (160, 168, 178),
    }
    status_color = status_colors.get(control.status, (160, 168, 178))
    cv2.rectangle(display, (1000, 22), (1256, 56), (38, 44, 54), -1)
    cv2.rectangle(display, (1000, 22), (1256, 56), status_color, 1)
    draw_text(
        display,
        f"控制状态：{status_labels.get(control.status, control.status)}",
        (1014, 45),
        scale=0.48,
        color=status_color,
        thickness=2,
    )

    left = (24, 88, 760, 608)
    right = (808, 88, 448, 608)
    draw_card(display, left)
    draw_card(display, right)

    camera_x, camera_y = left[0] + 14, left[1] + 50
    camera_w, camera_h = left[2] - 28, 548
    camera_view = cv2.resize(frame, (camera_w, camera_h), interpolation=cv2.INTER_AREA)
    display[camera_y:camera_y + camera_h, camera_x:camera_x + camera_w] = camera_view
    cv2.rectangle(display, (camera_x, camera_y), (camera_x + camera_w, camera_y + camera_h), (83, 94, 112), 1)
    draw_text(display, "实时画面与手部关键点", (left[0] + 18, left[1] + 30), scale=0.5, color=(176, 188, 204))
    cv2.rectangle(display, (camera_x + 16, camera_y + 16), (camera_x + camera_w - 16, camera_y + camera_h - 16), (82, 100, 118), 1)

    if preview is not None:
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR) if preview.ndim == 2 else preview
        preview_bgr = cv2.resize(preview_bgr, (120, 120), interpolation=cv2.INTER_NEAREST)
        px = camera_x + camera_w - 136
        py = camera_y + 20
        display[py:py + 120, px:px + 120] = preview_bgr
        cv2.rectangle(display, (px, py), (px + 120, py + 120), (245, 247, 250), 1)
        draw_text(display, "识别区域", (px + 8, py + 112), scale=0.44, color=(245, 247, 250))

    panel_x = right[0] + 20
    panel_y = right[1] + 34
    display_label = gesture_labels.get(label, label)
    display_raw_label = gesture_labels.get(raw_label, raw_label)
    draw_text(display, "当前识别结果", (panel_x, panel_y), scale=0.48, color=(154, 166, 184))
    draw_text(display, display_label, (panel_x, panel_y + 46), scale=1.04, color=color, thickness=2)
    draw_text(
        display,
        f"原始结果：{display_raw_label} | 检测到手：{'是' if hand_present else '否'} {handedness}",
        (panel_x, panel_y + 78),
        scale=0.43,
        color=(190, 200, 214),
    )
    draw_progress(display, (panel_x, panel_y + 96, 390, 12), confidence, color=color)
    draw_text(display, f"置信度 {confidence:.2f}", (panel_x, panel_y + 130), scale=0.48, color=(206, 214, 225))

    metrics_y = panel_y + 168
    draw_metric(display, "FPS", f"{fps:.1f}", (panel_x, metrics_y))
    draw_metric(display, "延迟", f"{latency_ms:.1f} ms", (panel_x + 132, metrics_y), color=(95, 177, 255))
    hand_rate = stats.hand_frames / max(stats.total_frames, 1)
    draw_metric(display, "检出率", f"{hand_rate:.0%}", (panel_x + 292, metrics_y), color=(230, 195, 84))
    draw_text(
        display,
        f"已处理帧数：{stats.processed_frames}/{stats.total_frames} | 目标推理 FPS：{process_fps:.1f} | 本帧新推理：{'是' if processed else '否'}",
        (panel_x, metrics_y + 70),
        scale=0.42,
        color=(154, 166, 184),
    )

    control_y = metrics_y + 108
    draw_text(display, "控制信息", (panel_x, control_y), scale=0.48, color=(154, 166, 184))
    screen_text = f"{control.screen_size[0]}x{control.screen_size[1]}" if control.screen_size != (0, 0) else "未获取"
    cursor_text = f"{control.cursor[0]},{control.cursor[1]}" if control.cursor else "未获取"
    draw_text(
        display,
        f"模式：{mode_labels.get(control.mode, control.mode)} | 已启用：{'是' if control.enabled else '否'} | 屏幕：{screen_text}",
        (panel_x, control_y + 30),
        scale=0.45,
    )
    draw_text(display, f"指针位置：{cursor_text}", (panel_x, control_y + 56), scale=0.45, color=(206, 214, 225))
    for line_index, line in enumerate(wrap_text(f"当前动作：{control.action}", 26)[:2]):
        draw_text(display, line, (panel_x, control_y + 88 + line_index * 24), scale=0.45, color=(111, 220, 205))

    guide_y = control_y + 148
    draw_text(display, "手势说明", (panel_x, guide_y), scale=0.48, color=(154, 166, 184))
    guide_lines = [
        ("张开手掌", "准备状态 / 不执行操作"),
        ("单指指向", "移动鼠标指针"),
        ("捏合", "短按单击 / 长按拖拽"),
        ("剪刀手", "切换窗口 Alt+Tab"),
        ("握拳", "发送 Esc / 重置"),
    ]
    y = guide_y + 30
    for gesture, action in guide_lines:
        draw_text(display, gesture, (panel_x, y), scale=0.46, color=(245, 247, 250), thickness=1)
        draw_text(display, action, (panel_x + 146, y), scale=0.46, color=(190, 200, 214), thickness=1)
        y += 27

    if top3:
        draw_text(display, "字母预测 Top-3", (panel_x + 250, guide_y), scale=0.48, color=(154, 166, 184))
        top_y = guide_y + 30
        for prediction in top3[:3]:
            draw_text(
                display,
                f"{prediction['label']}: {prediction['confidence']:.2f}",
                (panel_x + 250, top_y),
                scale=0.46,
                color=(230, 195, 84),
            )
            top_y += 27

    footer_y = 668
    cv2.rectangle(display, (24, footer_y - 22), (1256, 704), (35, 39, 48), -1)
    cv2.rectangle(display, (24, footer_y - 22), (1256, 704), (68, 76, 90), 1)
    draw_text(
        display,
        "按键：C 开启系统控制 | 空格 暂停/恢复 | Q 或 Esc 退出 | 请保持单只手清晰进入画面",
        (42, footer_y + 5),
        scale=0.47,
        color=(235, 238, 242),
    )
    draw_text(
        display,
        f"当前模式：{mode_labels.get(mode, mode)} | 运行时间：{elapsed:.1f}s | 安全锁：{'已开启' if not control.enabled else '已解除'}",
        (42, footer_y + 31),
        scale=0.42,
        color=(154, 166, 184),
    )

    if control.warning:
        warning = wrap_text(control.warning, 32)[0]
        draw_text(display, warning, (700, footer_y + 31), scale=0.42, color=(83, 101, 230))

    return display


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = None
    checkpoint: dict[str, Any] = {}
    if args.mode == "letter":
        model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    hand_landmarker = create_hand_landmarker(args)
    capture = open_camera(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        backend_name = capture.getBackendName()
    except Exception:
        backend_name = "unknown"

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    print(
        f"摄像头已打开：source={resolve_camera_source(args.camera)!r} "
        f"backend={backend_name} width={actual_width} height={actual_height} "
        f"fps={actual_fps:.2f} mode={args.mode}"
    )

    show_window = should_show_window(args)
    stats = RuntimeStats(mode=args.mode, camera=str(resolve_camera_source(args.camera)))
    smoother = LabelSmoother(args.smoothing_window)
    controller = MouseKeyboardController(args)
    writer = None
    previous_time = 0.0
    start_time = time.time()
    window_name = "手势控制计算机演示系统"
    process_interval = 0.0 if args.process_fps <= 0 else 1.0 / float(args.process_fps)
    last_process_time = 0.0
    last_detection: HandDetection | None = None
    last_raw_label = "No hand"
    last_stable_label = "No hand"
    last_confidence = 0.0
    last_color = (0, 120, 255)
    last_preview = None
    last_top3 = None
    last_handedness = ""
    last_processed_at = 0.0

    control_snapshot = controller.snapshot()
    if control_snapshot.warning:
        print(f"控制提示：{control_snapshot.warning}")

    try:
        while True:
            loop_start = time.time()
            success, frame = capture.read()
            if not success:
                raise RuntimeError("摄像头画面意外中断。")
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            processed = process_interval == 0.0 or now - last_process_time >= process_interval
            if processed:
                last_process_time = now
                last_processed_at = now
                timestamp_ms = time.monotonic_ns() // 1_000_000
                detection = detect_hand(frame, hand_landmarker, timestamp_ms)
                raw_label = "No hand"
                stable_label = "No hand"
                confidence = 0.0
                color = (0, 120, 255)
                preview = None
                top3 = None
                handedness = ""

                if detection is not None:
                    handedness = f"({detection.handedness} {detection.score:.2f})"
                    if args.mode == "showcase":
                        result = classify_showcase_gesture(detection.landmarks)
                        raw_label = result.label
                        confidence = result.confidence
                        stable_label = smoother.update(raw_label)
                        color = (0, 255, 0) if confidence >= args.min_confidence else (0, 215, 255)
                    else:
                        assert model is not None
                        result, preview, bbox = run_letter_prediction(
                            frame=frame,
                            detection=detection,
                            model=model,
                            checkpoint=checkpoint,
                            device=device,
                            args=args,
                        )
                        raw_label = result.label
                        confidence = result.confidence
                        stable_label = smoother.update(raw_label)
                        top3 = result.details["top3"]
                        color = (0, 255, 0) if confidence >= args.min_confidence else (0, 215, 255)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                else:
                    smoother.labels.clear()

                last_detection = detection
                last_raw_label = raw_label
                last_stable_label = stable_label
                last_confidence = confidence
                last_color = color
                last_preview = preview
                last_top3 = top3
                last_handedness = handedness

            detection = last_detection
            if detection is not None and now - last_processed_at > 0.6:
                detection = None
                last_detection = None
                last_raw_label = "No hand"
                last_stable_label = "No hand"
                last_confidence = 0.0
                last_color = (0, 120, 255)
                last_preview = None
                last_top3 = None
                last_handedness = ""

            hand_present = detection is not None
            raw_label = last_raw_label
            stable_label = last_stable_label
            confidence = last_confidence
            color = last_color
            preview = last_preview
            top3 = last_top3
            handedness = last_handedness

            if detection is not None:
                draw_landmarks(frame, detection.landmarks, (0, 255, 0))

            current_time = time.time()
            fps = 0.0 if previous_time == 0 else 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            latency_ms = (time.time() - loop_start) * 1000.0
            elapsed = time.time() - start_time
            control_label = stable_label if args.mode == "showcase" else "Gesture"
            control_snapshot = controller.update(
                label=control_label,
                detection=detection,
                confidence=confidence,
                min_confidence=args.min_confidence,
            )

            stats.observe(
                hand_present=hand_present,
                label=stable_label,
                confidence=confidence,
                min_confidence=args.min_confidence,
                fps=fps,
                latency_ms=latency_ms,
                processed=processed,
            )

            display = render_overlay(
                frame,
                mode=args.mode,
                label=stable_label,
                raw_label=raw_label,
                confidence=confidence,
                fps=fps,
                latency_ms=latency_ms,
                hand_present=hand_present,
                handedness=handedness,
                elapsed=elapsed,
                color=color,
                control=control_snapshot,
                stats=stats,
                processed=processed,
                process_fps=args.process_fps,
                preview=preview,
                top3=top3,
            )

            if writer is None and args.save_video:
                save_path = Path(args.save_video)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_fps = actual_fps if actual_fps and actual_fps > 1 else float(args.fps)
                writer = cv2.VideoWriter(
                    str(save_path),
                    fourcc,
                    output_fps,
                    (display.shape[1], display.shape[0]),
                )
            if writer is not None:
                writer.write(display)

            if show_window:
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c"):
                    controller.enable()
                elif key == ord(" "):
                    controller.toggle_pause()

            if args.duration > 0 and elapsed >= args.duration:
                break
    finally:
        controller.close()
        capture.release()
        hand_landmarker.close()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    payload = stats.to_dict()
    if args.save_metrics:
        save_json(payload, args.save_metrics)
    print(
        "运行摘要："
        f"frames={payload['total_frames']} "
        f"hand_rate={payload['detection_success_rate']:.2%} "
        f"avg_fps={payload['average_fps']:.2f} "
        f"p95_latency_ms={payload['p95_latency_ms']:.2f}"
    )


if __name__ == "__main__":
    main()
