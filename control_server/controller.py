"""Gesture-event → desktop-input controller.

Owns pyautogui state, cooldowns, cursor smoothing, and the safety lock.
The Web client sends *gesture events*; this module decides what (if any)
real input to inject. All injection paths share a try/except wrapper that
auto-pauses on the first error so a flaky input system does not freeze
the UI thread.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import ACTION_LABELS_CN, DEFAULT_BINDINGS

LOG = logging.getLogger(__name__)

DEFAULT_COOLDOWNS_MS = {
    "click": 250,
    "drag_press": 150,
    "drag_release": 150,
    "alt_tab": 1100,
    "press_escape": 900,
    "media_playpause": 600,
    "volume_up": 200,
    "volume_down": 200,
    "media_next": 600,
    "media_prev": 600,
    "show_desktop": 1500,
    "minimize": 900,
    "close_window": 2500,
    "scroll": 80,
}

EVDEV_KEY_MAP = {
    # name → linux input-event keycode (from <linux/input-event-codes.h>)
    "esc": 1,
    "1": 2,
    "playpause": 164,
    "volumeup": 115,
    "volumedown": 114,
    "nexttrack": 163,
    "prevtrack": 165,
    "alt": 56,
    "tab": 15,
    "win": 125,           # KEY_LEFTMETA
    "leftmeta": 125,
    "d": 32,
    "f4": 62,
    "down": 108,
}


class EvdevBackend:
    """Wayland-friendly key/mouse injector using /dev/uinput directly.

    Needs:
      * python-evdev installed (pip install evdev)
      * /dev/uinput readable+writable by the running user. The recommended
        setup is one udev rule + adding the user to the `input` group:

            echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/60-uinput.rules
            sudo udevadm control --reload-rules && sudo udevadm trigger
            sudo usermod -aG input $USER  # then re-login

    Provides a pyautogui-like surface (moveTo/click/press/hotkey/scroll) so it
    can drop in for GestureController._backend.
    """

    def __init__(self) -> None:
        self.available = False
        self.warning = ""
        self.screen_size = (1920, 1080)
        self._mouse = None
        self._keyboard = None

        try:
            from evdev import UInput, AbsInfo, ecodes  # type: ignore
        except ImportError:
            self.warning = (
                "未安装 evdev；请在后端环境里运行 pip install evdev"
                "（或 conda 环境的 pip）后重启后端。"
            )
            return

        self._ecodes = ecodes
        try:
            screen_w, screen_h = _detect_screen_size()
            self.screen_size = (screen_w, screen_h)

            mouse_caps = {
                ecodes.EV_KEY: [
                    ecodes.BTN_LEFT,
                    ecodes.BTN_RIGHT,
                    ecodes.BTN_MIDDLE,
                ],
                ecodes.EV_ABS: [
                    (ecodes.ABS_X, AbsInfo(0, 0, screen_w, 0, 0, 0)),
                    (ecodes.ABS_Y, AbsInfo(0, 0, screen_h, 0, 0, 0)),
                ],
                ecodes.EV_REL: [ecodes.REL_WHEEL],
            }
            self._mouse = UInput(mouse_caps, name="gesture-control-mouse")

            kb_keys = list(EVDEV_KEY_MAP.values()) + [
                ecodes.KEY_LEFTSHIFT,
                ecodes.KEY_LEFTCTRL,
                ecodes.KEY_LEFTALT,
            ]
            self._keyboard = UInput(
                {ecodes.EV_KEY: list(set(kb_keys))}, name="gesture-control-kbd"
            )
            self.available = True
        except PermissionError as error:
            self.warning = (
                "无法打开 /dev/uinput：权限不足。请运行项目里 docs 目录的"
                " udev 设置脚本，并把当前用户加入 input 组后重新登录。"
                f"（{error}）"
            )
        except Exception as error:  # noqa: BLE001
            self.warning = f"evdev 初始化失败：{type(error).__name__}: {error}"

    # ---- pyautogui-like surface ----------------------------------------------

    def moveTo(self, x: int, y: int, _duration: float = 0.0) -> None:
        if not self._mouse:
            return
        ec = self._ecodes
        self._mouse.write(ec.EV_ABS, ec.ABS_X, int(x))
        self._mouse.write(ec.EV_ABS, ec.ABS_Y, int(y))
        self._mouse.syn()

    def click(self) -> None:
        self.mouseDown()
        time.sleep(0.02)
        self.mouseUp()

    def mouseDown(self) -> None:
        if not self._mouse:
            return
        ec = self._ecodes
        self._mouse.write(ec.EV_KEY, ec.BTN_LEFT, 1)
        self._mouse.syn()

    def mouseUp(self) -> None:
        if not self._mouse:
            return
        ec = self._ecodes
        self._mouse.write(ec.EV_KEY, ec.BTN_LEFT, 0)
        self._mouse.syn()

    def scroll(self, amount: int) -> None:
        if not self._mouse:
            return
        ec = self._ecodes
        self._mouse.write(ec.EV_REL, ec.REL_WHEEL, int(amount))
        self._mouse.syn()

    def press(self, key: str) -> None:
        code = EVDEV_KEY_MAP.get(key.lower())
        if code is None or not self._keyboard:
            return
        ec = self._ecodes
        self._keyboard.write(ec.EV_KEY, code, 1)
        self._keyboard.syn()
        time.sleep(0.02)
        self._keyboard.write(ec.EV_KEY, code, 0)
        self._keyboard.syn()

    def hotkey(self, *keys: str) -> None:
        codes = [EVDEV_KEY_MAP.get(k.lower()) for k in keys]
        if any(c is None for c in codes) or not self._keyboard:
            return
        ec = self._ecodes
        for code in codes:
            self._keyboard.write(ec.EV_KEY, code, 1)
        self._keyboard.syn()
        time.sleep(0.03)
        for code in reversed(codes):
            self._keyboard.write(ec.EV_KEY, code, 0)
        self._keyboard.syn()

    def size(self) -> tuple[int, int]:
        return self.screen_size

    def close(self) -> None:
        for ui in (self._mouse, self._keyboard):
            try:
                if ui:
                    ui.close()
            except Exception:
                pass


def _detect_screen_size() -> tuple[int, int]:
    """Best-effort screen size discovery for absolute mouse mapping.

    Tries xrandr first (works under XWayland), then a Wayland-only fallback
    via wlr-randr if available, finally a hard-coded 1920x1080 default.
    """
    for cmd in (["xrandr"], ["wlr-randr"]):
        try:
            out = subprocess.check_output(cmd, text=True, timeout=1, env=os.environ)
        except Exception:
            continue
        for line in out.splitlines():
            line = line.strip()
            if "current" in line:
                # xrandr: "Screen 0: ... current 1920 x 1080 ..."
                parts = line.split()
                try:
                    idx = parts.index("current")
                    return int(parts[idx + 1]), int(parts[idx + 3])
                except (ValueError, IndexError):
                    pass
            if "connected" in line and "primary" in line:
                for token in line.split():
                    if "x" in token and "+" in token:
                        try:
                            w, h = token.split("+")[0].split("x")
                            return int(w), int(h)
                        except ValueError:
                            pass
    return 1920, 1080


PINCH_LONG_PRESS_S = 0.42
PINCH_CLICK_MIN_S = 0.06
PINCH_CLICK_MAX_S = 0.42
DEAD_ZONE_PIXELS = 4
EMA_SLOW = 0.78
EMA_FAST = 0.45


@dataclass
class ControlState:
    available: bool = False
    enabled: bool = False
    paused: bool = False
    state: str = "locked"  # active | locked | paused | wayland-blocked | fallback
    last_action: str = "等待启用"
    warning: str = ""
    screen_size: tuple[int, int] = (0, 0)
    cursor: tuple[int, int] | None = None
    last_event_at: float = 0.0
    cooldowns_until: dict[str, float] = field(default_factory=dict)


class GestureController:
    def __init__(
        self,
        *,
        sensitivity: float = 1.2,
        safe_start: bool = True,
        cooldowns_ms: dict[str, int] | None = None,
        bindings: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.sensitivity = max(0.2, float(sensitivity))
        self.cooldowns_ms = {**DEFAULT_COOLDOWNS_MS, **(cooldowns_ms or {})}
        self.bindings = {**DEFAULT_BINDINGS, **(bindings or {})}

        self.state = ControlState()
        self.state.enabled = not safe_start
        self.state.state = "locked" if safe_start else "active"

        self._smoothed_cursor: tuple[float, float] | None = None
        self._pinch_started_at: float | None = None
        self._drag_active = False
        self._backend: Any = None
        self._backend_name = "none"

        wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"

        # Preferred backend everywhere: evdev → /dev/uinput. Works on both X11
        # and Wayland because uinput injects below the display server.
        evdev_backend = EvdevBackend()
        if evdev_backend.available:
            self._backend = evdev_backend
            self._backend_name = "evdev"
            self.state.screen_size = evdev_backend.size()
            self.state.available = True
            if not safe_start:
                self.state.state = "active"
            self.state.warning = ""

        # If evdev is not usable, on Wayland surface the install hint and stop.
        elif wayland:
            self.state.available = False
            self.state.state = "wayland-blocked"
            self.state.warning = (
                evdev_backend.warning
                or "Wayland 下需要通过 /dev/uinput 注入键鼠。请按文档配置 evdev/uinput 权限。"
            )
            self.state.last_action = "Wayland 下需 evdev 才能注入"

        # X11 fallback: try pyautogui.
        if self._backend is None and not wayland:
            try:
                pg = importlib.import_module("pyautogui")
                pg.PAUSE = 0
                pg.MINIMUM_DURATION = 0
                pg.FAILSAFE = True
                size = pg.size()
                width = getattr(size, "width", None) or size[0]
                height = getattr(size, "height", None) or size[1]
                self._backend = pg
                self._backend_name = "pyautogui"
                self.state.screen_size = (int(width), int(height))
                self.state.available = True
                if not safe_start:
                    self.state.state = "active"
                # X11 path: clean.
                self.state.warning = ""
            except Exception as error:  # noqa: BLE001
                if not self.state.warning:
                    self.state.warning = f"pyautogui 不可用：{type(error).__name__}: {error}"
                if self.state.state != "wayland-blocked":
                    self.state.available = False
                    self.state.enabled = False
                    self.state.state = "fallback"
                if not self.state.last_action or self.state.last_action == "等待启用":
                    self.state.last_action = "未检测到鼠标键盘控制能力"

    # ------------------------------------------------------------------ control

    def enable(self) -> ControlState:
        with self._lock:
            if not self.state.available:
                return self.snapshot()
            self.state.enabled = True
            self.state.paused = False
            self.state.state = "active"
            self.state.last_action = "已开启控制"
            return self.snapshot()

    def lock(self) -> ControlState:
        with self._lock:
            self._release_drag_locked()
            self.state.enabled = False
            self.state.state = "locked"
            self.state.last_action = "已锁定，按页面解锁按钮恢复"
            return self.snapshot()

    def toggle_pause(self) -> ControlState:
        with self._lock:
            if not self.state.available:
                return self.snapshot()
            if not self.state.enabled:
                self.state.enabled = True
                self.state.paused = False
                self.state.state = "active"
                self.state.last_action = "已开启控制"
                return self.snapshot()
            self.state.paused = not self.state.paused
            if self.state.paused:
                self._release_drag_locked()
                self.state.state = "paused"
                self.state.last_action = "已暂停控制"
            else:
                self.state.state = "active"
                self.state.last_action = "已恢复控制"
            return self.snapshot()

    def update_bindings(self, bindings: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        with self._lock:
            for label, spec in bindings.items():
                action = spec.get("action")
                enabled = bool(spec.get("enabled", True))
                if action not in ACTION_LABELS_CN:
                    continue
                self.bindings[label] = {"action": action, "enabled": enabled}
            return dict(self.bindings)

    def snapshot(self) -> ControlState:
        return ControlState(
            available=self.state.available,
            enabled=self.state.enabled,
            paused=self.state.paused,
            state=self.state.state,
            last_action=self.state.last_action,
            warning=self.state.warning,
            screen_size=self.state.screen_size,
            cursor=self.state.cursor,
            last_event_at=self.state.last_event_at,
            cooldowns_until=dict(self.state.cooldowns_until),
        )

    # --------------------------------------------------------------- dispatching

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            label = str(event.get("label", "") or "")
            confidence = float(event.get("confidence", 0.0))
            anchor = event.get("anchor")
            now = time.time()
            self.state.last_event_at = now

            if not self.state.available:
                return self._ack("noop", "控制不可用，仅记录手势", ok=False)
            if not self.state.enabled:
                self._release_drag_locked()
                return self._ack("noop", "已锁定，按解锁后再试", ok=False)
            if self.state.paused:
                self._release_drag_locked()
                return self._ack("noop", "控制暂停中", ok=False)
            if self.state.state == "wayland-blocked":
                return self._ack("noop", "Wayland 会话不支持注入", ok=False)

            binding = self.bindings.get(label)
            if not binding or not binding.get("enabled", True):
                self._release_drag_locked()
                return self._ack("noop", f"{label} 未绑定动作", ok=False)
            action = binding.get("action", "noop")
            if action == "noop":
                return self._ack("noop", "禁用", ok=False)

            # Pinch finalization happens whenever current label is *not* pinch
            if action != "click_or_drag":
                self._finish_pinch_locked(now, allow_click=True)

            # Dispatch
            if action == "release":
                self._release_drag_locked()
                self.state.last_action = "准备态"
                return self._ack(action, "准备态", ok=True)
            if action == "move_cursor":
                cursor = self._move_cursor(anchor)
                if cursor is None:
                    return self._ack("move_cursor", "等待手部坐标", ok=False)
                self.state.last_action = f"移动到 ({cursor[0]}, {cursor[1]})"
                return self._ack("move_cursor", self.state.last_action, ok=True)
            if action == "click_or_drag":
                return self._do_pinch(now, anchor, confidence)
            if action == "press_escape":
                return self._do_cooldowned("press_escape", lambda: self._press("esc"), "已发送 Esc")
            if action == "alt_tab":
                return self._do_cooldowned("alt_tab", lambda: self._hotkey("alt", "tab"), "切换窗口")
            if action == "media_playpause":
                return self._do_cooldowned(
                    "media_playpause", lambda: self._press("playpause"), "播放/暂停"
                )
            if action == "volume_up":
                return self._do_cooldowned("volume_up", lambda: self._press("volumeup"), "音量 +")
            if action == "volume_down":
                return self._do_cooldowned(
                    "volume_down", lambda: self._press("volumedown"), "音量 -"
                )
            if action == "media_next":
                return self._do_cooldowned(
                    "media_next", lambda: self._press("nexttrack"), "下一首"
                )
            if action == "media_prev":
                return self._do_cooldowned(
                    "media_prev", lambda: self._press("prevtrack"), "上一首"
                )
            if action == "show_desktop":
                return self._do_cooldowned(
                    "show_desktop", lambda: self._hotkey("win", "d"), "显示桌面"
                )
            if action == "minimize":
                return self._do_cooldowned(
                    "minimize", lambda: self._hotkey("win", "down"), "最小化窗口"
                )
            if action == "close_window":
                return self._do_cooldowned(
                    "close_window", lambda: self._hotkey("alt", "f4"), "关闭窗口"
                )
            if action == "scroll_up":
                return self._do_scroll(+3, "向上滚动")
            if action == "scroll_down":
                return self._do_scroll(-3, "向下滚动")
            return self._ack("noop", f"未知动作 {action}", ok=False)

    # ---------------------------------------------------------------- internals

    def _ack(self, action: str, message: str, *, ok: bool) -> dict[str, Any]:
        return {
            "type": "ack",
            "action": action,
            "ok": ok,
            "message": message,
            "control_state": self.state.state,
            "cursor": list(self.state.cursor) if self.state.cursor else None,
            "warning": self.state.warning,
        }

    def _move_cursor(self, anchor: Any) -> tuple[int, int] | None:
        if self._backend is None or anchor is None:
            return None
        try:
            ax = float(anchor[0])
            ay = float(anchor[1])
        except (TypeError, IndexError, ValueError):
            return None
        screen_w, screen_h = self.state.screen_size
        # apply sensitivity around screen center
        cx = 0.5 + (ax - 0.5) * self.sensitivity
        cy = 0.5 + (ay - 0.5) * self.sensitivity
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        target_x = cx * max(screen_w - 1, 1)
        target_y = cy * max(screen_h - 1, 1)
        if self._smoothed_cursor is None:
            sx, sy = target_x, target_y
        else:
            prev_x, prev_y = self._smoothed_cursor
            distance = ((target_x - prev_x) ** 2 + (target_y - prev_y) ** 2) ** 0.5
            ema = EMA_SLOW if distance < 30 else EMA_FAST
            sx = prev_x * ema + target_x * (1.0 - ema)
            sy = prev_y * ema + target_y * (1.0 - ema)
        self._smoothed_cursor = (sx, sy)
        cursor = (int(round(sx)), int(round(sy)))
        last = self.state.cursor
        if last is not None:
            dx = cursor[0] - last[0]
            dy = cursor[1] - last[1]
            if dx * dx + dy * dy < DEAD_ZONE_PIXELS * DEAD_ZONE_PIXELS:
                # below dead zone, do not inject
                return last
        self.state.cursor = cursor
        try:
            self._backend.moveTo(cursor[0], cursor[1], 0)
        except Exception as error:  # noqa: BLE001
            self._fail(error)
            return None
        return cursor

    def _do_pinch(
        self,
        now: float,
        anchor: Any,
        confidence: float,
    ) -> dict[str, Any]:
        cursor = self._move_cursor(anchor) if anchor is not None else self.state.cursor
        if self._pinch_started_at is None:
            self._pinch_started_at = now
            self.state.last_action = "捏合中（短=单击，长=拖拽）"
            return self._ack("click_or_drag", "捏合开始", ok=True)
        duration = now - self._pinch_started_at
        if duration >= PINCH_LONG_PRESS_S and not self._drag_active:
            if not self._cooldown_ready_locked("drag_press", now):
                return self._ack("click_or_drag", "拖拽冷却中", ok=False)
            if self._call("mouseDown"):
                self._drag_active = True
                self._set_cooldown_locked("drag_press", now)
                self.state.last_action = "进入拖拽"
                return self._ack("click_or_drag", "进入拖拽", ok=True)
        if self._drag_active:
            self.state.last_action = "拖拽中"
            return self._ack("click_or_drag", "拖拽中", ok=True)
        return self._ack("click_or_drag", "捏合保持中", ok=True)

    def _finish_pinch_locked(self, now: float, *, allow_click: bool) -> None:
        if self._pinch_started_at is None:
            return
        duration = now - self._pinch_started_at
        if self._drag_active:
            if self._call("mouseUp"):
                self.state.last_action = "释放拖拽"
            self._drag_active = False
        elif (
            allow_click
            and PINCH_CLICK_MIN_S <= duration < PINCH_CLICK_MAX_S
            and self._cooldown_ready_locked("click", now)
        ):
            if self._call("click"):
                self._set_cooldown_locked("click", now)
                self.state.last_action = "已单击"
        self._pinch_started_at = None

    def _release_drag_locked(self) -> None:
        if self._drag_active:
            self._call("mouseUp")
            self._drag_active = False
        self._pinch_started_at = None
        self._smoothed_cursor = None

    def _do_cooldowned(
        self,
        cooldown_key: str,
        action_callable,
        success_message: str,
    ) -> dict[str, Any]:
        now = time.time()
        if not self._cooldown_ready_locked(cooldown_key, now):
            remaining = self.state.cooldowns_until.get(cooldown_key, now) - now
            return self._ack(
                cooldown_key, f"{success_message} 冷却中 ({remaining:.1f}s)", ok=False
            )
        if action_callable():
            self._set_cooldown_locked(cooldown_key, now)
            self.state.last_action = success_message
            return self._ack(cooldown_key, success_message, ok=True)
        return self._ack(cooldown_key, "执行失败", ok=False)

    def _do_scroll(self, amount: int, message: str) -> dict[str, Any]:
        now = time.time()
        if not self._cooldown_ready_locked("scroll", now):
            return self._ack("scroll", "滚动冷却中", ok=False)
        if self._call("scroll", amount):
            self._set_cooldown_locked("scroll", now)
            self.state.last_action = message
            return self._ack("scroll", message, ok=True)
        return self._ack("scroll", "滚动失败", ok=False)

    def _cooldown_ready_locked(self, key: str, now: float) -> bool:
        until = self.state.cooldowns_until.get(key, 0.0)
        return now >= until

    def _set_cooldown_locked(self, key: str, now: float) -> None:
        ms = self.cooldowns_ms.get(key, 250)
        self.state.cooldowns_until[key] = now + ms / 1000.0

    def _call(self, method: str, *args: Any) -> bool:
        if self._backend is None:
            return False
        try:
            getattr(self._backend, method)(*args)
            return True
        except Exception as error:  # noqa: BLE001
            self._fail(error)
            return False

    def _press(self, key: str) -> bool:
        return self._call("press", key)

    def _hotkey(self, *keys: str) -> bool:
        if self._backend is None:
            return False
        try:
            self._backend.hotkey(*keys)
            return True
        except Exception as error:  # noqa: BLE001
            self._fail(error)
            return False

    def _fail(self, error: Exception) -> None:
        LOG.exception("control injection failed: %s", error)
        self._drag_active = False
        self.state.paused = True
        self.state.state = "paused"
        self.state.warning = f"控制注入异常：{type(error).__name__}: {error}"
        self.state.last_action = "检测到注入异常，已自动暂停"
