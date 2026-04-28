"""Controller state-machine tests.

Real evdev/pyautogui injection is replaced by a stub backend so the tests run
on a CI box without /dev/uinput access.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from control_server.controller import GestureController


class StubBackend:
    """Pyautogui-like surface that records calls instead of injecting."""

    def __init__(self, w: int = 1920, h: int = 1080) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.screen = (w, h)

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args))
        return record

    def size(self):
        return self.screen


@pytest.fixture
def controller(monkeypatch):
    stub = StubBackend()
    ctrl = GestureController.__new__(GestureController)
    ctrl._lock = __import__("threading").Lock()
    ctrl.sensitivity = 1.0
    from control_server import DEFAULT_BINDINGS
    from control_server.controller import (
        DEFAULT_COOLDOWNS_MS,
        ControlState,
    )
    ctrl.cooldowns_ms = dict(DEFAULT_COOLDOWNS_MS)
    ctrl.bindings = {l: dict(s) for l, s in DEFAULT_BINDINGS.items()}
    ctrl.state = ControlState(
        available=True, enabled=True, paused=False, state="active",
        screen_size=stub.screen,
    )
    ctrl._backend = stub
    ctrl._backend_name = "stub"
    ctrl._smoothed_cursor = None
    ctrl._pinch_started_at = None
    ctrl._drag_active = False
    return ctrl, stub


def test_locked_state_does_not_inject(controller):
    ctrl, stub = controller
    ctrl.lock()
    ack = ctrl.handle_event({"label": "victory", "confidence": 0.95})
    assert ack["ok"] is False
    assert all(call[0] != "hotkey" for call in stub.calls)


def test_unlock_then_alt_tab_fires_once(controller):
    ctrl, stub = controller
    ctrl.enable()
    ack1 = ctrl.handle_event({"label": "victory", "confidence": 0.95})
    assert ack1["ok"] is True
    assert ("hotkey", ("alt", "tab")) in stub.calls

    # second event within cooldown must not fire again
    ack2 = ctrl.handle_event({"label": "victory", "confidence": 0.95})
    hotkey_calls = [c for c in stub.calls if c == ("hotkey", ("alt", "tab"))]
    assert len(hotkey_calls) == 1
    assert ack2["ok"] is False


def test_unbound_label_is_noop(controller):
    ctrl, stub = controller
    ctrl.enable()
    ctrl.bindings["victory"] = {"action": "noop", "enabled": True}
    ack = ctrl.handle_event({"label": "victory", "confidence": 0.95})
    assert ack["ok"] is False


def test_disabled_binding_is_skipped(controller):
    ctrl, stub = controller
    ctrl.enable()
    ctrl.bindings["fist"] = {"action": "press_escape", "enabled": False}
    ack = ctrl.handle_event({"label": "fist", "confidence": 0.95})
    assert ack["ok"] is False
    assert all(c != ("press", ("esc",)) for c in stub.calls)


def test_pause_resume_cycle(controller):
    ctrl, stub = controller
    ctrl.enable()
    ctrl.toggle_pause()
    assert ctrl.state.paused
    ack = ctrl.handle_event({"label": "victory", "confidence": 0.95})
    assert ack["ok"] is False
    ctrl.toggle_pause()
    assert not ctrl.state.paused


def test_move_cursor_skips_if_no_anchor(controller):
    ctrl, stub = controller
    ctrl.enable()
    ack = ctrl.handle_event({"label": "point", "confidence": 0.9})
    # No anchor → cannot map cursor; ack should be ok=False
    assert ack["ok"] is False


def test_move_cursor_with_anchor_invokes_moveto(controller):
    ctrl, stub = controller
    ctrl.enable()
    ack = ctrl.handle_event(
        {"label": "point", "confidence": 0.9, "anchor": [0.5, 0.5]}
    )
    assert ack["ok"] is True
    assert any(c[0] == "moveTo" for c in stub.calls)
