"""Persisted gesture-to-action bindings on disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import DEFAULT_BINDINGS

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "gesture_control"
CONFIG_PATH = CONFIG_DIR / "bindings.json"


def load_bindings() -> dict[str, dict[str, Any]]:
    if not CONFIG_PATH.exists():
        return {label: dict(spec) for label, spec in DEFAULT_BINDINGS.items()}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {label: dict(spec) for label, spec in DEFAULT_BINDINGS.items()}
    merged = {label: dict(spec) for label, spec in DEFAULT_BINDINGS.items()}
    if isinstance(data, dict):
        for label, spec in data.items():
            if not isinstance(spec, dict):
                continue
            merged.setdefault(label, {})
            if "action" in spec:
                merged[label]["action"] = spec["action"]
            if "enabled" in spec:
                merged[label]["enabled"] = bool(spec["enabled"])
    return merged


def save_bindings(bindings: dict[str, dict[str, Any]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(bindings, indent=2, ensure_ascii=False), encoding="utf-8")
