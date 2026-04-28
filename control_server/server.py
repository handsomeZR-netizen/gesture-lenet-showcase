"""FastAPI bridge between the browser and pyautogui.

Endpoints:
    GET  /                      → static web demo (web_control_demo/)
    GET  /api/status            → controller + screen + warning state
    GET  /api/bindings          → current gesture→action map
    PUT  /api/bindings          → replace bindings (persisted)
    POST /api/control/{lock|unlock|toggle_pause}
    POST /api/dataset/{label}   → append landmark samples to JSONL
    GET  /api/actions           → list of available action names + cn labels
    WS   /ws/control            → real-time gesture events
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ACTION_LABELS_CN, AVAILABLE_ACTIONS
from .bindings import load_bindings, save_bindings
from .controller import GestureController

LOG = logging.getLogger("gesture_control.server")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web_control_demo"
DATA_ROOT = PROJECT_ROOT / "data" / "gesture_keypoints"

ALLOWED_DATASET_LABELS = {
    "open_palm",
    "point",
    "pinch",
    "fist",
    "victory",
    "ok",
    "thumbs_up",
    "thumbs_down",
    "three",
    "call",
}


class BindingSpec(BaseModel):
    action: str
    enabled: bool = True


class BindingsPayload(BaseModel):
    bindings: dict[str, BindingSpec]


class DatasetSample(BaseModel):
    landmarks: list[list[float]]
    handedness: str = "Right"
    ts: float | None = None


class DatasetPayload(BaseModel):
    samples: list[DatasetSample]


def create_app(*, sensitivity: float = 1.2, safe_start: bool = True) -> FastAPI:
    bindings_state = load_bindings()
    controller = GestureController(
        sensitivity=sensitivity,
        safe_start=safe_start,
        bindings=bindings_state,
    )

    app = FastAPI(title="手势控制后端", version="0.1.0")

    @app.get("/")
    def index() -> RedirectResponse:
        # 重定向到带尾斜杠的静态目录，让 index.html 里的相对路径
        # ./styles.css / ./app.js / ./modules/... 解析正确
        return RedirectResponse(url="/web_control_demo/", status_code=307)

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        snap = controller.snapshot()
        return JSONResponse(
            {
                "available": snap.available,
                "enabled": snap.enabled,
                "paused": snap.paused,
                "state": snap.state,
                "last_action": snap.last_action,
                "warning": snap.warning,
                "screen_size": list(snap.screen_size),
                "cursor": list(snap.cursor) if snap.cursor else None,
                "bindings": controller.bindings,
            }
        )

    @app.get("/api/actions")
    def api_actions() -> JSONResponse:
        return JSONResponse(
            {
                "actions": [
                    {"name": name, "label": ACTION_LABELS_CN[name]}
                    for name in AVAILABLE_ACTIONS
                ]
            }
        )

    @app.get("/api/bindings")
    def api_bindings_get() -> JSONResponse:
        return JSONResponse({"bindings": controller.bindings})

    @app.put("/api/bindings")
    def api_bindings_put(payload: BindingsPayload) -> JSONResponse:
        new_bindings = {
            label: spec.model_dump() for label, spec in payload.bindings.items()
        }
        merged = controller.update_bindings(new_bindings)
        save_bindings(merged)
        return JSONResponse({"bindings": merged})

    @app.post("/api/control/unlock")
    def api_control_unlock() -> JSONResponse:
        snap = controller.enable()
        return JSONResponse({"state": snap.state, "last_action": snap.last_action})

    @app.post("/api/control/lock")
    def api_control_lock() -> JSONResponse:
        snap = controller.lock()
        return JSONResponse({"state": snap.state, "last_action": snap.last_action})

    @app.post("/api/control/toggle_pause")
    def api_control_toggle() -> JSONResponse:
        snap = controller.toggle_pause()
        return JSONResponse({"state": snap.state, "last_action": snap.last_action})

    @app.post("/api/dataset/{label}")
    def api_dataset_post(label: str, payload: DatasetPayload) -> JSONResponse:
        if label not in ALLOWED_DATASET_LABELS:
            raise HTTPException(status_code=400, detail=f"unknown label {label}")
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        path = DATA_ROOT / f"{label}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for sample in payload.samples:
                if len(sample.landmarks) != 21:
                    continue
                if any(len(pt) != 3 for pt in sample.landmarks):
                    continue
                record = {
                    "landmarks": sample.landmarks,
                    "handedness": sample.handedness,
                    "ts": sample.ts or time.time(),
                }
                fh.write(json.dumps(record) + "\n")
        return JSONResponse({"label": label, "added": len(payload.samples)})

    @app.get("/api/dataset/summary")
    def api_dataset_summary() -> JSONResponse:
        summary: dict[str, int] = {}
        for label in ALLOWED_DATASET_LABELS:
            path = DATA_ROOT / f"{label}.jsonl"
            if not path.exists():
                summary[label] = 0
                continue
            with path.open("r", encoding="utf-8") as fh:
                summary[label] = sum(1 for line in fh if line.strip())
        return JSONResponse({"counts": summary})

    @app.websocket("/ws/control")
    async def ws_control(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "invalid json"})
                    continue
                msg_type = payload.get("type", "gesture")
                if msg_type == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
                    continue
                if msg_type == "control":
                    action = payload.get("action")
                    if action == "unlock":
                        snap = controller.enable()
                    elif action == "lock":
                        snap = controller.lock()
                    elif action == "toggle_pause":
                        snap = controller.toggle_pause()
                    else:
                        await ws.send_json({"type": "error", "message": f"unknown control {action}"})
                        continue
                    await ws.send_json(
                        {
                            "type": "control_state",
                            "state": snap.state,
                            "last_action": snap.last_action,
                            "warning": snap.warning,
                        }
                    )
                    continue
                # default: treat as gesture event
                # offload pyautogui call so the event loop is not blocked by it
                ack = await asyncio.to_thread(controller.handle_event, payload)
                await ws.send_json(ack)
        except WebSocketDisconnect:
            return
        except Exception as error:  # noqa: BLE001
            LOG.exception("ws error: %s", error)
            try:
                await ws.send_json({"type": "error", "message": str(error)})
            except Exception:
                pass

    # 静态文件挂载
    app.mount(
        "/web_control_demo",
        StaticFiles(directory=str(WEB_ROOT), html=True),
        name="web_control_demo",
    )
    # MediaPipe 手部模型在项目根 models/，浏览器通过 ../models/ 引用
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        app.mount(
            "/models",
            StaticFiles(directory=str(models_dir)),
            name="hand_models",
        )

    return app


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sensitivity", type=float, default=1.2)
    parser.add_argument("--no-safe-start", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(sensitivity=args.sensitivity, safe_start=not args.no_safe_start)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
