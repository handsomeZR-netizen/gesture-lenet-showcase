#!/usr/bin/env python3
"""Inspect the local runtime environment for this project."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def import_version(module_name: str) -> tuple[bool, str]:
    try:
        module = __import__(module_name)
    except Exception as error:  # pragma: no cover - diagnostic script
        return False, f"unavailable ({type(error).__name__}: {error})"
    return True, getattr(module, "__version__", "unknown")


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def detect_system_camera_nodes() -> list[str]:
    return sorted(str(path) for path in Path("/dev").glob("video*"))


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run_command_capture(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 127, ""
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def main() -> None:
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")

    torch_ok, torch_version = import_version("torch")
    print(f"Torch: {torch_version}")
    if torch_ok:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: unknown (torch not installed)")

    cv2_ok, cv2_version = import_version("cv2")
    numpy_ok, numpy_version = import_version("numpy")
    pyautogui_ok, pyautogui_version = import_version("pyautogui")
    print(f"OpenCV: {cv2_version}")
    print(f"NumPy: {numpy_version}")
    print(f"PyAutoGUI package: {package_version('PyAutoGUI')}")
    print(f"PyAutoGUI runtime: {pyautogui_version}")
    print(f"nvidia-smi: {which('nvidia-smi') or 'not found'}")
    print(f"conda: {which('conda') or 'not found'}")
    print("Camera nodes:", ", ".join(detect_system_camera_nodes()) or "none")

    code, output = run_command_capture(["nvidia-smi"])
    if code == 0:
        print("\nnvidia-smi output:")
        print(output)


if __name__ == "__main__":
    main()
