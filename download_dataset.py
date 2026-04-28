#!/usr/bin/env python3
"""Attempt to download Sign Language MNIST via the Kaggle CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path


DATASET = "datamunge/sign-language-mnist"
EXPECTED_FILES = {"sign_mnist_train.csv", "sign_mnist_test.csv"}
PUBLIC_MIRRORS = {
    "sign_mnist_train.csv": [
        "https://raw.githubusercontent.com/emanbuc/ASL-Recognition-Deep-Learning/main/datasets/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv",
        "https://raw.githubusercontent.com/gchilingaryan/Sign-Language/master/sign_mnist_train.csv",
    ],
    "sign_mnist_test.csv": [
        "https://raw.githubusercontent.com/emanbuc/ASL-Recognition-Deep-Learning/main/datasets/sign-language-mnist/sign_mnist_test.csv",
        "https://raw.githubusercontent.com/gchilingaryan/Sign-Language/master/sign_mnist_test.csv",
    ],
}


def has_kaggle_credentials() -> bool:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    env_ready = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    return token_path.exists() or env_ready


def verify_dataset(output_dir: Path) -> None:
    existing_files = {path.name for path in output_dir.iterdir()}
    missing = EXPECTED_FILES - existing_files
    if missing:
        raise RuntimeError(
            "Dataset is still incomplete. Missing files: " + ", ".join(sorted(missing))
        )


def download_public_mirror(output_dir: Path) -> None:
    for filename, urls in PUBLIC_MIRRORS.items():
        destination = output_dir / filename
        if destination.exists():
            continue

        last_error: Exception | None = None
        for url in urls:
            try:
                print(f"Downloading {filename} from {url}")
                urllib.request.urlretrieve(url, destination)
                break
            except Exception as error:  # pragma: no cover - network-dependent
                last_error = error
                if destination.exists():
                    destination.unlink()
        else:
            raise RuntimeError(f"Unable to download {filename} from public mirrors: {last_error}")


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    if EXPECTED_FILES.issubset({path.name for path in output_dir.iterdir()}):
        print("Dataset already present under data/raw.")
        return

    if not has_kaggle_credentials():
        print("Kaggle credentials not found. Falling back to public GitHub mirrors.")
        download_public_mirror(output_dir)
        verify_dataset(output_dir)
        print("Dataset downloaded successfully from public mirrors.")
        return

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print("Kaggle download failed. Falling back to public GitHub mirrors.")
        download_public_mirror(output_dir)
        verify_dataset(output_dir)
        print("Dataset downloaded successfully from public mirrors.")
        return

    verify_dataset(output_dir)

    print("Dataset downloaded successfully.")


if __name__ == "__main__":
    main()
