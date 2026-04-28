#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source "${HOME}/miniconda3/bin/activate" gesture-py310

python infer_camera.py \
  --mode letter \
  --checkpoint outputs/improved_train/best_model.pth \
  --camera /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 30
