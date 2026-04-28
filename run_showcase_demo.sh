#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source "${HOME}/miniconda3/bin/activate" gesture-py310

python infer_camera.py \
  --mode showcase \
  --control-mode mouse \
  --camera /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --process-fps 20 \
  --safe-start \
  --mirror \
  "$@"
