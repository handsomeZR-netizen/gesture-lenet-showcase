#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${HOME}/miniconda3/bin/conda"
ENV_NAME="gesture-py310"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "conda not found at ${CONDA_BIN}"
  echo "Run ./install_miniconda.sh first."
  exit 1
fi

cd "${PROJECT_DIR}"
"${CONDA_BIN}" env create -f "${PROJECT_DIR}/environment.yml" || "${CONDA_BIN}" env update -f "${PROJECT_DIR}/environment.yml" --prune
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo
echo "Environment ready. Activate it with:"
echo "  source ${HOME}/miniconda3/bin/activate ${ENV_NAME}"
