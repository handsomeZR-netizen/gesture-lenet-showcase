#!/usr/bin/env bash
set -euo pipefail

MINICONDA_DIR="${HOME}/miniconda3"
INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

if [[ -d "${MINICONDA_DIR}" ]]; then
  echo "Miniconda already exists at ${MINICONDA_DIR}"
  exit 0
fi

echo "Downloading Miniconda installer..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${INSTALLER_URL}" -o "${INSTALLER}"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "${INSTALLER}" "${INSTALLER_URL}"
else
  echo "Neither curl nor wget is available."
  exit 1
fi

echo "Installing Miniconda into ${MINICONDA_DIR}..."
bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"

echo
echo "Miniconda installed. Initialize your shell with:"
echo "  ${MINICONDA_DIR}/bin/conda init bash"
