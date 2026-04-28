#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}/web_control_demo/"

python3 -m http.server "${PORT}" --bind "${HOST}" >/tmp/gesture_control_demo.log 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

sleep 1
xdg-open "${URL}" >/dev/null 2>&1 || true

echo "已启动中文 Web 展示端：${URL}"
echo "按 Ctrl-C 可停止本地服务。"

wait "${SERVER_PID}"
