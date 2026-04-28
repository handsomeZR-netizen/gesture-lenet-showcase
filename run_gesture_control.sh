#!/usr/bin/env bash
# 启动手势控制后端：FastAPI + 浏览器界面 + evdev /dev/uinput 注入。
set -euo pipefail

cd "$(dirname "$0")"

# 当前进程的 supplementary groups 必须包含 input，否则 evdev 打不开 /dev/uinput。
# 如果没有 input，就用 sg 重新派生一个带 input 组的 shell 再回来跑自己。
if ! id -nG | tr ' ' '\n' | grep -qx input; then
  if getent group input | grep -q "$USER"; then
    echo "[info] 当前 shell 缺少 input 组，使用 sg input + bash 重新派生..."
    # sg 默认走 /bin/sh，显式指定 bash 才能用 source/数组等 bash 语法
    exec sg input -c "/bin/bash $(printf %q "$0") $*"
  else
    echo "[warning] 当前用户不在 input 组，evdev 注入将不可用。" >&2
    echo "          请运行: sudo bash setup_uinput.sh （只需一次）" >&2
  fi
fi

source "${HOME}/miniconda3/bin/activate" gesture-py310

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
URL="http://${HOST}:${PORT}/"

echo "启动 Python 后端: ${URL}"
echo "  - 浏览器界面会在端口启动后自动打开"
echo "  - 按 Ctrl-C 退出"
echo ""

(
  for _ in $(seq 1 20); do
    if python3 -c "import urllib.request,sys; urllib.request.urlopen('${URL}api/status', timeout=1)" >/dev/null 2>&1; then
      xdg-open "${URL}" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 0.5
  done
) &

exec python -m control_server.server --host "${HOST}" --port "${PORT}" "$@"
