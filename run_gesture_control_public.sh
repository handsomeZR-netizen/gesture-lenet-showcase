#!/usr/bin/env bash
# 启动「公网完整版」后端：自动生成 / 复用 token，绑定全网卡，启用 CORS。
# 配套 Cloudflare Tunnel 暴露公网 URL。
#
# 用法：
#   FRONTEND_ORIGIN=https://gesture.pages.dev ./run_gesture_control_public.sh
#
# 环境变量：
#   GESTURE_AUTH_TOKEN   不指定就在 ~/.config/gesture_control/auth_token 自动生成持久化
#   FRONTEND_ORIGIN      Cloudflare Pages / Vercel 等前端的 origin（开 CORS 用），可逗号分隔多个
#   PORT                 默认 8765
#   HOST                 默认 0.0.0.0（公开监听）

set -euo pipefail
cd "$(dirname "$0")"

# 自动生成持久化 token
TOKEN_FILE="${HOME}/.config/gesture_control/auth_token"
if [[ -z "${GESTURE_AUTH_TOKEN:-}" ]]; then
  if [[ -s "${TOKEN_FILE}" ]]; then
    export GESTURE_AUTH_TOKEN="$(cat "${TOKEN_FILE}")"
  else
    mkdir -p "$(dirname "${TOKEN_FILE}")"
    GESTURE_AUTH_TOKEN="$(openssl rand -hex 24)"
    echo "${GESTURE_AUTH_TOKEN}" > "${TOKEN_FILE}"
    chmod 600 "${TOKEN_FILE}"
    export GESTURE_AUTH_TOKEN
    echo "[info] 已生成新 token：${TOKEN_FILE}"
  fi
fi

if [[ -z "${FRONTEND_ORIGIN:-}" ]]; then
  echo "[warn] 未设 FRONTEND_ORIGIN；CORS 将关闭，前端可能跨域被拒" >&2
fi

if ! id -nG | tr ' ' '\n' | grep -qx input; then
  if getent group input | grep -q "$USER"; then
    echo "[info] 切到 input 组重启..."
    exec sg input -c "FRONTEND_ORIGIN=${FRONTEND_ORIGIN:-} GESTURE_AUTH_TOKEN=${GESTURE_AUTH_TOKEN} bash $(printf %q "$0") $*"
  fi
fi

source "${HOME}/miniconda3/bin/activate" gesture-py310

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"

echo "==================================================================="
echo " 公网模式启动"
echo "  - 后端监听：http://${HOST}:${PORT}"
echo "  - Token   ：${GESTURE_AUTH_TOKEN}"
echo "  - 前端跨域：${FRONTEND_ORIGIN:-(未设)}"
echo ""
echo "  下一步：在另一个终端跑 cloudflared tunnel run gesture-control"
echo "  浏览器访问： \${前端}/?api=https://gesture-api.example.com&token=${GESTURE_AUTH_TOKEN}"
echo "==================================================================="

CORS_ARGS=()
if [[ -n "${FRONTEND_ORIGIN:-}" ]]; then
  IFS=',' read -ra origins <<< "${FRONTEND_ORIGIN}"
  for o in "${origins[@]}"; do
    CORS_ARGS+=(--cors-origin "${o}")
  done
fi

exec python -m control_server.server \
  --host "${HOST}" --port "${PORT}" \
  --auth-token "${GESTURE_AUTH_TOKEN}" \
  "${CORS_ARGS[@]}" "$@"
