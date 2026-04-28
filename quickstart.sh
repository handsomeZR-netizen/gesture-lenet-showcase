#!/usr/bin/env bash
# 快速启动：从零到能用，一行搞定。
#
#   ./quickstart.sh
#
# 它会按顺序检查并自动修复：
#   1. Conda 环境 gesture-py310 是否存在
#   2. Python 依赖是否装齐
#   3. /dev/uinput 是否给 input 组开了读写权限
#   4. 当前用户是否在 input 组
#   5. 端口 8765 是否被占用（占用则提示）
#   6. 启动后端 + 自动开浏览器

set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
ENV_NAME="gesture-py310"
CONDA_BIN="${HOME}/miniconda3/bin/conda"
PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}/"

# --------------------------------------------------------------- 输出小工具

c_red()    { printf '\033[31m%s\033[0m' "$1"; }
c_green()  { printf '\033[32m%s\033[0m' "$1"; }
c_yellow() { printf '\033[33m%s\033[0m' "$1"; }
c_cyan()   { printf '\033[36m%s\033[0m' "$1"; }
c_bold()   { printf '\033[1m%s\033[0m' "$1"; }

step()  { echo; echo "$(c_cyan "▶") $(c_bold "$1")"; }
ok()    { echo "  $(c_green "✓") $1"; }
warn()  { echo "  $(c_yellow "⚠") $1"; }
fail()  { echo "  $(c_red "✗") $1"; exit 1; }

# --------------------------------------------------------------- 1. Conda 环境

step "检查 Conda 环境"
if [[ ! -x "${CONDA_BIN}" ]]; then
  fail "找不到 ${CONDA_BIN}。请先跑 ./install_miniconda.sh"
fi
if ! "${CONDA_BIN}" env list 2>/dev/null | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  warn "Conda 环境 ${ENV_NAME} 不存在，正在创建（约 3 分钟）..."
  bash ./install_env.sh
  ok "环境 ${ENV_NAME} 创建完毕"
else
  ok "环境 ${ENV_NAME} 已存在"
fi

# 激活环境（影响后续命令的 python）
# shellcheck disable=SC1091
source "${HOME}/miniconda3/bin/activate" "${ENV_NAME}"
ok "环境已激活：$(python --version 2>&1)"

# --------------------------------------------------------------- 2. Python 依赖

step "检查 Python 依赖"
MISSING=()
for mod in mediapipe fastapi uvicorn evdev onnx onnxruntime torch numpy pydantic; do
  if ! python -c "import ${mod}" 2>/dev/null; then
    MISSING+=("${mod}")
  fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  warn "缺少模块：${MISSING[*]}，正在 pip install..."
  pip install -q -r requirements.txt
  ok "依赖装齐"
else
  ok "依赖完整"
fi

# --------------------------------------------------------------- 3. /dev/uinput 权限

step "检查 /dev/uinput 权限（evdev 注入需要）"
if [[ ! -e /dev/uinput ]]; then
  fail "/dev/uinput 不存在，内核可能没装 uinput 模块。试试 sudo modprobe uinput"
fi
PERMS="$(stat -c '%a' /dev/uinput)"
GROUP="$(stat -c '%G' /dev/uinput)"
if [[ "${GROUP}" != "input" || ! "${PERMS}" =~ 6[0-7]$ ]]; then
  warn "/dev/uinput 权限：mode=${PERMS} group=${GROUP}，需要 input 组可写"
  echo "    将运行 sudo bash setup_uinput.sh —— 输一次密码"
  sudo bash setup_uinput.sh
  ok "权限已配置"
else
  ok "/dev/uinput  ${PERMS} root:${GROUP}"
fi

# --------------------------------------------------------------- 4. input 组

step "检查 input 组"
if ! getent group input | grep -qw "$USER"; then
  warn "用户 $USER 不在 input 组，加入中（需注销重登或用 sg 派生）..."
  sudo usermod -aG input "$USER"
  warn "已加入 input 组，但当前 shell 不会立即生效。"
  echo "    本脚本会用 sg input 临时获取 input 组身份后再启动后端。"
fi
if id -nG | tr ' ' '\n' | grep -qx input; then
  ok "当前 shell 已带 input 组"
  USE_SG=0
else
  warn "当前 shell 缺 input 组，将通过 sg input 派生进程启动"
  USE_SG=1
fi

# --------------------------------------------------------------- 5. 端口检查

step "检查端口 ${PORT}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then
  warn "端口 ${PORT} 被占用，尝试结束已有 control_server 进程..."
  pkill -9 -f 'control_server.server' 2>/dev/null || true
  sleep 1
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then
    fail "端口 ${PORT} 仍被占用，请手动 kill 占用进程或用 PORT=8766 ./quickstart.sh"
  fi
fi
ok "端口空闲"

# --------------------------------------------------------------- 6. 启动 + 开浏览器

step "启动后端 ${URL}"
echo "    日志会输出到当前终端，按 $(c_bold Ctrl-C) 退出"
echo

# 后台等服务起来后开浏览器
(
  for _ in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('${URL}api/status', timeout=1)" >/dev/null 2>&1; then
      xdg-open "${URL}" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 0.4
  done
) &

PY_BIN="${HOME}/miniconda3/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  fail "找不到环境 python：${PY_BIN}"
fi

if [[ "${USE_SG}" == "1" ]]; then
  # sg 默认用 /bin/sh 解释命令，所以这里不能用 source / activate；
  # 直接用 conda env 里的绝对路径 python，PYTHONPATH 自动找到 control_server。
  exec sg input -c "cd '${PROJECT_DIR}' && '${PY_BIN}' -m control_server.server --host '${HOST}' --port '${PORT}'"
else
  exec "${PY_BIN}" -m control_server.server --host "${HOST}" --port "${PORT}"
fi
