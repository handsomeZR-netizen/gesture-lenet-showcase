#!/usr/bin/env bash
# 一次性配置：让 /dev/uinput 对 input 组开放，从而让本程序能注入键鼠。
# 用法：sudo bash setup_uinput.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "请用 sudo 运行: sudo bash $0" >&2
  exit 1
fi

RULE_PATH=/etc/udev/rules.d/60-uinput.rules
RULE='KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"'

echo "[1/4] 写入 udev 规则到 ${RULE_PATH}"
echo "${RULE}" > "${RULE_PATH}"

echo "[2/4] 重载 udev 规则并触发"
udevadm control --reload-rules
udevadm trigger

echo "[3/4] 当前会话临时打开 /dev/uinput 权限（重启后由 udev 规则继续生效）"
chgrp input /dev/uinput
chmod 660 /dev/uinput

echo "[4/4] 完成。当前权限："
ls -la /dev/uinput

REAL_USER=${SUDO_USER:-$USER}
if id -nG "${REAL_USER}" | grep -qw input; then
  echo
  echo "${REAL_USER} 已经在 input 组里。直接运行 ./run_gesture_control.sh 即可。"
else
  echo
  echo "把 ${REAL_USER} 加入 input 组（必须重新登录一次才生效）："
  usermod -aG input "${REAL_USER}"
  echo "已加入。请注销重登后再启动后端。"
fi
