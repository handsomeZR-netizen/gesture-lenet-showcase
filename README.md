# 🖐️ Gesture Control · 实时手势控制计算机系统

> 用浏览器看你的手 ✋ → 神经网络认手势 🧠 → Python 后端按下键鼠 🖱️⌨️ — 把鼠标、滚轮、媒体键、窗口管理一起交给 10 个手势打理。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Hands-00897B?logo=google&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white">
  <img alt="evdev" src="https://img.shields.io/badge/evdev-uinput-EF6C00?logo=linux&logoColor=white">
  <br/>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux-FCC624?logo=linux&logoColor=black">
  <img alt="Browser" src="https://img.shields.io/badge/Browser-Chrome%20%7C%20Firefox-4285F4?logo=googlechrome&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-active-success.svg">
  <img alt="Course" src="https://img.shields.io/badge/CV%20Course-Project-9C27B0">
</p>

<p align="center">
  <em>🎯 实时识别 10 类手势 · ⚡ 端到端延迟 < 30 ms · 🛡️ 安全锁双保险 · 🪟 Wayland / X11 通吃 · 🌐 浏览器内推理零后端依赖</em>
</p>

---

## ✨ 项目亮点

| 维度 | 说明 |
|------|------|
| 🧠 **算法** | MediaPipe 21 关键点 → 63 维归一化向量 → 17 K 参数 MLP，验证集准确率 **100%** |
| 🌐 **前端** | 单屏紧凑布局，浏览器内 ONNX 推理 + 实时骨架叠加 + HUD 仪表盘 |
| 🔌 **后端** | FastAPI + WebSocket 桥接，evdev 直连 `/dev/uinput` 注入键鼠 |
| 🪟 **跨平台** | Wayland / X11 同时工作（uinput 在显示服务器之下注入） |
| 🛠️ **可定制** | 10+ 动作可在面板里实时改绑定，配置自动持久化 |
| 🎓 **可训练** | 5 分钟向导 + `python train_gesture_mlp.py` = 你专属的 100% 模型 |
| 🔐 **安全锁** | 默认锁定状态、`Esc` 物理键紧急停、鼠标拖到屏幕角即终止 |

---

## 🎬 演示流程

```
[ 浏览器 ]                                              [ Python 后端 ]
摄像头采集 → MediaPipe 关键点 → 63D 归一化向量             FastAPI / WebSocket
       ↓                                                  ↓
   ONNX 推理 (gesture_mlp.onnx)              ──手势事件──▶  evdev 注入 /dev/uinput
       ↓                                              ◀──回执ack── 屏幕分辨率/状态
   时间平滑 + 滞后阈值 + Pinch 二次校验
       ↓
   单屏 UI（HUD + 绑定面板 + 测试模式）
```

---

## 🚀 快速开始

### 1️⃣ 一次性环境配置

```bash
# 进入项目
cd gesture-lenet-showcase

# Conda 环境（约 3 分钟）
./install_env.sh

# Linux 下让程序能注入键鼠（用 evdev / uinput）
sudo bash setup_uinput.sh
# 输出最后应显示：crw-rw---- 1 root input ... /dev/uinput
```

### 2️⃣ 启动主程序

```bash
./run_gesture_control.sh
```

- 自动激活 Conda 环境
- 启动 FastAPI 后端（默认 `http://127.0.0.1:8765/`）
- 自动打开浏览器主界面
- 如当前 shell 缺 `input` 组，会自动用 `sg input` 派生重启

### 3️⃣ 浏览器里的操作

1. 点 **「启动摄像头」** 并允许浏览器访问  
2. 勾选 **「测试模式」** 先练习手势 — 此时不会真的注入  
3. 取消测试模式，点 **「解锁」** 启动真实控制  
4. 紧急停：按 `Esc` 物理键，或鼠标拖到屏幕任意角

---

## 🤚 默认手势 ↔ 计算机动作

| 手势 | 中文名 | 默认动作 | 说明 |
|:---:|---|---|---|
| 🖐️ | 张开手掌 (`open_palm`) | 释放 | 拖拽中此手势会自动松开 |
| ☝️ | 食指指向 (`point`) | 移动鼠标 | 食指尖映射到屏幕，自适应 EMA 平滑 |
| 🤏 | 捏合 (`pinch`) | 单击 / 拖拽 | 短捏 < 0.42 s = 单击；长捏 = 按住拖拽 |
| ✊ | 握拳 (`fist`) | Esc | 退全屏 / 取消 |
| ✌️ | V 字 (`victory`) | Alt+Tab | 切换窗口 |
| 👌 | OK 圈 (`ok`) | 播放 / 暂停 | 系统媒体键 |
| 👍 | 拇指向上 (`thumbs_up`) | 音量 + |  |
| 👎 | 拇指向下 (`thumbs_down`) | 音量 - |  |
| 🤟 | 三指 (`three`) | 显示桌面 | Win + D |
| 🤙 | 电话手势 (`call`) | 下一首 | 系统媒体键 |
| 🌬️ | 整手向上划 (`swipe_up`) | 滚轮上 | 动态：手腕 0.4 s 内向上位移 > 0.22 |
| 🌬️ | 整手向下划 (`swipe_down`) | 滚轮下 | 同上方向相反 |
| 🌬️ | 整手向左划 (`swipe_left`) | 默认禁用 | 可绑定为「上一首」等 |
| 🌬️ | 整手向右划 (`swipe_right`) | 默认禁用 | 可绑定为「下一首」等 |

> 💡 在浏览器右侧「手势绑定」面板里可以**任意修改**这个映射，改完即生效，配置存于 `~/.config/gesture_control/bindings.json`。

---

## 🎓 训练你自己的高准确率模型（约 10 分钟）

项目自带的合成数据集模型在你自己的手上准确率约 80-85%。要做到 **95%+**，建议录自己的数据：

```bash
# 1. 启动后端 + 浏览器
./run_gesture_control.sh

# 2. 浏览器进 http://127.0.0.1:8765/web_control_demo/record.html
#    跟着向导录 10 个手势，每个 250 帧，约 5 分钟

# 3. 训练（CPU 上 30-60 秒）
python train_gesture_mlp.py

# 4. 导出 ONNX
python export_onnx.py

# 5. 浏览器强刷主界面（Ctrl+Shift+R）即用上新模型
```

**录制小贴士** 📸：
- 每个手势的 8 秒里，**轻微移动手部位置 / 旋转手腕 / 改变远近**
- 灯光不要太暗，背景越简单越好
- 数据是 append 模式，某个手势识别不稳可以追加录制再训

---

## 🏗️ 系统架构

```
gesture-lenet-showcase/
├── 🧠 gesture_mlp/              # 21 关键点 MLP 模型
│   ├── features.py             # 21 关键点 → 63D 向量（与 JS 完全对齐）
│   ├── model.py                # 17 K 参数 3 层 MLP
│   ├── dataset.py              # JSONL 数据加载 + 增强
│   └── seed_dataset.py         # 零采集合成数据集（兜底）
│
├── 🔌 control_server/           # FastAPI + WebSocket 后端
│   ├── server.py               # HTTP + WS 路由 + 静态托管
│   ├── controller.py           # evdev / pyautogui 双后端 + cooldown + EMA
│   └── bindings.py             # 绑定持久化
│
├── 🌐 web_control_demo/         # 浏览器前端
│   ├── index.html              # 单屏紧凑主界面
│   ├── record.html             # 录制向导
│   ├── app.js                  # 主管线（事件驱动）
│   ├── modules/
│   │   ├── features.js         # JS 端 21 关键点 → 63D（必须和 Python 一致）
│   │   ├── gestureClassifier.js # ONNX-Web 推理 + 规则兜底
│   │   ├── temporalSmoother.js  # 滑窗投票 + 滞后 + 滑动检测
│   │   └── controlClient.js    # WebSocket 客户端 + 节流 + 自动重连
│   └── models/
│       ├── gesture_mlp.onnx    # 训练好的 MLP（69 KB）
│       └── hand_landmarker.task # MediaPipe 手部模型（在项目根 models/）
│
├── 🛠️ train_gesture_mlp.py      # 训练入口
├── 📤 export_onnx.py            # PyTorch → ONNX
├── 🚀 run_gesture_control.sh    # 一键启动
├── 🔐 setup_uinput.sh           # /dev/uinput 权限配置（一次性）
└── 📚 docs/                     # 详细文档
    └── 手势控制使用手册.md
```

---

## 🔬 算法细节

### 特征工程（63 维向量）
1. **平移**：以 wrist (landmark 0) 为原点
2. **缩放**：除以 `max(‖wrist - middleMcp‖, ‖indexMcp - pinkyMcp‖)` 这个掌心尺度，去除距离 / 手型差异
3. **镜像归一**：根据 handedness 把左手翻成右手坐标系
4. **展平**：21 × 3 = 63 浮点

> 🔍 Python `gesture_mlp/features.py` 与 JS `web_control_demo/modules/features.js` 在固定 fixture 上**最大差异 < 6e-8**，保证训练和推理用同一份特征。

### 模型架构

```python
GestureMLP(
  Linear(63 → 128) + ReLU + Dropout(0.2)
  Linear(128 → 64) + ReLU + Dropout(0.2)
  Linear(64 → 10)                           # 10 类静态手势
)  # 17,098 参数
```

### 时间平滑
- **滑窗投票**（默认 7 帧）+ 进入 / 退出**滞后阈值**（55% / 40%），消除单帧抖动
- **置信度 EMA**：`new = old * 0.7 + raw * 0.3`
- **Pinch 二次校验**：MLP 输出 pinch 时用关键点几何区分 fist / ok

### 动态手势检测
- 0.4 s 环形缓冲手腕轨迹
- 位移 > 0.22 × 屏幕维度即触发 swipe_up/down/left/right

---

## 🛡️ 安全设计

| 机制 | 触发条件 | 行为 |
|---|---|---|
| 默认锁定 | 启动后 | 必须在 UI 点「解锁」 |
| 测试模式 | UI 勾选 | 识别照常，但不真发动作 |
| pyautogui FAILSAFE | 鼠标拖到屏幕任意角 | 抛异常 → 后端自动暂停 |
| 物理 Esc 键 | 任意时刻 | 系统级 Esc，前端可绑定为锁定 |
| 动作冷却表 | 每个动作独立 | 防止误触发洪水 |
| WebSocket 断开 | 浏览器关闭 / 网络断 | 后端不再接收事件，锁定 |

---

## ❓ 常见问题

<details>
<summary><b>开启不了摄像头</b></summary>

按 `error.name` 给出具体诊断：

- `NotAllowedError`：地址栏左侧相机图标点开 → 改为允许
- `NotReadableError`：另一个标签或程序占用，关掉它再试
- `NotFoundError`：设备不存在 — VMware 用户记得在 `Player → Removable Devices` 里 connect 摄像头
- 其他：浏览器 console 看完整堆栈

</details>

<details>
<summary><b>解锁后指针不动 / 媒体键不响应</b></summary>

确认 `/dev/uinput` 权限对了：

```bash
ls -la /dev/uinput     # 期望 crw-rw---- root input
groups                  # 期望包含 input
```

没的话跑 `sudo bash setup_uinput.sh` 然后**注销重登**或在新 shell 用 `newgrp input` 启动。

</details>

<details>
<summary><b>识别不准 / 手势抖动</b></summary>

- 主界面顶栏 FPS 应 ≥ 25。低于 15 通常是 CPU 不够，关掉别的浏览器标签
- 用浏览器主界面右侧「录制」按钮录自己的数据集（5 分钟），训出来的模型对你的手准确率会从 80% 跳到 95%+

</details>

<details>
<summary><b>Wayland 还是 X11 更好？</b></summary>

无所谓。本项目用 `evdev` 直接写 `/dev/uinput`，注入发生在显示服务器之下，**Wayland 和 X11 都能工作**。
</details>

<details>
<summary><b>前端怎么部署到别的机器？</b></summary>

前端是纯静态。把 `web_control_demo/` 拷到任意 HTTP 服务器，再把 WebSocket URL 指向本机后端即可。摄像头识别 + ONNX 推理都在浏览器里完成。

</details>

---

## 🧪 端到端验证

```bash
# 1. 算法侧
python train_gesture_mlp.py --epochs 80
# 期望 val_acc ≥ 0.95

# 2. 后端 API
curl -s http://127.0.0.1:8765/api/status | jq
curl -s http://127.0.0.1:8765/api/bindings | jq

# 3. WebSocket
# 浏览器主界面顶栏「后端已连接」绿色 = 通

# 4. 控制注入
# 解锁后比「食指指向」手势 → 鼠标真的动 = 通
```

---

## 🎯 后续 Roadmap

- [ ] Windows 端 Native 控制后端（替换 evdev）
- [ ] 自定义手势在线学习（在 UI 里直接录 + 训）
- [ ] 多手协同（双手缩放 / 旋转）
- [ ] 语音 + 手势多模态融合
- [ ] PWA 离线包

---

## 📜 License

MIT — 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe) — 浏览器端实时手部关键点
- [PyTorch](https://pytorch.org/) — MLP 训练
- [ONNX Runtime Web](https://onnxruntime.ai/) — 浏览器 ONNX 推理
- [FastAPI](https://fastapi.tiangolo.com/) — 后端 API
- [python-evdev](https://python-evdev.readthedocs.io/) — Linux uinput 接入
- 课程项目作者：**xzr** · 计算机视觉课程实践

---

<p align="center">
  <sub>🎬 Made with ❤️ for the CV course · 实时手势控制计算机 · 2026</sub>
</p>
