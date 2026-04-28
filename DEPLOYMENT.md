# 🚀 部署指南

本项目天然分成两半：

| 组件 | 能否上云 | 原因 |
|---|---|---|
| **前端** `web_control_demo/` | ✅ 完美适配 | 纯静态 HTML/JS/CSS，浏览器内推理 |
| **后端** `control_server/` | ❌ 不建议 | 用 evdev 直接写 `/dev/uinput`，必须运行在你**自己的电脑**上才能控制你的鼠标键盘 |

> 💡 「云端控制远程机器」毫无意义——容器没有显示设备，也没人愿意让公网服务操作自己电脑。
> 真正部署的方式是：**前端上云做演示**，**后端在本地真实控制**。

---

## 方案 A · Vercel（推荐用于演示 / 课堂展示）

把 `web_control_demo/` 部署成 Vercel 静态站。任何人打开 URL 都能看到识别 + UI；没运行本地后端时**自动进入「演示模式」**：手势照常识别、绑定面板照常操作，只是不真发动作。

### 步骤

```bash
# 1. 一次性安装 Vercel CLI
npm i -g vercel

# 2. 仓库根目录跑一次（首次会问几个问题）
cd gesture-lenet-showcase
vercel

# 关键回答：
#   - Set up and deploy? Y
#   - Which scope? <你的账号>
#   - Link to existing project? N
#   - Project name? gesture-control（或自取）
#   - In which directory is your code located? ./
#   - 检测到的 framework? Other

# 3. 第一次部署后，绑定 GitHub 自动持续部署
vercel git connect
```

或者直接在 https://vercel.com/new 选 GitHub 仓库 → Import → Framework Preset 选 **Other** → Root Directory 留 `./` → 部署即可（`vercel.json` 已经写好规则）。

部署后访问 `https://gesture-control-xzr.vercel.app/`，浏览器允许摄像头即可使用。

### 演示模式行为
- 顶栏 WS 状态显示「演示模式（无后端）」
- 测试模式自动开启且无法关闭（防止动作发到不存在的后端）
- 触发动作时 UI 显示 `[演示] 手势 → 动作`
- 所有识别功能（10 静态 + 4 动态手势 + 时间平滑 + reranker）照常工作

### 限制
- Vercel 静态站**不能列出子目录**，所以只能加载根目录的 `models/gesture_mlp.onnx`（默认模型）。要看多模型 UI，必须本地跑后端。
- 课堂汇报时这正好是优势：URL 一发，老师同学打开自己电脑都能看见识别效果。

---

## 方案 B · GitHub Pages（零成本备选）

```bash
# 在 .github/workflows/pages.yml 加一段（可选自动化）
# 或手动把 web_control_demo/ 推到 gh-pages 分支：
git subtree push --prefix web_control_demo origin gh-pages

# 在 GitHub 仓库 Settings → Pages → Source 选 gh-pages
```

地址：`https://handsomeZR-netizen.github.io/gesture-lenet-showcase/`

### 限制
- 比 Vercel 多一层路径前缀，需要相应调整 `app.js` 里 `fetch("/api/...")` 的 base URL。Vercel 部署不需要改。

---

## 方案 C · Railway / Fly.io（不推荐，仅供参考）

Railway 能跑 FastAPI + WebSocket，你**可以**把后端塞进 Docker 跑在 Railway。但：

1. 容器没有 `/dev/uinput`，evdev 后端会失败
2. 即便把容器加 `--privileged --device=/dev/uinput`，那也是控制 *Railway 容器* 的虚拟 uinput，跟你的电脑无关
3. 把容器当成 **手势事件中继** 也没意义——浏览器和你电脑直连本地后端更快、更安全

**结论：后端不要上 Railway/Fly/Cloud Run。** 让用户在自己电脑上跑 `./run_gesture_control.sh`。

如果你确实想要一个云端「云中继」（把 A 浏览器的手势事件转发到 B 电脑的本机后端），那是 NAT 穿透 + 信令服务的活，超出本项目范围。

---

## 方案 D · 本地 + 局域网（最实用的真实使用）

适合：你想用平板的浏览器拍手，控制书桌电脑。

```bash
# 在书桌电脑（被控端）上启动后端，监听局域网
cd gesture-lenet-showcase
HOST=0.0.0.0 ./run_gesture_control.sh

# 在平板浏览器打开
http://<书桌电脑的局域网 IP>:8765/
```

注意：
- 局域网 HTTP 下 `getUserMedia` 在 Chrome 里会被拒绝（仅 `localhost` / HTTPS 允许）
- 解决办法：在平板浏览器里把 `chrome://flags/#unsafely-treat-insecure-origin-as-secure` 加上 `http://192.168.x.y:8765`
- 或者用 `ssh -L 8765:127.0.0.1:8765 user@desktop` 反向端口转发，让平板访问 `http://127.0.0.1:8765/`

---

## 推荐组合（针对课堂展示）

```
┌──────────────────┐     发 URL 给老师同学
│   Vercel 静态站   │ ────────────────────▶
│  (演示模式)       │
└──────────────────┘

┌──────────────────┐     真实演示控制电脑
│   你的笔记本      │  ./run_gesture_control.sh
│  (本地后端)       │
└──────────────────┘
```

汇报当天：
1. 投影你笔记本的浏览器，跑本地后端 → 演示真实控制
2. 同时把 Vercel URL 写在 PPT 上 → 同学想自己玩可以打开
3. GitHub 仓库链接也写上 → 给老师看代码

---

## CI / CD

GitHub Actions 已配置（`.github/workflows/ci.yml`）：
- 每次 push 到 main 跑 `pytest tests/`（20 个用例）
- 跑 JS `node tests/test_features_js.mjs`（跨语言特征一致性）

绿色 CI 徽章可加进 README：
```
![CI](https://github.com/handsomeZR-netizen/gesture-lenet-showcase/actions/workflows/ci.yml/badge.svg)
```
