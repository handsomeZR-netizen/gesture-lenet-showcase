// Main orchestration for the gesture control center.
//
// Pipeline per frame:
//   1. MediaPipe Tasks Vision detects 21 hand landmarks
//   2. gestureClassifier scores them with the trained MLP (or rule fallback)
//   3. LabelSmoother + SwipeDetector produce a stable label / dynamic event
//   4. controlClient sends the gesture event to the Python backend over WS
//   5. UI updates: skeleton overlay, telemetry, bindings panel state, action bubble

import { ControlClient } from "./modules/controlClient.js";
import {
  classifyByRule,
  classifyLandmarks,
  isModelLoaded,
  loadGestureModel,
} from "./modules/gestureClassifier.js";
import { LabelSmoother, SwipeDetector } from "./modules/temporalSmoother.js";

const MP_VERSION = "0.10.33";
const MP_VISION_MODULE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL_PATH = "../models/hand_landmarker.task";
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const GESTURE_LABELS_CN = {
  open_palm: "张开手掌",
  point: "食指指向",
  pinch: "捏合",
  fist: "握拳",
  victory: "V 字",
  ok: "OK 圈",
  thumbs_up: "拇指向上",
  thumbs_down: "拇指向下",
  three: "三指",
  call: "电话手势",
  swipe_up: "整手向上划",
  swipe_down: "整手向下划",
  swipe_left: "整手向左划",
  swipe_right: "整手向右划",
};

// 给绑定行右下角的提示文案。静态手势靠 MLP 直接分类；动态手势靠 0.4s 内
// 手腕的位移轨迹检测（像在镜头前快速划一下）。
const GESTURE_HINTS_CN = {
  open_palm: "五指自然张开",
  point: "仅食指竖起",
  pinch: "拇指食指轻轻捏在一起",
  fist: "整只手握拳",
  victory: "食指中指 V 字",
  ok: "拇指食指捏成圈，其他三指张开",
  thumbs_up: "握拳后竖大拇指",
  thumbs_down: "拇指朝下",
  three: "食指中指无名指竖起",
  call: "拇指与小指伸出",
  swipe_up: "动态：在镜头前把手快速向上划过",
  swipe_down: "动态：在镜头前把手快速向下划过",
  swipe_left: "动态：在镜头前把手快速向左划过",
  swipe_right: "动态：在镜头前把手快速向右划过",
};

const STATIC_LABELS = [
  "open_palm",
  "point",
  "pinch",
  "fist",
  "victory",
  "ok",
  "thumbs_up",
  "thumbs_down",
  "three",
  "call",
];

const DYNAMIC_LABELS = ["swipe_up", "swipe_down", "swipe_left", "swipe_right"];

const ui = {
  startButton: document.getElementById("startButton"),
  unlockButton: document.getElementById("unlockButton"),
  pauseButton: document.getElementById("pauseButton"),
  testMode: document.getElementById("testMode"),
  cameraState: document.getElementById("cameraState"),
  controlState: document.getElementById("controlState"),
  wsState: document.getElementById("wsState"),
  warningStrip: document.getElementById("warningStrip"),
  cameraOverlay: document.getElementById("cameraOverlay"),
  overlayRetry: document.getElementById("overlayRetry"),
  actionBubble: document.getElementById("actionBubble"),
  gestureLabel: document.getElementById("gestureLabel"),
  confidenceBar: document.getElementById("confidenceBar"),
  confidenceValue: document.getElementById("confidenceValue"),
  fpsValue: document.getElementById("fpsValue"),
  latencyValue: document.getElementById("latencyValue"),
  handsValue: document.getElementById("handsValue"),
  uptimeValue: document.getElementById("uptimeValue"),
  lastActionValue: document.getElementById("lastActionValue"),
  screenSizeValue: document.getElementById("screenSizeValue"),
  cursorValue: document.getElementById("cursorValue"),
  triggerCountValue: document.getElementById("triggerCountValue"),
  bindingsList: document.getElementById("bindingsList"),
};

const video = document.getElementById("cameraFeed");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayContext = overlayCanvas.getContext("2d");

const state = {
  handLandmarker: null,
  videoReady: false,
  runtimeStarted: false,
  lastVideoTime: -1,
  fps: 0,
  latencyMs: 0,
  uptimeStartedAt: 0,
  triggerCount: 0,
  smoothedLabel: null,
  smoothedConfidence: 0,
  bindings: {},
  actions: [],
  testMode: false,
  pendingActionLabel: null,
  controlSnapshot: null,
};

const smoother = new LabelSmoother();
const swipe = new SwipeDetector();
const controlClient = new ControlClient({ url: deriveWsUrl() });

function deriveWsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/control`;
}

function setStatusPill(el, text, kind) {
  el.textContent = text;
  el.className = `status-pill status-${kind}`;
}

function showBubble(message, kind = "info") {
  ui.actionBubble.textContent = message;
  ui.actionBubble.dataset.kind = kind;
  ui.actionBubble.classList.add("visible");
  clearTimeout(ui.actionBubble._hideTimer);
  ui.actionBubble._hideTimer = setTimeout(() => {
    ui.actionBubble.classList.remove("visible");
  }, 1100);
}

function setOverlayMessage(text, { showRetry = false } = {}) {
  if (!text) {
    ui.cameraOverlay.hidden = true;
    if (ui.overlayRetry) ui.overlayRetry.hidden = true;
    return;
  }
  ui.cameraOverlay.hidden = false;
  ui.cameraOverlay.querySelector(".overlay-title").textContent = text.title || "提示";
  ui.cameraOverlay.querySelector(".overlay-body").textContent = text.body || "";
  if (ui.overlayRetry) ui.overlayRetry.hidden = !showRetry;
}

function setWarning(message) {
  if (!message) {
    ui.warningStrip.hidden = true;
    return;
  }
  ui.warningStrip.textContent = message;
  ui.warningStrip.hidden = false;
}

async function loadVisionModule() {
  try {
    const mod = await import(/* @vite-ignore */ MP_VISION_MODULE);
    return mod;
  } catch (error) {
    console.error("failed to load mediapipe vision", error);
    setOverlayMessage({
      title: "加载手部模型失败",
      body: "请检查网络是否能访问 jsdelivr CDN。",
    });
    throw error;
  }
}

async function ensureHandLandmarker() {
  if (state.handLandmarker) return state.handLandmarker;
  const mod = await loadVisionModule();
  const fileset = await mod.FilesetResolver.forVisionTasks(MP_WASM_ROOT);
  state.handLandmarker = await mod.HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL_PATH },
    numHands: 1,
    runningMode: "VIDEO",
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.5,
  });
  return state.handLandmarker;
}

function describeCameraError(error) {
  if (!error) return { title: "无法启动摄像头", body: "未知错误" };
  const name = error.name || "";
  const msg = error.message || String(error);
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "摄像头权限被拒绝",
        body:
          "请点击地址栏左侧的相机图标，把权限改回「允许」，然后点重试。" +
          "（系统层面：检查 GNOME 设置 → 隐私 → 摄像头是否开启。）",
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return {
        title: "找不到可用摄像头",
        body: "请确认有摄像头设备，且未被其他程序占用（如 Zoom/录制软件）。",
      };
    case "NotReadableError":
      return {
        title: "摄像头被占用",
        body:
          "另一个程序或浏览器标签正在使用摄像头。" +
          "请关掉其他用到摄像头的程序（包括本网页其他标签、record.html）后重试。",
      };
    case "AbortError":
      return {
        title: "摄像头启动被中断",
        body: "刚才的启动被打断；点重试再试一次。",
      };
    default:
      return {
        title: "无法启动摄像头",
        body: `${name || "Error"}: ${msg}`,
      };
  }
}

async function startCamera() {
  if (state.runtimeStarted) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setOverlayMessage({
      title: "浏览器不支持 getUserMedia",
      body: "请使用最新的 Chrome / Firefox / Edge，并确认是从 http://127.0.0.1 访问（非 file:// 或外网 IP）。",
    });
    return;
  }
  setOverlayMessage({ title: "正在启动摄像头", body: "请允许浏览器访问摄像头。" });
  ui.startButton.disabled = true;
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    state.videoReady = true;
    setStatusPill(ui.cameraState, "摄像头已启动", "active");
    syncOverlaySize();
  } catch (error) {
    console.error("getUserMedia failed", error);
    setStatusPill(ui.cameraState, "摄像头错误", "warn");
    const detail = describeCameraError(error);
    setOverlayMessage(detail, { showRetry: true });
    ui.startButton.disabled = false;
    return;
  }

  setOverlayMessage({ title: "加载手部模型…", body: "约 2 秒，请稍候。" });
  try {
    await ensureHandLandmarker();
  } catch (error) {
    setOverlayMessage(
      {
        title: "手部模型加载失败",
        body: `${error?.name || "Error"}: ${error?.message || error}. 请检查网络（jsdelivr CDN）。`,
      },
      { showRetry: true },
    );
    ui.startButton.disabled = false;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    state.videoReady = false;
    return;
  }

  // Try to load the trained MLP. Failure is non-fatal — we fall back to rules.
  try {
    await loadGestureModel({
      modelUrl: "models/gesture_mlp.onnx",
      metaUrl: "models/gesture_mlp.meta.json",
    });
  } catch (error) {
    console.warn("gesture MLP not loaded; using rule-based classifier", error);
    showBubble("MLP 未就绪，已降级到规则识别", "warn");
  }

  state.runtimeStarted = true;
  state.uptimeStartedAt = performance.now();
  setOverlayMessage(null);
  ui.startButton.textContent = "运行中";
  ui.startButton.disabled = true;
  requestAnimationFrame(loop);
}

function syncOverlaySize() {
  const rect = video.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  overlayCanvas.width = Math.round(rect.width * dpr);
  overlayCanvas.height = Math.round(rect.height * dpr);
  overlayCanvas.style.width = `${rect.width}px`;
  overlayCanvas.style.height = `${rect.height}px`;
  overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawSkeleton(landmarks) {
  const ctx = overlayContext;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!landmarks) return;
  const w = overlayCanvas.width / (window.devicePixelRatio || 1);
  const h = overlayCanvas.height / (window.devicePixelRatio || 1);
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "rgba(61, 215, 194, 0.85)";
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
  }
  ctx.stroke();
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.fillStyle = "#ffd66b";
    ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function classify(landmarks, handedness) {
  let result;
  if (isModelLoaded()) {
    result = await classifyLandmarks(landmarks, handedness);
  } else {
    result = classifyByRule(landmarks);
  }
  return rerankPinchVsFistOk(result, landmarks);
}

// MLP 经常把「握拳」和「OK 圈」识别成「捏合」（拇指食指都靠在一起）。
// 这里用关键点几何在 pinch 输出上做二次判定：
//   * 中无小三指都收拢 → 改判为 fist
//   * 中无小三指都伸展 → 改判为 ok
//   * 中无小三指部分伸展 → 保留 pinch
function rerankPinchVsFistOk(result, landmarks) {
  if (!result || result.label !== "pinch") return result;
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const palmScale = Math.max(
    Math.hypot(wrist.x - middleMcp.x, wrist.y - middleMcp.y),
    1e-4,
  );
  const isExtended = (tipIdx, pipIdx) => {
    const tip = landmarks[tipIdx];
    const pip = landmarks[pipIdx];
    return tip.y < pip.y - 0.02;
  };
  const middleExt = isExtended(12, 10);
  const ringExt = isExtended(16, 14);
  const pinkyExt = isExtended(20, 18);
  const extOther = (middleExt ? 1 : 0) + (ringExt ? 1 : 0) + (pinkyExt ? 1 : 0);

  const tipsAvgToWrist =
    [8, 12, 16, 20]
      .map((i) =>
        Math.hypot(landmarks[i].x - wrist.x, landmarks[i].y - wrist.y) / palmScale,
      )
      .reduce((a, b) => a + b, 0) / 4;

  // 全部三指收拢 + 所有 tip 离 wrist 都很近 → 握拳
  if (extOther === 0 && tipsAvgToWrist < 1.5) {
    return { ...result, label: "fist", confidence: Math.max(result.confidence, 0.8) };
  }
  // 中无小都伸展 → 拇指食指捏圈 + 其他三指张开 = OK
  if (extOther >= 2) {
    return { ...result, label: "ok", confidence: Math.max(result.confidence, 0.8) };
  }
  // 否则保持 pinch
  return result;
}

function deriveAnchor(landmarks) {
  const indexTip = landmarks[8];
  const thumbTip = landmarks[4];
  const x = (indexTip.x + thumbTip.x) * 0.5;
  const y = (indexTip.y + thumbTip.y) * 0.5;
  return { x, y };
}

function updateUiTelemetry(detection) {
  ui.handsValue.textContent = detection ? "1" : "0";
  ui.latencyValue.textContent = `${state.latencyMs.toFixed(0)} ms`;
  ui.fpsValue.textContent = state.fps.toFixed(1);
  ui.uptimeValue.textContent = `${((performance.now() - state.uptimeStartedAt) / 1000).toFixed(0)} s`;
}

function updateGestureUi(label, confidence) {
  if (!label) {
    ui.gestureLabel.textContent = "等待手势";
    ui.confidenceBar.style.width = "0%";
    ui.confidenceValue.textContent = "0%";
    return;
  }
  ui.gestureLabel.textContent = GESTURE_LABELS_CN[label] || label;
  const pct = Math.round(confidence * 100);
  ui.confidenceBar.style.width = `${pct}%`;
  ui.confidenceValue.textContent = `${pct}%`;
}

function applyControlSnapshot(snapshot) {
  if (!snapshot) return;
  state.controlSnapshot = snapshot;
  let kind = "idle";
  let text = "未连接";
  switch (snapshot.state) {
    case "active":
      kind = "active";
      text = "控制已开启";
      break;
    case "locked":
      kind = "warn";
      text = "已锁定";
      break;
    case "paused":
      kind = "warn";
      text = "已暂停";
      break;
    case "wayland-blocked":
      kind = "warn";
      text = "Wayland 会话不支持";
      break;
    case "fallback":
      kind = "warn";
      text = "pyautogui 不可用";
      break;
  }
  setStatusPill(ui.controlState, text, kind);
  ui.lastActionValue.textContent = snapshot.last_action || "—";
  if (snapshot.warning) {
    setWarning(snapshot.warning);
  } else {
    setWarning(null);
  }
  if (snapshot.screen_size && snapshot.screen_size[0]) {
    ui.screenSizeValue.textContent = `${snapshot.screen_size[0]} × ${snapshot.screen_size[1]}`;
  }
  if (snapshot.cursor) {
    ui.cursorValue.textContent = `(${snapshot.cursor[0]}, ${snapshot.cursor[1]})`;
  }
}

let lastFrameTimestamp = 0;
let lastInferenceStart = 0;

async function loop() {
  if (!state.runtimeStarted) return;
  syncOverlaySize();
  const now = performance.now();
  if (lastFrameTimestamp > 0) {
    const dt = now - lastFrameTimestamp;
    state.fps = state.fps * 0.85 + (1000 / Math.max(dt, 1)) * 0.15;
  }
  lastFrameTimestamp = now;

  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    lastInferenceStart = performance.now();
    let result;
    try {
      result = state.handLandmarker.detectForVideo(video, now);
    } catch (error) {
      console.warn("detectForVideo failed", error);
    }
    state.latencyMs =
      state.latencyMs * 0.7 + (performance.now() - lastInferenceStart) * 0.3;

    if (result && result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      const handedness =
        result.handednesses?.[0]?.[0]?.categoryName || "Right";
      drawSkeleton(landmarks);

      let prediction;
      try {
        prediction = await classify(landmarks, handedness);
      } catch (error) {
        prediction = classifyByRule(landmarks);
      }

      const smoothed = smoother.push(prediction.label, prediction.confidence);
      state.smoothedLabel = smoothed.label;
      state.smoothedConfidence = smoothed.confidence;
      updateGestureUi(smoothed.label, smoothed.confidence);

      // dynamic swipe detection runs in parallel with static recognition
      const wrist = landmarks[0];
      const swipeLabel = swipe.observe({ x: wrist.x, y: wrist.y }, now);
      if (swipeLabel) {
        dispatchEvent(swipeLabel, 0.85, deriveAnchor(landmarks), handedness);
      }

      if (smoothed.label) {
        dispatchEvent(
          smoothed.label,
          smoothed.confidence,
          deriveAnchor(landmarks),
          handedness,
        );
      }
      updateUiTelemetry({});
    } else {
      drawSkeleton(null);
      smoother.reset();
      swipe.reset();
      state.smoothedLabel = null;
      updateGestureUi(null, 0);
      updateUiTelemetry(null);
      // tell the backend "no hand" so it releases drag state etc.
      controlClient.sendGesture({
        label: "open_palm",
        confidence: 0.0,
        anchor: null,
        handedness: "Right",
        action: "release",
      });
    }
  }

  requestAnimationFrame(loop);
}

function dispatchEvent(label, confidence, anchor, handedness) {
  const binding = state.bindings[label];
  if (!binding || !binding.enabled) return;
  if (state.testMode) {
    const cn = GESTURE_LABELS_CN[label] || label;
    showBubble(`[测试] ${cn} → ${actionLabel(binding.action)}`, "info");
    state.triggerCount += 1;
    ui.triggerCountValue.textContent = String(state.triggerCount);
    return;
  }
  const sent = controlClient.sendGesture({
    label,
    confidence,
    anchor,
    handedness,
    action: binding.action,
  });
  if (sent) {
    state.triggerCount += 1;
    ui.triggerCountValue.textContent = String(state.triggerCount);
  }
}

function actionLabel(action) {
  return state.actions.find((a) => a.name === action)?.label || action;
}

function renderBindingsPanel() {
  const list = ui.bindingsList;
  list.innerHTML = "";
  for (const label of [...STATIC_LABELS, ...DYNAMIC_LABELS]) {
    const binding = state.bindings[label] || { action: "noop", enabled: false };
    const row = document.createElement("div");
    row.className = "binding-row";

    const nameBlock = document.createElement("div");
    nameBlock.className = "binding-name";
    const cn = document.createElement("p");
    cn.className = "binding-cn";
    cn.textContent = GESTURE_LABELS_CN[label] || label;
    const hint = document.createElement("p");
    hint.className = "binding-code";
    hint.textContent = GESTURE_HINTS_CN[label] || label;
    nameBlock.title = `${GESTURE_LABELS_CN[label] || label}\n${GESTURE_HINTS_CN[label] || ""}`;
    nameBlock.appendChild(cn);
    nameBlock.appendChild(hint);

    const select = document.createElement("select");
    select.className = "binding-select";
    for (const action of state.actions) {
      const opt = document.createElement("option");
      opt.value = action.name;
      opt.textContent = action.label;
      if (action.name === binding.action) opt.selected = true;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      state.bindings[label] = { ...binding, action: select.value };
      pushBindings();
    });

    const toggle = document.createElement("label");
    toggle.className = "binding-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = !!binding.enabled;
    checkbox.addEventListener("change", () => {
      state.bindings[label] = { ...binding, enabled: checkbox.checked };
      pushBindings();
    });
    const toggleText = document.createElement("span");
    toggleText.textContent = "启用";
    toggle.appendChild(checkbox);
    toggle.appendChild(toggleText);

    row.appendChild(nameBlock);
    row.appendChild(select);
    row.appendChild(toggle);
    list.appendChild(row);
  }
}

async function pushBindings() {
  try {
    const payload = {};
    for (const [label, spec] of Object.entries(state.bindings)) {
      payload[label] = { action: spec.action, enabled: !!spec.enabled };
    }
    await fetch("/api/bindings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bindings: payload }),
    });
  } catch (error) {
    console.warn("failed to push bindings", error);
  }
}

async function loadInitialState() {
  try {
    const [statusResp, actionsResp] = await Promise.all([
      fetch("/api/status").then((r) => r.json()),
      fetch("/api/actions").then((r) => r.json()),
    ]);
    state.bindings = statusResp.bindings || {};
    state.actions = actionsResp.actions || [];
    applyControlSnapshot({
      state: statusResp.state,
      last_action: statusResp.last_action,
      warning: statusResp.warning,
      screen_size: statusResp.screen_size,
      cursor: statusResp.cursor,
    });
    renderBindingsPanel();
  } catch (error) {
    setStatusPill(ui.controlState, "后端不可达", "warn");
    setWarning("无法连接 Python 后端。请确认运行了 ./run_gesture_control.sh。");
  }
}

function bindControlClientEvents() {
  controlClient.addEventListener("connect", () => {
    setStatusPill(ui.wsState, "后端已连接", "active");
  });
  controlClient.addEventListener("disconnect", () => {
    setStatusPill(ui.wsState, "后端断开，重连中", "warn");
  });
  controlClient.addEventListener("error", () => {
    setStatusPill(ui.wsState, "WS 出错", "warn");
  });
  controlClient.addEventListener("message", (event) => {
    const data = event.detail;
    if (!data) return;
    if (data.type === "ack") {
      if (data.ok && data.action !== "noop" && data.action !== "move_cursor") {
        showBubble(data.message || data.action, "ok");
      } else if (!data.ok && data.message) {
        // only show explicit failure messages (skip noop spam)
        if (!/未绑定|noop/.test(data.message)) {
          showBubble(data.message, "warn");
        }
      }
      applyControlSnapshot({
        state: data.control_state,
        last_action: data.message,
        warning: data.warning,
        cursor: data.cursor,
        screen_size: state.controlSnapshot?.screen_size,
      });
    } else if (data.type === "control_state") {
      applyControlSnapshot({
        state: data.state,
        last_action: data.last_action,
        warning: data.warning,
        screen_size: state.controlSnapshot?.screen_size,
      });
    }
  });
}

function bindUi() {
  ui.startButton.addEventListener("click", startCamera);
  if (ui.overlayRetry) {
    ui.overlayRetry.addEventListener("click", () => {
      if (state.runtimeStarted) return;
      startCamera();
    });
  }
  ui.unlockButton.addEventListener("click", () => {
    fetch("/api/control/unlock", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        applyControlSnapshot({
          state: data.state,
          last_action: data.last_action,
          screen_size: state.controlSnapshot?.screen_size,
        });
        showBubble(data.last_action || "已解锁", "ok");
      })
      .catch(() => showBubble("解锁请求失败", "warn"));
  });
  ui.pauseButton.addEventListener("click", () => {
    fetch("/api/control/toggle_pause", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        applyControlSnapshot({
          state: data.state,
          last_action: data.last_action,
          screen_size: state.controlSnapshot?.screen_size,
        });
        showBubble(data.last_action || "切换暂停", "info");
      })
      .catch(() => showBubble("暂停请求失败", "warn"));
  });
  ui.testMode.addEventListener("change", () => {
    state.testMode = ui.testMode.checked;
    showBubble(state.testMode ? "测试模式：动作不会真实发送" : "测试模式关闭", "info");
  });
}

(async function main() {
  bindUi();
  bindControlClientEvents();
  controlClient.connect();
  await loadInitialState();
  state.uptimeStartedAt = performance.now();
})();
