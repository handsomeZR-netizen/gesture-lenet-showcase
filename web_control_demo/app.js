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
  modelSelect: document.getElementById("modelSelect"),
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
  models: [],
  activeModelName: null,
  demoMode: false,
  wsAttempts: 0,
};

const smoother = new LabelSmoother();
const swipe = new SwipeDetector();

// 从 URL 读取自定义后端地址 + token，方便云端前端连本地 Tunnel。
//   ?api=https://gesture-api.example.com         覆盖后端 URL
//   ?token=xxxx                                  附加 auth token
//   也支持 localStorage 持久化（输入一次后存住）
const params = new URLSearchParams(window.location.search);
const apiOverride =
  params.get("api") || localStorage.getItem("gesture_api_base") || "";
const authToken =
  params.get("token") || localStorage.getItem("gesture_auth_token") || "";
if (params.get("api")) localStorage.setItem("gesture_api_base", apiOverride);
if (params.get("token")) localStorage.setItem("gesture_auth_token", authToken);

function apiBase() {
  if (apiOverride) return apiOverride.replace(/\/$/, "");
  return ""; // same-origin
}

function buildApiUrl(path) {
  const base = apiBase();
  const sep = path.includes("?") ? "&" : "?";
  const tokenPart = authToken ? `${sep}token=${encodeURIComponent(authToken)}` : "";
  return `${base}${path}${tokenPart}`;
}

function fetchApi(path, init = {}) {
  const headers = { ...(init.headers || {}) };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  return fetch(buildApiUrl(path), { ...init, headers });
}

function deriveWsUrl() {
  const base = apiBase();
  let wsBase;
  if (base) {
    wsBase = base.replace(/^http/, "ws");
  } else {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    wsBase = `${protocol}//${window.location.host}`;
  }
  const tokenPart = authToken ? `?token=${encodeURIComponent(authToken)}` : "";
  return `${wsBase}/ws/control${tokenPart}`;
}

const controlClient = new ControlClient({ url: deriveWsUrl() });

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

  await loadActiveGestureModel({ initial: true });

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
  result = rerankPinchVsFistOk(result, landmarks);
  result = rerankFistVsThumbs(result, landmarks);
  return result;
}

// MLP 经常把「拇指向上 / 拇指向下」识别成「握拳」（四指都收拢，外形相近）。
// 这里在 fist 输出上做二次判定：如果拇指 tip 明显高于其他四指 → thumbs_up，
// 明显低于其他四指 → thumbs_down，否则保留 fist。
function rerankFistVsThumbs(result, landmarks) {
  if (!result || result.label !== "fist") return result;
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const palmScale = Math.max(
    Math.hypot(wrist.x - middleMcp.x, wrist.y - middleMcp.y),
    1e-4,
  );
  const thumbTip = landmarks[4];
  const otherTips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]];
  const minOtherY = Math.min(...otherTips.map((p) => p.y));
  const maxOtherY = Math.max(...otherTips.map((p) => p.y));
  // 拇指相对其他指尖的纵向偏移，按手掌尺度归一化
  const upGap = (minOtherY - thumbTip.y) / palmScale;
  const downGap = (thumbTip.y - maxOtherY) / palmScale;
  // 拇指还得离掌心有距离（不是收在拳心里），避免把真正的握拳误判
  const thumbExtension =
    Math.hypot(thumbTip.x - wrist.x, thumbTip.y - wrist.y) / palmScale;

  if (upGap > 0.35 && thumbExtension > 1.05) {
    return { ...result, label: "thumbs_up", confidence: Math.max(result.confidence, 0.78) };
  }
  if (downGap > 0.35 && thumbExtension > 1.05) {
    return { ...result, label: "thumbs_down", confidence: Math.max(result.confidence, 0.78) };
  }
  return result;
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
  if (state.testMode || state.demoMode) {
    const cn = GESTURE_LABELS_CN[label] || label;
    const tag = state.demoMode ? "[演示]" : "[测试]";
    showBubble(`${tag} ${cn} → ${actionLabel(binding.action)}`, "info");
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
    await fetchApi("/api/bindings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bindings: payload }),
    });
  } catch (error) {
    console.warn("failed to push bindings", error);
  }
}

async function loadAvailableModels() {
  try {
    const data = await fetchApi("/api/models").then((r) => r.json());
    state.models = data.models || [];
    state.activeModelName = data.active || (state.models[0]?.name ?? null);
    renderModelSelect();
  } catch (error) {
    // Vercel/static-only deployment: list known models by probing meta files.
    state.models = await probeStaticModels();
    state.activeModelName = state.models[0]?.name || null;
    renderModelSelect();
  }
}

async function probeStaticModels() {
  // Try the flat default first; subdirectories cannot be enumerated without a
  // backend, so static deployments will only see the default model.
  const candidates = [
    {
      name: "default",
      display_name: "default",
      model_url: "models/gesture_mlp.onnx",
      meta_url: "models/gesture_mlp.meta.json",
    },
  ];
  const found = [];
  for (const c of candidates) {
    try {
      const r = await fetch(c.meta_url, { method: "HEAD" });
      if (r.ok) found.push(c);
    } catch {
      // ignore
    }
  }
  return found;
}

function renderModelSelect() {
  const sel = ui.modelSelect;
  if (!sel) return;
  sel.innerHTML = "";
  if (state.models.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "（无可用模型）";
    opt.disabled = true;
    sel.appendChild(opt);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.display_name || m.name}${m.name === state.activeModelName ? " ✓" : ""}`;
    if (m.name === state.activeModelName) opt.selected = true;
    sel.appendChild(opt);
  }
}

async function loadActiveGestureModel({ initial = false } = {}) {
  if (!state.models.length) return;
  const name = state.activeModelName || state.models[0].name;
  const found = state.models.find((m) => m.name === name) || state.models[0];
  try {
    await loadGestureModel({
      modelUrl: found.model_url,
      metaUrl: found.meta_url,
      force: !initial,
    });
    if (!initial) {
      showBubble(`已切换到模型 ${found.display_name || found.name}`, "ok");
    }
  } catch (error) {
    console.warn("gesture MLP not loaded; using rule-based classifier", error);
    showBubble("MLP 未就绪，已降级到规则识别", "warn");
  }
}

async function setActiveModel(name) {
  if (!name || name === state.activeModelName) return;
  try {
    const res = await fetchApi("/api/models/active", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) {
      const text = await res.text();
      showBubble(`切换失败：${text}`, "warn");
      return;
    }
    state.activeModelName = name;
    renderModelSelect();
    await loadActiveGestureModel();
  } catch (error) {
    console.warn("set active model failed", error);
    showBubble("切换失败", "warn");
  }
}

function enterDemoMode() {
  state.demoMode = true;
  state.testMode = true;
  if (ui.testMode) {
    ui.testMode.checked = true;
    ui.testMode.disabled = true;
  }
  setStatusPill(ui.wsState, "演示模式（无后端）", "warn");
  setStatusPill(ui.controlState, "演示模式", "warn");
  setWarning(
    "未检测到本地 Python 后端 — 已自动进入演示模式：识别和 UI 全部正常，但不会真的注入键鼠。" +
    "若要真实控制电脑，请按 README 在本机运行 ./run_gesture_control.sh。",
  );
  showBubble("演示模式：动作仅显示，不真实发送", "warn");
  if (state.actions.length === 0) {
    state.actions = buildDemoActions();
  }
  if (Object.keys(state.bindings).length === 0) {
    state.bindings = buildDemoBindings();
  }
  if (state.models.length === 0) {
    state.models = [
      {
        name: "default",
        display_name: "default",
        model_url: "models/gesture_mlp.onnx",
        meta_url: "models/gesture_mlp.meta.json",
      },
    ];
    state.activeModelName = "default";
    renderModelSelect();
  }
  renderBindingsPanel();
}

function buildDemoActions() {
  return [
    { name: "release", label: "释放（无动作）" },
    { name: "move_cursor", label: "移动鼠标" },
    { name: "click_or_drag", label: "单击 / 长捏拖拽" },
    { name: "press_escape", label: "Esc / 取消" },
    { name: "alt_tab", label: "切换窗口（Alt+Tab）" },
    { name: "media_playpause", label: "播放 / 暂停" },
    { name: "volume_up", label: "音量 +" },
    { name: "volume_down", label: "音量 -" },
    { name: "show_desktop", label: "显示桌面" },
    { name: "media_next", label: "下一首" },
    { name: "scroll_up", label: "向上滚动" },
    { name: "scroll_down", label: "向下滚动" },
    { name: "noop", label: "禁用" },
  ];
}

function buildDemoBindings() {
  return {
    open_palm: { action: "release", enabled: true },
    point: { action: "move_cursor", enabled: true },
    pinch: { action: "click_or_drag", enabled: true },
    fist: { action: "press_escape", enabled: true },
    victory: { action: "alt_tab", enabled: true },
    ok: { action: "media_playpause", enabled: true },
    thumbs_up: { action: "volume_up", enabled: true },
    thumbs_down: { action: "volume_down", enabled: true },
    three: { action: "show_desktop", enabled: true },
    call: { action: "media_next", enabled: true },
    swipe_up: { action: "scroll_up", enabled: true },
    swipe_down: { action: "scroll_down", enabled: true },
  };
}

async function loadInitialState() {
  try {
    const [statusResp, actionsResp] = await Promise.all([
      fetchApi("/api/status").then((r) => r.json()),
      fetchApi("/api/actions").then((r) => r.json()),
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
    enterDemoMode();
  }
}

function bindControlClientEvents() {
  controlClient.addEventListener("connect", () => {
    state.demoMode = false;
    state.wsAttempts = 0;
    setStatusPill(ui.wsState, "后端已连接", "active");
    if (ui.testMode) {
      ui.testMode.disabled = false;
    }
  });
  controlClient.addEventListener("disconnect", () => {
    state.wsAttempts += 1;
    if (state.wsAttempts >= 2 && !state.demoMode) {
      enterDemoMode();
    } else {
      setStatusPill(ui.wsState, "后端断开，重连中", "warn");
    }
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
    fetchApi("/api/control/unlock", { method: "POST" })
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
    fetchApi("/api/control/toggle_pause", { method: "POST" })
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
  if (ui.modelSelect) {
    ui.modelSelect.addEventListener("change", (event) => {
      setActiveModel(event.target.value);
    });
  }
}

(async function main() {
  bindUi();
  bindControlClientEvents();
  controlClient.connect();
  await Promise.all([loadInitialState(), loadAvailableModels()]);
  state.uptimeStartedAt = performance.now();
})();
