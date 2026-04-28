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
import { InPageController } from "./modules/inPageController.js";
import { LabelSmoother, SwipeDetector } from "./modules/temporalSmoother.js";

const inPageController = new InPageController();

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
const processedCanvas = document.getElementById("processedCanvas");
const processedCtx = processedCanvas ? processedCanvas.getContext("2d") : null;
const cameraStage = document.getElementById("cameraStage");
// 离屏缓冲：把 video 帧降采样到这里再做像素处理，保证 30 FPS
const _proc = {
  scratch: document.createElement("canvas"),
  scratchCtx: null,
  width: 320,
  height: 180,
};
_proc.scratch.width = _proc.width;
_proc.scratch.height = _proc.height;
_proc.scratchCtx = _proc.scratch.getContext("2d", { willReadFrequently: true });

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
  viewMode: localStorage.getItem("gesture_view_mode") || "color",
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
  if (processedCanvas) {
    processedCanvas.width = _proc.width;
    processedCanvas.height = _proc.height;
    processedCanvas.style.width = `${rect.width}px`;
    processedCanvas.style.height = `${rect.height}px`;
  }
}

// =================== 视角切换：实时图像处理 =====================

function applyViewMode(mode) {
  state.viewMode = mode;
  localStorage.setItem("gesture_view_mode", mode);
  if (cameraStage) cameraStage.dataset.view = mode;
  if (processedCanvas) {
    const showProcessed = mode === "binary" || mode === "edges";
    processedCanvas.hidden = !showProcessed;
  }
  document.querySelectorAll(".view-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.view === mode);
  });
}

function paintProcessedFrame(mode) {
  if (!processedCtx || !_proc.scratchCtx) return;
  if (mode !== "binary" && mode !== "edges") return;
  if (video.readyState < 2) return;
  const w = _proc.width;
  const h = _proc.height;
  _proc.scratchCtx.drawImage(video, 0, 0, w, h);
  const img = _proc.scratchCtx.getImageData(0, 0, w, h);
  if (mode === "binary") otsuBinary(img);
  else if (mode === "edges") sobelEdges(img);
  processedCtx.putImageData(img, 0, 0);
}

// ITU-R BT.601 luma 公式：Y = 0.299R + 0.587G + 0.114B
function _grayLuma(data, i) {
  return data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
}

// Otsu 自适应二值化：扫一遍直方图找类间方差最大的阈值
function otsuBinary(img) {
  const data = img.data;
  const n = data.length / 4;
  const hist = new Uint32Array(256);
  for (let i = 0; i < data.length; i += 4) {
    hist[Math.round(_grayLuma(data, i))] += 1;
  }
  let sum = 0;
  for (let t = 0; t < 256; t += 1) sum += t * hist[t];
  let sumB = 0;
  let wB = 0;
  let varMax = -1;
  let thresh = 127;
  for (let t = 0; t < 256; t += 1) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = n - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > varMax) {
      varMax = v;
      thresh = t;
    }
  }
  for (let i = 0; i < data.length; i += 4) {
    const v = _grayLuma(data, i) > thresh ? 255 : 0;
    data[i] = data[i + 1] = data[i + 2] = v;
    data[i + 3] = 255;
  }
}

// Sobel 一阶梯度边缘：3x3 Gx/Gy 卷积，magnitude → 灰度输出
function sobelEdges(img) {
  const w = img.width;
  const h = img.height;
  const data = img.data;
  // 先把灰度算到一个 typed array，避免反复读 RGBA
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    gray[p] = _grayLuma(data, i);
  }
  const out = new Float32Array(w * h);
  let maxMag = 0;
  for (let y = 1; y < h - 1; y += 1) {
    for (let x = 1; x < w - 1; x += 1) {
      const i = y * w + x;
      const tl = gray[i - w - 1];
      const tc = gray[i - w];
      const tr = gray[i - w + 1];
      const ml = gray[i - 1];
      const mr = gray[i + 1];
      const bl = gray[i + w - 1];
      const bc = gray[i + w];
      const br = gray[i + w + 1];
      const gx = -tl - 2 * ml - bl + tr + 2 * mr + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const m = Math.hypot(gx, gy);
      out[i] = m;
      if (m > maxMag) maxMag = m;
    }
  }
  const norm = maxMag > 1 ? 255 / maxMag : 1;
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const v = Math.min(255, out[p] * norm);
    // 反相：边缘亮、背景黑（更像教材里的 Canny 输出）
    data[i] = data[i + 1] = data[i + 2] = v;
    data[i + 3] = 255;
  }
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
  result = rerankPinchVsFist(result, landmarks);
  return result;
}

// 双向校验 pinch ↔ fist：用食指相对掌心的伸展程度区分。
// 真正的 pinch：食指还在伸（捏拇指时食指其实是伸出去的），indexExtension 大；
// 真正的 fist：食指收进掌心，indexExtension 小。
function rerankPinchVsFist(result, landmarks) {
  if (!result) return result;
  if (result.label !== "pinch" && result.label !== "fist") return result;
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const palmScale = Math.max(
    Math.hypot(wrist.x - middleMcp.x, wrist.y - middleMcp.y),
    1e-4,
  );
  const indexTip = landmarks[8];
  const thumbTip = landmarks[4];
  const indexExtension =
    Math.hypot(indexTip.x - wrist.x, indexTip.y - wrist.y) / palmScale;
  const pinchDist =
    Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y) / palmScale;

  if (result.label === "pinch") {
    // 食指收得太近 wrist → 实际是握拳/类拳
    if (indexExtension < 1.1) {
      return { ...result, label: "fist", confidence: 0.72 };
    }
  } else {
    // result.label === "fist"
    // 食指明显伸展 + 拇指食指距离很近 → 实际是捏合
    if (indexExtension > 1.4 && pinchDist < 0.45) {
      return { ...result, label: "pinch", confidence: 0.72 };
    }
  }
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

// 平滑后的 anchor，避免摄像头噪声让指针抖
const _anchorSmoother = { x: null, y: null, prevLabel: null };

function deriveAnchor(landmarks, label) {
  const indexTip = landmarks[8];
  const thumbTip = landmarks[4];
  let rawX, rawY;
  if (label === "pinch") {
    // 拇指食指中点
    rawX = (indexTip.x + thumbTip.x) * 0.5;
    rawY = (indexTip.y + thumbTip.y) * 0.5;
  } else {
    // 默认锁定食指尖
    rawX = indexTip.x;
    rawY = indexTip.y;
  }
  // 一阶低通：旧值 65% + 新值 35%；切换手势时复位避免跳跃
  if (_anchorSmoother.prevLabel !== label || _anchorSmoother.x === null) {
    _anchorSmoother.x = rawX;
    _anchorSmoother.y = rawY;
  } else {
    _anchorSmoother.x = _anchorSmoother.x * 0.65 + rawX * 0.35;
    _anchorSmoother.y = _anchorSmoother.y * 0.65 + rawY * 0.35;
  }
  _anchorSmoother.prevLabel = label;
  return { x: _anchorSmoother.x, y: _anchorSmoother.y };
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
    // 视角处理（binary / edges 时把帧画到 processedCanvas）
    paintProcessedFrame(state.viewMode);
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
        dispatchEvent(swipeLabel, 0.85, deriveAnchor(landmarks, swipeLabel), handedness);
      }

      if (smoothed.label) {
        dispatchEvent(
          smoothed.label,
          smoothed.confidence,
          deriveAnchor(landmarks, smoothed.label),
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

// 需要长按确认 1 秒才触发的动作（防误触）
const HOLD_CONFIRM_ACTIONS = new Set([
  "show_desktop",
  "close_window",
  "alt_tab",
  "press_escape",
]);
const HOLD_CONFIRM_MS = 1000;

const holdState = {
  label: null,
  action: null,
  startedAt: 0,
  fired: false,
};

function holdProgressRatio() {
  if (!holdState.label) return 0;
  const elapsed = performance.now() - holdState.startedAt;
  return Math.max(0, Math.min(1, elapsed / HOLD_CONFIRM_MS));
}

function resetHold() {
  if (holdState.label) {
    setHoldOverlay(null, 0);
  }
  holdState.label = null;
  holdState.action = null;
  holdState.startedAt = 0;
  holdState.fired = false;
}

function setHoldOverlay(text, ratio) {
  let el = document.getElementById("holdOverlay");
  if (!text) {
    if (el) el.style.opacity = "0";
    return;
  }
  if (!el) {
    el = document.createElement("div");
    el.id = "holdOverlay";
    el.className = "hold-overlay";
    el.innerHTML = `
      <svg class="hold-ring" viewBox="0 0 80 80" width="80" height="80">
        <circle cx="40" cy="40" r="34" stroke="rgba(255,255,255,0.12)" stroke-width="6" fill="none"/>
        <circle id="holdRingFill" cx="40" cy="40" r="34" stroke="#3dd7c2" stroke-width="6"
                fill="none" stroke-linecap="round"
                stroke-dasharray="213.6" stroke-dashoffset="213.6"
                transform="rotate(-90 40 40)"/>
      </svg>
      <p id="holdText" class="hold-text"></p>
    `;
    const stage = document.querySelector(".camera-stage");
    if (stage) stage.appendChild(el);
    else document.body.appendChild(el);
  }
  el.style.opacity = "1";
  document.getElementById("holdText").textContent = text;
  const fill = document.getElementById("holdRingFill");
  if (fill) {
    const circumference = 2 * Math.PI * 34;
    fill.setAttribute("stroke-dashoffset", String(circumference * (1 - ratio)));
  }
}

function dispatchEvent(label, confidence, anchor, handedness) {
  const binding = state.bindings[label];
  if (!binding || !binding.enabled) {
    resetHold();
    return;
  }

  // 长按确认逻辑
  if (HOLD_CONFIRM_ACTIONS.has(binding.action)) {
    const cn = GESTURE_LABELS_CN[label] || label;
    if (holdState.label !== label) {
      holdState.label = label;
      holdState.action = binding.action;
      holdState.startedAt = performance.now();
      holdState.fired = false;
    }
    if (holdState.fired) return;
    const ratio = holdProgressRatio();
    setHoldOverlay(`${cn} 保持中...`, ratio);
    if (ratio >= 1.0) {
      holdState.fired = true;
      setHoldOverlay(`${cn} ✓`, 1);
      setTimeout(() => setHoldOverlay(null, 0), 600);
      // fall through, 触发实际动作
    } else {
      return;
    }
  } else if (holdState.label && holdState.label !== label) {
    resetHold();
  }
  if (state.testMode) {
    const cn = GESTURE_LABELS_CN[label] || label;
    showBubble(`[测试] ${cn} → ${actionLabel(binding.action)}`, "info");
    state.triggerCount += 1;
    ui.triggerCountValue.textContent = String(state.triggerCount);
    return;
  }
  if (state.demoMode) {
    // 没有本地后端时，把动作派给页面内控制器：在用户当前浏览器里真做
    const ack = inPageController.handle(binding.action, { anchor, confidence });
    if (ack.ok) {
      showBubble(`${GESTURE_LABELS_CN[label] || label} → ${ack.message}`, "ok");
    } else if (ack.message && !/未绑定|noop/.test(ack.message)) {
      showBubble(ack.message, "warn");
    }
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

    // 模拟触发按钮（不用真比手势就能测试这个绑定的效果）
    const sim = document.createElement("button");
    sim.type = "button";
    sim.className = "binding-sim";
    sim.textContent = "▶";
    sim.title = "模拟触发该手势（用于调试与演示）";
    sim.addEventListener("click", () => simulateGesture(label));

    row.appendChild(nameBlock);
    row.appendChild(select);
    row.appendChild(toggle);
    row.appendChild(sim);
    list.appendChild(row);
  }
}

// 模拟一次手势触发（点绑定行的 ▶ 按钮调用），用于调试 + 课堂演示。
// 长按确认类动作会做 4 次循环喂数据填满 1 秒进度条；普通动作直接 fire 一次。
async function simulateGesture(label) {
  const binding = state.bindings[label];
  if (!binding || !binding.enabled) {
    showBubble(`${GESTURE_LABELS_CN[label] || label} 未启用`, "warn");
    return;
  }
  showBubble(`[模拟] ${GESTURE_LABELS_CN[label] || label}`, "info");
  const anchor = { x: 0.5, y: 0.5 };
  if (HOLD_CONFIRM_ACTIONS.has(binding.action)) {
    // 模拟保持 1 秒：每 100ms 喂一次
    const ticks = Math.ceil(HOLD_CONFIRM_MS / 100) + 2;
    for (let i = 0; i < ticks; i += 1) {
      dispatchEvent(label, 0.95, anchor, "Right");
      await new Promise((r) => setTimeout(r, 100));
    }
  } else {
    dispatchEvent(label, 0.95, anchor, "Right");
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
  state.testMode = false; // 演示模式自带网页内真控制，不再强制 testMode
  if (ui.testMode) {
    ui.testMode.checked = false;
    ui.testMode.disabled = false;
  }
  setStatusPill(ui.wsState, "网页内控制", "ok");
  setStatusPill(ui.controlState, "网页内（无后端）", "active");
  setWarning(
    "未检测到本地 Python 后端 — 已切换到「网页内控制」模式：" +
    "手势直接控制当前浏览器的滚动、点击、视频播放、全屏、后退等。" +
    "想要控制操作系统级（鼠标移到桌面、Alt+Tab、调系统音量）请在本机跑 ./run_gesture_control.sh。",
  );
  showBubble("网页内控制：手势在你的浏览器里直接生效", "ok");
  if (state.actions.length === 0) {
    state.actions = buildDemoActions();
  }
  if (Object.keys(state.bindings).length === 0) {
    state.bindings = buildInPageBindings();
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
    state.actions = buildDemoActions();
    state.bindings = buildInPageBindings();
    enterDemoMode();
  }
}

// 演示模式下的默认绑定 — 全是网页内能做到的事，不出现 OS 级动作
function buildInPageBindings() {
  return {
    open_palm: { action: "release", enabled: true },
    point: { action: "move_cursor", enabled: true },
    pinch: { action: "click_or_drag", enabled: true },
    fist: { action: "press_escape", enabled: true },
    victory: { action: "alt_tab", enabled: true },          // 在演示模式 = 全屏切换
    ok: { action: "media_playpause", enabled: true },
    thumbs_up: { action: "volume_up", enabled: true },
    thumbs_down: { action: "volume_down", enabled: true },
    three: { action: "show_desktop", enabled: true },        // 演示模式 = 滚到顶部
    call: { action: "media_next", enabled: true },           // 演示模式 = 视频快进 10s
    swipe_up: { action: "scroll_up", enabled: true },
    swipe_down: { action: "scroll_down", enabled: true },
  };
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
  document.querySelectorAll(".view-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyViewMode(btn.dataset.view);
      showBubble(`视角：${btn.textContent}`, "info");
    });
  });
  applyViewMode(state.viewMode);
}

(async function main() {
  bindUi();
  bindControlClientEvents();
  controlClient.connect();
  await Promise.all([loadInitialState(), loadAvailableModels()]);
  state.uptimeStartedAt = performance.now();
})();
