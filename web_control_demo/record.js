// Wizard for collecting per-gesture keypoint samples.
// Recorded samples POST to the Python backend at /api/dataset/<label>.

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

const GESTURES = [
  { id: "open_palm", cn: "张开手掌", tip: "五指自然张开，掌心面向镜头" },
  { id: "point", cn: "食指指向", tip: "仅食指竖起，其他手指收拢" },
  { id: "pinch", cn: "捏合", tip: "拇指食指轻轻捏在一起，距离 < 1cm" },
  { id: "fist", cn: "握拳", tip: "整只手握紧成拳头" },
  { id: "victory", cn: "V 字", tip: "食指与中指竖起呈 V 字" },
  { id: "ok", cn: "OK 圈", tip: "拇指食指捏成圈，中指无名指小指自然伸展" },
  { id: "thumbs_up", cn: "拇指向上", tip: "握拳后只竖大拇指" },
  { id: "thumbs_down", cn: "拇指向下", tip: "拇指朝下，其余四指收拢" },
  { id: "three", cn: "三指", tip: "食指、中指、无名指竖起，小指收拢" },
  { id: "call", cn: "电话手势", tip: "拇指与小指伸出，中间三指收拢" },
];

const SAMPLES_PER_GESTURE = 250;
const COUNTDOWN_SECONDS = 3;

const ui = {
  startCamera: document.getElementById("startCamera"),
  recordButton: document.getElementById("recordButton"),
  skipButton: document.getElementById("skipButton"),
  cameraState: document.getElementById("cameraState"),
  progressState: document.getElementById("progressState"),
  cameraOverlay: document.getElementById("cameraOverlay"),
  recordingBanner: document.getElementById("recordingBanner"),
  recordingText: document.getElementById("recordingText"),
  recordingProgress: document.getElementById("recordingProgress"),
  currentLabel: document.getElementById("currentLabel"),
  currentTip: document.getElementById("currentTip"),
  recordList: document.getElementById("recordList"),
  finishedHint: document.getElementById("finishedHint"),
};

const video = document.getElementById("cameraFeed");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayContext = overlayCanvas.getContext("2d");

const state = {
  handLandmarker: null,
  videoReady: false,
  recording: false,
  countdownActive: false,
  collected: {},
  currentIndex: 0,
  lastVideoTime: -1,
};

function setPill(el, text, kind) {
  el.textContent = text;
  el.className = `status-pill status-${kind}`;
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  try {
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "zh-CN";
    utter.rate = 1.05;
    speechSynthesis.cancel();
    speechSynthesis.speak(utter);
  } catch {
    // ignore
  }
}

async function loadDatasetSummary() {
  try {
    const res = await fetch("/api/dataset/summary");
    if (!res.ok) return;
    const data = await res.json();
    state.collected = data.counts || {};
  } catch (error) {
    console.warn("dataset summary unavailable", error);
  }
}

function renderList() {
  const list = ui.recordList;
  list.innerHTML = "";
  GESTURES.forEach((gesture, idx) => {
    const collected = state.collected[gesture.id] || 0;
    const row = document.createElement("div");
    row.className = "record-row";
    if (idx === state.currentIndex) row.classList.add("is-active");
    if (collected >= SAMPLES_PER_GESTURE) row.classList.add("is-done");

    const name = document.createElement("div");
    name.className = "row-name";
    name.textContent = `${gesture.cn} · ${gesture.id}`;

    const count = document.createElement("div");
    count.className = "row-count";
    count.textContent = `${collected} / ${SAMPLES_PER_GESTURE}`;

    const stateLabel = document.createElement("div");
    stateLabel.className = "row-state";
    if (collected >= SAMPLES_PER_GESTURE) stateLabel.textContent = "完成";
    else if (idx === state.currentIndex) stateLabel.textContent = "进行中";
    else stateLabel.textContent = "待采集";

    row.appendChild(name);
    row.appendChild(count);
    row.appendChild(stateLabel);
    list.appendChild(row);
  });
  refreshHeader();
  refreshFinishedHint();
}

function refreshHeader() {
  const gesture = GESTURES[state.currentIndex];
  if (!gesture) {
    ui.currentLabel.textContent = "全部完成";
    ui.currentTip.textContent = "可以关闭页面，回终端跑训练。";
    return;
  }
  ui.currentLabel.textContent = `${state.currentIndex + 1}/${GESTURES.length} · ${gesture.cn}`;
  ui.currentTip.textContent = gesture.tip;
}

function refreshFinishedHint() {
  const allDone = GESTURES.every(
    (g) => (state.collected[g.id] || 0) >= SAMPLES_PER_GESTURE,
  );
  ui.finishedHint.hidden = !allDone;
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
  ctx.strokeStyle = "rgba(255, 190, 92, 0.85)";
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
    ctx.fillStyle = "#3dd7c2";
    ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function ensureHandLandmarker() {
  if (state.handLandmarker) return state.handLandmarker;
  const mod = await import(/* @vite-ignore */ MP_VISION_MODULE);
  const fileset = await mod.FilesetResolver.forVisionTasks(MP_WASM_ROOT);
  state.handLandmarker = await mod.HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL_PATH },
    numHands: 1,
    runningMode: "VIDEO",
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  return state.handLandmarker;
}

async function startCamera() {
  ui.startCamera.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    state.videoReady = true;
    setPill(ui.cameraState, "摄像头已启动", "active");
    setPill(ui.progressState, "等待开始", "ok");
    ui.cameraOverlay.hidden = true;
    syncOverlaySize();
  } catch (error) {
    setPill(ui.cameraState, "摄像头被拒绝", "warn");
    ui.cameraOverlay.querySelector(".overlay-body").textContent = error?.message || "无法访问摄像头";
    ui.startCamera.disabled = false;
    return;
  }
  await ensureHandLandmarker();
  ui.recordButton.disabled = false;
  ui.startCamera.textContent = "已启动";
  requestAnimationFrame(detectLoop);
}

let latestLandmarks = null;
let latestHandedness = "Right";

async function detectLoop() {
  syncOverlaySize();
  if (state.videoReady && video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    try {
      const result = state.handLandmarker.detectForVideo(video, performance.now());
      if (result?.landmarks?.length) {
        latestLandmarks = result.landmarks[0];
        latestHandedness = result.handednesses?.[0]?.[0]?.categoryName || "Right";
        drawSkeleton(latestLandmarks);
      } else {
        latestLandmarks = null;
        drawSkeleton(null);
      }
    } catch (error) {
      console.warn("detect failed", error);
    }
  }
  requestAnimationFrame(detectLoop);
}

async function recordCurrent() {
  if (state.recording || state.countdownActive) return;
  const gesture = GESTURES[state.currentIndex];
  if (!gesture) return;

  state.countdownActive = true;
  ui.recordingBanner.hidden = false;
  ui.recordButton.disabled = true;
  setPill(ui.progressState, "倒计时中", "warning");

  speak(`${gesture.cn}，准备好`);
  for (let i = COUNTDOWN_SECONDS; i > 0; i -= 1) {
    ui.recordingText.textContent = `${i}`;
    ui.recordingProgress.style.width = "0%";
    await sleep(1000);
  }
  state.countdownActive = false;

  state.recording = true;
  setPill(ui.progressState, "录制中", "active");
  speak("开始");
  ui.recordingText.textContent = `录制 ${gesture.cn}（请保持手势并轻微移动）`;

  const samples = [];
  const target = SAMPLES_PER_GESTURE;
  while (samples.length < target) {
    if (latestLandmarks) {
      samples.push({
        landmarks: latestLandmarks.map((p) => [p.x, p.y, p.z]),
        handedness: latestHandedness,
        ts: Date.now() / 1000,
      });
      ui.recordingProgress.style.width = `${(samples.length / target) * 100}%`;
    }
    await sleep(33);
  }

  state.recording = false;
  ui.recordingBanner.hidden = true;
  setPill(ui.progressState, "上传中", "warning");

  try {
    await fetch(`/api/dataset/${gesture.id}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ samples }),
    });
    state.collected[gesture.id] = (state.collected[gesture.id] || 0) + samples.length;
    speak(`${gesture.cn}录制完成`);
  } catch (error) {
    console.warn("upload failed", error);
    speak("上传失败");
  }

  setPill(ui.progressState, "等待开始", "ok");
  ui.recordButton.disabled = false;

  // advance to next unfinished gesture
  const next = GESTURES.findIndex(
    (g, idx) => idx > state.currentIndex && (state.collected[g.id] || 0) < SAMPLES_PER_GESTURE,
  );
  if (next >= 0) {
    state.currentIndex = next;
  } else {
    const fallback = GESTURES.findIndex(
      (g) => (state.collected[g.id] || 0) < SAMPLES_PER_GESTURE,
    );
    state.currentIndex = fallback >= 0 ? fallback : GESTURES.length;
  }
  renderList();
  if (state.currentIndex >= GESTURES.length) {
    speak("全部录制完成。请到终端运行训练脚本。");
    setPill(ui.progressState, "全部完成", "ok");
  }
}

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

function bindUi() {
  ui.startCamera.addEventListener("click", startCamera);
  ui.recordButton.addEventListener("click", recordCurrent);
  ui.skipButton.addEventListener("click", () => {
    if (state.recording || state.countdownActive) return;
    if (state.currentIndex >= GESTURES.length - 1) {
      state.currentIndex = GESTURES.length;
    } else {
      state.currentIndex += 1;
    }
    renderList();
  });
}

(async function main() {
  bindUi();
  await loadDatasetSummary();
  // start at first unfinished gesture
  const firstUnfinished = GESTURES.findIndex(
    (g) => (state.collected[g.id] || 0) < SAMPLES_PER_GESTURE,
  );
  state.currentIndex = firstUnfinished >= 0 ? firstUnfinished : 0;
  renderList();
})();
