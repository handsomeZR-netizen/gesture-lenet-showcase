import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const MP_TASKS_VERSION = "0.10.33";
const MP_TASKS_VISION_MODULE =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_TASKS_VERSION}/vision_bundle.mjs`;
const MP_TASKS_WASM_ROOT =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_TASKS_VERSION}/wasm`;
const HAND_MODEL_PATH = "../models/hand_landmarker.task";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const LANDMARK_INDEX = {
  wrist: 0,
  thumbTip: 4,
  indexMcp: 5,
  indexTip: 8,
  middleMcp: 9,
  middleTip: 12,
  ringTip: 16,
  pinkyMcp: 17,
  pinkyTip: 20,
};

const defaultSphere = {
  x: 0,
  y: 0,
  scale: 1,
  rotX: 0.35,
  rotY: 0.55,
  rotZ: -0.12,
};

const state = {
  handLandmarker: null,
  videoReady: false,
  runtimeStarted: false,
  lastVideoTime: -1,
  interactionMode: "booting",
  interactionDetail: "Loading browser-side hand tracker",
  gestureLabel: "no hand",
  dragAnchor: null,
  dualAnchor: null,
  resetHoldStartedAt: 0,
  lastDetectionAt: 0,
  sphereCurrent: { ...defaultSphere },
  sphereTarget: { ...defaultSphere },
  glow: 0,
  glowTarget: 0,
};

const ui = {
  startButton: document.getElementById("startButton"),
  resetButton: document.getElementById("resetButton"),
  cameraState: document.getElementById("cameraState"),
  runtimeState: document.getElementById("runtimeState"),
  modeValue: document.getElementById("modeValue"),
  detailValue: document.getElementById("detailValue"),
  handsValue: document.getElementById("handsValue"),
  scaleValue: document.getElementById("scaleValue"),
  rotationValue: document.getElementById("rotationValue"),
  gestureValue: document.getElementById("gestureValue"),
  cameraOverlay: document.getElementById("cameraOverlay"),
};

const video = document.getElementById("cameraFeed");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayContext = overlayCanvas.getContext("2d");
const stageCanvas = document.getElementById("stageCanvas");

let renderer;
let scene;
let camera;
let orbGroup;
let orbCore;
let orbShell;
let orbRing;
let starField;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function distance2d(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5,
  };
}

function updateStatusPill(element, text, tone) {
  element.textContent = text;
  element.className = "status-pill";
  if (tone === "active") {
    element.classList.add("status-active");
  } else if (tone === "ready") {
    element.classList.add("status-ready");
  } else if (tone === "warning") {
    element.classList.add("status-warning");
  } else {
    element.classList.add("status-idle");
  }
}

function updateHud(handCount) {
  ui.modeValue.textContent = state.interactionMode;
  ui.detailValue.textContent = state.interactionDetail;
  ui.handsValue.textContent = String(handCount);
  ui.scaleValue.textContent = state.sphereTarget.scale.toFixed(2);
  ui.rotationValue.textContent = `${state.sphereTarget.rotY.toFixed(2)} rad`;
  ui.gestureValue.textContent = state.gestureLabel;

  const tone =
    state.interactionMode === "drag" || state.interactionMode === "scale_rotate"
      ? "active"
      : state.interactionMode === "ready"
        ? "ready"
        : state.interactionMode === "reset_hold"
          ? "warning"
          : "idle";
  updateStatusPill(ui.runtimeState, state.interactionMode, tone);
  updateStatusPill(ui.cameraState, state.videoReady ? "camera ready" : "camera off", state.videoReady ? "ready" : "idle");
}

function setOverlayMessage(title, body) {
  ui.cameraOverlay.innerHTML = `
    <p class="overlay-title">${title}</p>
    <p class="overlay-body">${body}</p>
  `;
}

function resetSphere() {
  state.sphereTarget = { ...defaultSphere };
  state.dragAnchor = null;
  state.dualAnchor = null;
  state.glowTarget = 0.25;
}

function summariseHand(landmarks, handedness) {
  const renderPoints = landmarks.map((landmark) => ({
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
  }));
  const controlPoints = landmarks.map((landmark) => ({
    x: 1 - landmark.x,
    y: landmark.y,
    z: landmark.z,
  }));

  const wrist = controlPoints[LANDMARK_INDEX.wrist];
  const thumbTip = controlPoints[LANDMARK_INDEX.thumbTip];
  const indexTip = controlPoints[LANDMARK_INDEX.indexTip];
  const middleTip = controlPoints[LANDMARK_INDEX.middleTip];
  const ringTip = controlPoints[LANDMARK_INDEX.ringTip];
  const pinkyTip = controlPoints[LANDMARK_INDEX.pinkyTip];
  const middleMcp = controlPoints[LANDMARK_INDEX.middleMcp];
  const indexMcp = controlPoints[LANDMARK_INDEX.indexMcp];
  const pinkyMcp = controlPoints[LANDMARK_INDEX.pinkyMcp];

  const palmWidth = Math.max(distance2d(indexMcp, pinkyMcp), 0.06);
  const palmHeight = Math.max(distance2d(wrist, middleMcp), 0.06);
  const handScale = Math.max((palmWidth + palmHeight) * 0.5, 0.08);
  const pinchDistance = distance2d(thumbTip, indexTip) / handScale;
  const pinchStrength = clamp(1 - pinchDistance / 0.62, 0, 1);

  const extensionScore =
    (
      distance2d(indexTip, wrist) +
      distance2d(middleTip, wrist) +
      distance2d(ringTip, wrist) +
      distance2d(pinkyTip, wrist)
    ) /
    (4 * handScale);

  const isPinched = pinchDistance < 0.42;
  const openPalm = extensionScore > 1.42 && pinchDistance > 0.56;

  return {
    handedness,
    renderPoints,
    controlPoints,
    pinchPoint: midpoint(thumbTip, indexTip),
    pinchDistance,
    pinchStrength,
    openPalm,
    isPinched,
  };
}

function beginDrag(hand) {
  state.dragAnchor = {
    handX: hand.pinchPoint.x,
    handY: hand.pinchPoint.y,
    sphereX: state.sphereTarget.x,
    sphereY: state.sphereTarget.y,
  };
}

function applyDrag(hand) {
  if (!state.dragAnchor) {
    beginDrag(hand);
  }
  const dx = hand.pinchPoint.x - state.dragAnchor.handX;
  const dy = hand.pinchPoint.y - state.dragAnchor.handY;
  state.sphereTarget.x = clamp(state.dragAnchor.sphereX + dx * 5.2, -2.35, 2.35);
  state.sphereTarget.y = clamp(state.dragAnchor.sphereY - dy * 3.9, -1.7, 1.7);
  state.interactionMode = "drag";
  state.interactionDetail = "Single-hand pinch: move to drag the sphere";
  state.gestureLabel = "pinch drag";
  state.glowTarget = 0.7;
}

function beginDualTransform(leftHand, rightHand) {
  const center = midpoint(leftHand.pinchPoint, rightHand.pinchPoint);
  state.dualAnchor = {
    center,
    distance: Math.max(distance2d(leftHand.pinchPoint, rightHand.pinchPoint), 0.08),
    angle: Math.atan2(
      rightHand.pinchPoint.y - leftHand.pinchPoint.y,
      rightHand.pinchPoint.x - leftHand.pinchPoint.x,
    ),
    x: state.sphereTarget.x,
    y: state.sphereTarget.y,
    scale: state.sphereTarget.scale,
    rotX: state.sphereTarget.rotX,
    rotY: state.sphereTarget.rotY,
  };
}

function applyDualTransform(leftHand, rightHand) {
  if (!state.dualAnchor) {
    beginDualTransform(leftHand, rightHand);
  }

  const currentCenter = midpoint(leftHand.pinchPoint, rightHand.pinchPoint);
  const currentDistance = Math.max(distance2d(leftHand.pinchPoint, rightHand.pinchPoint), 0.08);
  const currentAngle = Math.atan2(
    rightHand.pinchPoint.y - leftHand.pinchPoint.y,
    rightHand.pinchPoint.x - leftHand.pinchPoint.x,
  );
  const anchor = state.dualAnchor;

  state.sphereTarget.scale = clamp(anchor.scale * (currentDistance / anchor.distance), 0.55, 2.8);
  state.sphereTarget.rotY = anchor.rotY + (currentAngle - anchor.angle) * 1.8;
  state.sphereTarget.rotX = clamp(anchor.rotX + (currentCenter.y - anchor.center.y) * 3.4, -1.3, 1.3);
  state.sphereTarget.x = clamp(anchor.x + (currentCenter.x - anchor.center.x) * 4.8, -2.35, 2.35);
  state.sphereTarget.y = clamp(anchor.y - (currentCenter.y - anchor.center.y) * 3.8, -1.7, 1.7);
  state.interactionMode = "scale_rotate";
  state.interactionDetail = "Two-hand pinch: change distance to zoom and angle to rotate";
  state.gestureLabel = "dual pinch";
  state.glowTarget = 1.0;
}

function clearTransientAnchors() {
  state.dragAnchor = null;
  state.dualAnchor = null;
}

function drawOverlay(hands) {
  const { width, height } = overlayCanvas;
  overlayContext.clearRect(0, 0, width, height);

  if (!hands.length) {
    overlayContext.strokeStyle = "rgba(255,255,255,0.26)";
    overlayContext.lineWidth = 2;
    overlayContext.setLineDash([14, 10]);
    overlayContext.strokeRect(width * 0.18, height * 0.16, width * 0.64, height * 0.68);
    overlayContext.setLineDash([]);
    return;
  }

  hands.forEach((hand) => {
    overlayContext.lineWidth = 3;
    overlayContext.strokeStyle = "rgba(59, 226, 205, 0.9)";
    overlayContext.fillStyle = "rgba(255, 247, 214, 0.95)";

    HAND_CONNECTIONS.forEach(([start, end]) => {
      const startPoint = hand.renderPoints[start];
      const endPoint = hand.renderPoints[end];
      overlayContext.beginPath();
      overlayContext.moveTo(startPoint.x * width, startPoint.y * height);
      overlayContext.lineTo(endPoint.x * width, endPoint.y * height);
      overlayContext.stroke();
    });

    hand.renderPoints.forEach((point) => {
      overlayContext.beginPath();
      overlayContext.arc(point.x * width, point.y * height, 5, 0, Math.PI * 2);
      overlayContext.fill();
    });

    const rawPinchX = (1 - hand.pinchPoint.x) * width;
    const rawPinchY = hand.pinchPoint.y * height;
    overlayContext.beginPath();
    overlayContext.arc(rawPinchX, rawPinchY, 10, 0, Math.PI * 2);
    overlayContext.fillStyle = hand.isPinched ? "rgba(229, 114, 74, 0.95)" : "rgba(49, 194, 179, 0.95)";
    overlayContext.fill();

    overlayContext.fillStyle = "rgba(255, 255, 255, 0.94)";
    overlayContext.font = "600 18px 'JetBrains Mono'";
    overlayContext.fillText(
      `${hand.handedness || "Hand"} ${hand.isPinched ? "pinch" : hand.openPalm ? "ready" : "tracking"}`,
      rawPinchX + 14,
      rawPinchY - 12,
    );
  });
}

function processHands(result, nowMs) {
  const hands = (result.landmarks || []).map((landmarks, index) => {
    const handednessLabel =
      result.handednesses?.[index]?.[0]?.displayName ||
      result.handednesses?.[index]?.[0]?.categoryName ||
      "";
    return summariseHand(landmarks, handednessLabel);
  });

  hands.sort((a, b) => a.pinchPoint.x - b.pinchPoint.x);

  if (!hands.length) {
    clearTransientAnchors();
    state.resetHoldStartedAt = 0;
    state.interactionMode = "idle";
    state.interactionDetail = "Show one open hand to enter ready mode";
    state.gestureLabel = "no hand";
    state.glowTarget = 0;
    drawOverlay([]);
    updateHud(0);
    return;
  }

  state.lastDetectionAt = nowMs;
  ui.cameraOverlay.style.display = "none";

  const pinchedHands = hands.filter((hand) => hand.isPinched);
  const allOpen = hands.length === 2 && hands.every((hand) => hand.openPalm);

  if (hands.length >= 2 && pinchedHands.length >= 2) {
    state.dragAnchor = null;
    state.resetHoldStartedAt = 0;
    applyDualTransform(hands[0], hands[1]);
  } else if (allOpen) {
    clearTransientAnchors();
    if (!state.resetHoldStartedAt) {
      state.resetHoldStartedAt = nowMs;
    }
    const heldForSeconds = (nowMs - state.resetHoldStartedAt) / 1000;
    state.interactionMode = "reset_hold";
    state.interactionDetail = `Hold both open hands for ${Math.max(0, 1.2 - heldForSeconds).toFixed(1)}s to reset`;
    state.gestureLabel = "reset hold";
    state.glowTarget = 0.25;
    if (heldForSeconds >= 1.2) {
      resetSphere();
      state.interactionMode = "ready";
      state.interactionDetail = "Sphere reset. Pinch again to continue controlling.";
      state.gestureLabel = "reset complete";
      state.resetHoldStartedAt = 0;
    }
  } else if (pinchedHands.length === 1) {
    state.dualAnchor = null;
    state.resetHoldStartedAt = 0;
    applyDrag(pinchedHands[0]);
  } else if (hands.some((hand) => hand.openPalm)) {
    clearTransientAnchors();
    state.resetHoldStartedAt = 0;
    state.interactionMode = "ready";
    state.interactionDetail = "Open hand detected. Pinch thumb and index finger to drag.";
    state.gestureLabel = "ready palm";
    state.glowTarget = 0.18;
  } else {
    clearTransientAnchors();
    state.resetHoldStartedAt = 0;
    state.interactionMode = "tracking";
    state.interactionDetail = "Hand detected. Open your palm or pinch more clearly.";
    state.gestureLabel = "tracking";
    state.glowTarget = 0.1;
  }

  drawOverlay(hands);
  updateHud(hands.length);
}

function syncOverlaySize() {
  const rect = overlayCanvas.getBoundingClientRect();
  const deviceScale = window.devicePixelRatio || 1;
  const width = Math.round(rect.width * deviceScale);
  const height = Math.round(rect.height * deviceScale);
  if (overlayCanvas.width !== width || overlayCanvas.height !== height) {
    overlayCanvas.width = width;
    overlayCanvas.height = height;
  }
}

function initThree() {
  renderer = new THREE.WebGLRenderer({
    canvas: stageCanvas,
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 80);
  camera.position.set(0, 0.6, 7.6);

  const ambient = new THREE.HemisphereLight(0xfbf8ef, 0x0e1e21, 1.2);
  scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0xfff3d1, 1.6);
  keyLight.position.set(4.6, 6.2, 5.2);
  scene.add(keyLight);

  const rimLight = new THREE.PointLight(0x31c2b3, 3.8, 22);
  rimLight.position.set(-4.2, 2.8, 4.2);
  scene.add(rimLight);

  orbGroup = new THREE.Group();
  scene.add(orbGroup);

  orbCore = new THREE.Mesh(
    new THREE.SphereGeometry(1.18, 72, 72),
    new THREE.MeshPhysicalMaterial({
      color: 0xe8b76d,
      roughness: 0.16,
      metalness: 0.08,
      transmission: 0.08,
      clearcoat: 1,
      clearcoatRoughness: 0.12,
      emissive: 0x0f766e,
      emissiveIntensity: 0.14,
    }),
  );
  orbGroup.add(orbCore);

  orbShell = new THREE.Mesh(
    new THREE.SphereGeometry(1.34, 48, 48),
    new THREE.MeshBasicMaterial({
      color: 0x7de4d7,
      transparent: true,
      opacity: 0.12,
      wireframe: true,
    }),
  );
  orbGroup.add(orbShell);

  orbRing = new THREE.Mesh(
    new THREE.TorusGeometry(1.65, 0.05, 16, 140),
    new THREE.MeshStandardMaterial({
      color: 0x31c2b3,
      emissive: 0x31c2b3,
      emissiveIntensity: 0.35,
      roughness: 0.28,
      metalness: 0.3,
    }),
  );
  orbRing.rotation.x = Math.PI / 2.6;
  orbGroup.add(orbRing);

  const particleGeometry = new THREE.BufferGeometry();
  const particleCount = 220;
  const positions = new Float32Array(particleCount * 3);
  for (let index = 0; index < particleCount; index += 1) {
    const radius = 2.4 + Math.random() * 3.8;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[index * 3] = radius * Math.sin(phi) * Math.cos(theta);
    positions[index * 3 + 1] = radius * Math.cos(phi) * 0.55;
    positions[index * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
  }
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  starField = new THREE.Points(
    particleGeometry,
    new THREE.PointsMaterial({
      color: 0xffefc6,
      size: 0.045,
      transparent: true,
      opacity: 0.72,
    }),
  );
  scene.add(starField);

  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(5.2, 80),
    new THREE.MeshBasicMaterial({
      color: 0x0a1d20,
      transparent: true,
      opacity: 0.45,
    }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(0, -2.05, 0);
  scene.add(ground);

  resizeStage();
}

function resizeStage() {
  const rect = stageCanvas.getBoundingClientRect();
  const width = Math.max(rect.width, 1);
  const height = Math.max(rect.height, 1);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function animateSphere(timeMs) {
  const idleDrift = Math.sin(timeMs * 0.00055) * 0.04;
  state.glow = lerp(state.glow, state.glowTarget, 0.08);

  state.sphereCurrent.x = lerp(state.sphereCurrent.x, state.sphereTarget.x, 0.11);
  state.sphereCurrent.y = lerp(state.sphereCurrent.y, state.sphereTarget.y, 0.11);
  state.sphereCurrent.scale = lerp(state.sphereCurrent.scale, state.sphereTarget.scale, 0.11);
  state.sphereCurrent.rotX = lerp(state.sphereCurrent.rotX, state.sphereTarget.rotX, 0.11);
  state.sphereCurrent.rotY = lerp(state.sphereCurrent.rotY, state.sphereTarget.rotY, 0.11);
  state.sphereCurrent.rotZ = lerp(state.sphereCurrent.rotZ, state.sphereTarget.rotZ, 0.11);

  orbGroup.position.set(state.sphereCurrent.x, state.sphereCurrent.y + idleDrift, 0);
  orbGroup.scale.setScalar(state.sphereCurrent.scale);
  orbGroup.rotation.x = state.sphereCurrent.rotX;
  orbGroup.rotation.y = state.sphereCurrent.rotY + timeMs * 0.00012;
  orbGroup.rotation.z = state.sphereCurrent.rotZ;

  orbRing.rotation.z += 0.003 + state.glow * 0.02;
  orbShell.rotation.y -= 0.0025;
  starField.rotation.y += 0.0007;
  starField.rotation.x = Math.sin(timeMs * 0.0001) * 0.12;

  orbCore.material.emissiveIntensity = 0.14 + state.glow * 0.45;
  orbRing.material.emissiveIntensity = 0.28 + state.glow * 0.75;

  renderer.render(scene, camera);
}

async function initHandLandmarker() {
  const visionTasks = await import(MP_TASKS_VISION_MODULE);
  const { FilesetResolver, HandLandmarker } = visionTasks;
  const filesetResolver = await FilesetResolver.forVisionTasks(MP_TASKS_WASM_ROOT);
  state.handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_PATH,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  state.interactionMode = "idle";
  state.interactionDetail = "Tracker ready. Start the camera to begin.";
  updateHud(0);
}

async function startCamera() {
  if (state.runtimeStarted) {
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    state.videoReady = true;
    state.runtimeStarted = true;
    state.interactionMode = "idle";
    state.interactionDetail = "Camera ready. Show one open hand to enter ready mode.";
    setOverlayMessage(
      "Tracking active",
      "Open one hand for ready mode. Use one-hand pinch to drag. Use two-hand pinch to scale and rotate.",
    );
    ui.startButton.textContent = "Camera Running";
    ui.startButton.disabled = true;
    updateHud(0);
  } catch (error) {
    state.interactionMode = "error";
    state.interactionDetail = "Camera access failed. Allow webcam permission and refresh.";
    setOverlayMessage("Camera permission failed", String(error));
    updateHud(0);
  }
}

function processFrame() {
  syncOverlaySize();

  if (
    state.handLandmarker &&
    state.videoReady &&
    video.readyState >= 2 &&
    video.currentTime !== state.lastVideoTime
  ) {
    const nowMs = performance.now();
    const result = state.handLandmarker.detectForVideo(video, nowMs);
    processHands(result, nowMs);
    state.lastVideoTime = video.currentTime;
  }

  requestAnimationFrame(processFrame);
}

function renderStage(timeMs) {
  animateSphere(timeMs);
  requestAnimationFrame(renderStage);
}

async function bootstrap() {
  initThree();
  updateHud(0);
  setOverlayMessage(
    "Camera inactive",
    "Start the demo, allow webcam access, then use one-hand pinch to drag and two-hand pinch to scale and rotate.",
  );

  ui.startButton.addEventListener("click", startCamera);
  ui.resetButton.addEventListener("click", () => {
    resetSphere();
    state.interactionMode = "ready";
    state.interactionDetail = "Sphere reset manually.";
    state.gestureLabel = "manual reset";
    updateHud(state.videoReady ? 1 : 0);
  });

  window.addEventListener("resize", () => {
    resizeStage();
    syncOverlaySize();
  });

  await initHandLandmarker();
  requestAnimationFrame(processFrame);
  requestAnimationFrame(renderStage);
}

bootstrap().catch((error) => {
  state.interactionMode = "error";
  state.interactionDetail = "Initialization failed.";
  setOverlayMessage("Initialization failed", String(error));
  updateHud(0);
  console.error(error);
});
