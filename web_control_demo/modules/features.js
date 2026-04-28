// JS port of gesture_lenet/gesture_mlp/features.py.
// Must stay byte-identical (within 1e-5) so the ONNX model trained in Python
// receives the same feature vector the browser produces at runtime.

export const NUM_LANDMARKS = 21;
export const FEATURE_DIM = 63;
export const PALM_SCALE_FLOOR = 1e-4;

const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

function distance2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function palmScale(landmarks) {
  const a = distance2D(landmarks[WRIST], landmarks[MIDDLE_MCP]);
  const b = distance2D(landmarks[INDEX_MCP], landmarks[PINKY_MCP]);
  return Math.max(a, b, PALM_SCALE_FLOOR);
}

export function normalizeLandmarks(landmarks, handedness = "Right") {
  if (!landmarks || landmarks.length !== NUM_LANDMARKS) {
    throw new Error(`expected ${NUM_LANDMARKS} landmarks, got ${landmarks?.length}`);
  }
  const wrist = landmarks[WRIST];
  const scale = palmScale(landmarks);
  const mirror = String(handedness || "").toLowerCase().startsWith("l");
  const out = new Array(NUM_LANDMARKS);
  for (let i = 0; i < NUM_LANDMARKS; i += 1) {
    const lm = landmarks[i];
    let x = (lm.x - wrist.x) / scale;
    const y = (lm.y - wrist.y) / scale;
    const z = (lm.z - wrist.z) / scale;
    if (mirror) x = -x;
    out[i] = { x, y, z };
  }
  return out;
}

export function landmarksToFeature(landmarks, handedness = "Right") {
  const canonical = normalizeLandmarks(landmarks, handedness);
  const feat = new Float32Array(FEATURE_DIM);
  for (let i = 0; i < NUM_LANDMARKS; i += 1) {
    feat[i * 3 + 0] = canonical[i].x;
    feat[i * 3 + 1] = canonical[i].y;
    feat[i * 3 + 2] = canonical[i].z;
  }
  return feat;
}

// Plain-array landmarks ([x,y,z]) variant for record.html where MediaPipe
// results have already been serialized.
export function landmarksArrayToFeature(arrayLandmarks, handedness = "Right") {
  const objectified = arrayLandmarks.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
  return landmarksToFeature(objectified, handedness);
}
