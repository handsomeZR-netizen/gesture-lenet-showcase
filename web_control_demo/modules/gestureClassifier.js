// Loads gesture_mlp.onnx via onnxruntime-web and classifies a 63-d feature
// vector into one of the trained gesture labels. Falls back to a rule-based
// classifier if the model cannot be loaded (e.g. file missing during dev).

import { landmarksToFeature, FEATURE_DIM } from "./features.js";

const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.mjs";
const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
const META_URL = "models/gesture_mlp.meta.json";
const MODEL_URL = "models/gesture_mlp.onnx";

let _ort = null;
let _session = null;
let _labels = null;
let _loaded = false;
let _loadError = null;

async function loadOrt() {
  if (_ort) return _ort;
  try {
    const mod = await import(/* @vite-ignore */ ORT_CDN);
    if (mod?.env?.wasm) {
      mod.env.wasm.wasmPaths = ORT_WASM_BASE;
      mod.env.wasm.numThreads = 1;
    }
    _ort = mod;
    return mod;
  } catch (error) {
    _loadError = error;
    throw error;
  }
}

export async function loadGestureModel({
  modelUrl = MODEL_URL,
  metaUrl = META_URL,
} = {}) {
  if (_loaded) return { labels: _labels, ort: _ort };
  try {
    const ort = await loadOrt();
    const meta = await fetch(metaUrl, { cache: "no-cache" }).then((r) => r.json());
    _labels = meta.labels;
    _session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });
    _loaded = true;
    return { labels: _labels, ort };
  } catch (error) {
    _loadError = error;
    _loaded = false;
    throw error;
  }
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export async function classifyLandmarks(landmarks, handedness = "Right") {
  if (!_loaded) {
    throw new Error("gesture model not loaded yet");
  }
  const feature = landmarksToFeature(landmarks, handedness);
  const input = new _ort.Tensor("float32", feature, [1, FEATURE_DIM]);
  const outputs = await _session.run({ features: input });
  const logits = Array.from(outputs.logits.data);
  const probs = softmax(logits);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i += 1) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }
  return {
    label: _labels[bestIdx],
    confidence: probs[bestIdx],
    probs,
  };
}

export function isModelLoaded() {
  return _loaded;
}

export function loadError() {
  return _loadError;
}

// ---------------------------------------------------------------- rule fallback

const FINGER_TIPS = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const FINGER_PIPS = { thumb: 3, index: 6, middle: 10, ring: 14, pinky: 18 };

function fingerExtended(landmarks, finger) {
  if (finger === "thumb") {
    const tip = landmarks[4];
    const ip = landmarks[3];
    const mcp = landmarks[2];
    const wrist = landmarks[0];
    return Math.abs(tip.x - wrist.x) > Math.abs(ip.x - wrist.x) * 1.05;
  }
  const tip = landmarks[FINGER_TIPS[finger]];
  const pip = landmarks[FINGER_PIPS[finger]];
  return tip.y < pip.y - 0.02;
}

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function classifyByRule(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const palm = Math.max(dist2(wrist, middleMcp), 1e-4);
  const states = {
    thumb: fingerExtended(landmarks, "thumb"),
    index: fingerExtended(landmarks, "index"),
    middle: fingerExtended(landmarks, "middle"),
    ring: fingerExtended(landmarks, "ring"),
    pinky: fingerExtended(landmarks, "pinky"),
  };
  const extended = Object.values(states).filter(Boolean).length;
  const pinch = dist2(landmarks[4], landmarks[8]) / palm;

  if (pinch < 0.42) {
    return { label: "pinch", confidence: 0.7 };
  }
  if (states.index && states.middle && !states.ring && !states.pinky) {
    return { label: "victory", confidence: 0.7 };
  }
  if (states.index && !states.middle && !states.ring && !states.pinky) {
    return { label: "point", confidence: 0.7 };
  }
  if (extended >= 4) {
    return { label: "open_palm", confidence: 0.75 };
  }
  if (states.thumb && !states.index && !states.middle && !states.ring && !states.pinky) {
    if (landmarks[4].y < wrist.y - 0.05) {
      return { label: "thumbs_up", confidence: 0.65 };
    }
    if (landmarks[4].y > wrist.y + 0.05) {
      return { label: "thumbs_down", confidence: 0.65 };
    }
  }
  if (states.thumb && !states.index && !states.middle && !states.ring && states.pinky) {
    return { label: "call", confidence: 0.65 };
  }
  if (states.index && states.middle && states.ring && !states.pinky) {
    return { label: "three", confidence: 0.65 };
  }
  if (extended <= 1) {
    return { label: "fist", confidence: 0.6 };
  }
  return { label: "open_palm", confidence: 0.4 };
}
