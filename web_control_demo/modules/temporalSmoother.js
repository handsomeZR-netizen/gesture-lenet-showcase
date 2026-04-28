// Temporal post-processing: votes recent labels and tracks wrist trajectory
// to detect dynamic swipe gestures. Designed to keep UI labels stable while
// still firing actions promptly.

const DEFAULT_WINDOW = 7;
const DEFAULT_HYSTERESIS_ENTER = 0.55;
const DEFAULT_HYSTERESIS_EXIT = 0.4;
const SWIPE_BUFFER_MS = 450;
const SWIPE_DISTANCE_THRESHOLD = 0.22;
const SWIPE_COOLDOWN_MS = 700;

export class LabelSmoother {
  constructor({
    windowSize = DEFAULT_WINDOW,
    enterThreshold = DEFAULT_HYSTERESIS_ENTER,
    exitThreshold = DEFAULT_HYSTERESIS_EXIT,
  } = {}) {
    this.windowSize = windowSize;
    this.enterThreshold = enterThreshold;
    this.exitThreshold = exitThreshold;
    this.history = [];
    this.currentLabel = null;
    this.currentConfidence = 0;
  }

  push(label, confidence) {
    this.history.push({ label, confidence });
    if (this.history.length > this.windowSize) this.history.shift();

    const counts = {};
    for (const entry of this.history) {
      counts[entry.label] = (counts[entry.label] || 0) + 1;
    }
    let bestLabel = null;
    let bestCount = 0;
    for (const [label, count] of Object.entries(counts)) {
      if (count > bestCount) {
        bestLabel = label;
        bestCount = count;
      }
    }
    const ratio = bestCount / this.history.length;
    if (this.currentLabel !== bestLabel) {
      if (ratio >= this.enterThreshold) {
        this.currentLabel = bestLabel;
        this.currentConfidence = confidence;
      }
    } else {
      if (ratio < this.exitThreshold) {
        this.currentLabel = null;
        this.currentConfidence = 0;
      } else {
        this.currentConfidence = this.currentConfidence * 0.7 + confidence * 0.3;
      }
    }
    return {
      label: this.currentLabel,
      confidence: this.currentConfidence,
      ratio,
      bestRaw: bestLabel,
    };
  }

  reset() {
    this.history.length = 0;
    this.currentLabel = null;
    this.currentConfidence = 0;
  }
}

export class SwipeDetector {
  constructor({
    bufferMs = SWIPE_BUFFER_MS,
    distanceThreshold = SWIPE_DISTANCE_THRESHOLD,
    cooldownMs = SWIPE_COOLDOWN_MS,
  } = {}) {
    this.bufferMs = bufferMs;
    this.distanceThreshold = distanceThreshold;
    this.cooldownMs = cooldownMs;
    this.points = [];
    this.lastFireAt = 0;
  }

  observe(point, ts) {
    this.points.push({ x: point.x, y: point.y, ts });
    const cutoff = ts - this.bufferMs;
    while (this.points.length > 0 && this.points[0].ts < cutoff) {
      this.points.shift();
    }
    if (this.points.length < 5) return null;
    if (ts - this.lastFireAt < this.cooldownMs) return null;
    const first = this.points[0];
    const last = this.points[this.points.length - 1];
    const dx = last.x - first.x;
    const dy = last.y - first.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < this.distanceThreshold) return null;
    this.lastFireAt = ts;
    this.points.length = 0;
    if (Math.abs(dx) > Math.abs(dy)) {
      return dx > 0 ? "swipe_right" : "swipe_left";
    }
    return dy < 0 ? "swipe_up" : "swipe_down";
  }

  reset() {
    this.points.length = 0;
  }
}
