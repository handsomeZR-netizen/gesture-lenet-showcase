// WebSocket client that pipes gesture events to the Python backend.
// Auto-reconnects with backoff; throttles per-action send rate so a
// 60Hz inference loop does not flood pyautogui.

const PER_ACTION_THROTTLE_MS = {
  default: 33,           // ~30Hz move events
  click_or_drag: 16,     // pinch must update fast for drag tracking
};

export class ControlClient extends EventTarget {
  constructor({
    url,
    reconnectDelayMs = 800,
    maxReconnectDelayMs = 8000,
  } = {}) {
    super();
    this.url = url;
    this.reconnectDelayMs = reconnectDelayMs;
    this.maxReconnectDelayMs = maxReconnectDelayMs;
    this._currentDelay = reconnectDelayMs;
    this.socket = null;
    this.connected = false;
    this._lastSendByAction = new Map();
    this._closedExplicitly = false;
    this.lastAck = null;
  }

  connect() {
    this._closedExplicitly = false;
    this._open();
  }

  close() {
    this._closedExplicitly = true;
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  _open() {
    try {
      this.socket = new WebSocket(this.url);
    } catch (error) {
      this._scheduleReconnect();
      return;
    }
    this.socket.addEventListener("open", () => {
      this.connected = true;
      this._currentDelay = this.reconnectDelayMs;
      this.dispatchEvent(new CustomEvent("connect"));
    });
    this.socket.addEventListener("message", (event) => {
      let payload = null;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }
      this.lastAck = payload;
      this.dispatchEvent(new CustomEvent("message", { detail: payload }));
    });
    this.socket.addEventListener("close", () => {
      this.connected = false;
      this.socket = null;
      this.dispatchEvent(new CustomEvent("disconnect"));
      if (!this._closedExplicitly) {
        this._scheduleReconnect();
      }
    });
    this.socket.addEventListener("error", () => {
      this.dispatchEvent(new CustomEvent("error"));
      // close will follow; reconnect handled there.
    });
  }

  _scheduleReconnect() {
    setTimeout(() => {
      if (this._closedExplicitly) return;
      this._open();
    }, this._currentDelay);
    this._currentDelay = Math.min(this._currentDelay * 1.6, this.maxReconnectDelayMs);
  }

  isReady() {
    return this.connected && this.socket && this.socket.readyState === WebSocket.OPEN;
  }

  sendGesture({ label, confidence, anchor, handedness, action }) {
    if (!this.isReady()) return false;
    const now = performance.now();
    const throttleKey = action || label;
    const limit = PER_ACTION_THROTTLE_MS[throttleKey] ?? PER_ACTION_THROTTLE_MS.default;
    const last = this._lastSendByAction.get(throttleKey) ?? 0;
    if (now - last < limit) return false;
    this._lastSendByAction.set(throttleKey, now);
    try {
      this.socket.send(
        JSON.stringify({
          type: "gesture",
          label,
          confidence,
          anchor: anchor ? [anchor.x, anchor.y] : null,
          handedness: handedness || "Right",
          ts: Date.now(),
        }),
      );
      return true;
    } catch (error) {
      return false;
    }
  }

  sendControl(action) {
    if (!this.isReady()) return false;
    try {
      this.socket.send(JSON.stringify({ type: "control", action }));
      return true;
    } catch (error) {
      return false;
    }
  }

  sendPing() {
    if (!this.isReady()) return false;
    try {
      this.socket.send(JSON.stringify({ type: "ping" }));
      return true;
    } catch (error) {
      return false;
    }
  }
}
