// 网页内控制器：当后端不可达时，把手势映射到当前页面里能做的真实动作。
// 这样无需安装、无需后端，任何人打开 Cloudflare Pages 上的 URL 就能用手势
// 控制自己浏览器内的：
//   - 鼠标光标 + 点击 DOM 元素（在当前 tab 内）
//   - 滚动整页 / 滚动当前 hover 元素
//   - <video>/<audio> 的播放/暂停 / 音量 / 上下首
//   - 全屏切换、浏览器后退/前进
//
// 「整个 OS 级控制」必须本地启动 Python 后端，本模块只覆盖浏览器内能做的事。

const CURSOR_SIZE = 36;
const CURSOR_FILL = "rgba(255, 190, 92, 0.9)";
const CURSOR_RING = "rgba(255, 190, 92, 0.35)";
const SCROLL_STEP_PX = 120;

class GhostCursor {
  constructor() {
    this.el = null;
    this.x = 0;
    this.y = 0;
    this.visible = false;
    this._ensureElement();
  }

  _ensureElement() {
    if (this.el) return;
    const el = document.createElement("div");
    el.id = "gestureGhostCursor";
    Object.assign(el.style, {
      position: "fixed",
      width: `${CURSOR_SIZE}px`,
      height: `${CURSOR_SIZE}px`,
      borderRadius: "50%",
      pointerEvents: "none",
      zIndex: "2147483647",
      transform: "translate(-50%, -50%)",
      background: `radial-gradient(circle, ${CURSOR_FILL} 0%, ${CURSOR_FILL} 40%, ${CURSOR_RING} 70%, transparent 100%)`,
      transition: "left 60ms linear, top 60ms linear, opacity 200ms ease",
      boxShadow: "0 0 24px rgba(255, 190, 92, 0.45)",
      opacity: "0",
      left: "0px",
      top: "0px",
    });
    document.body.appendChild(el);
    this.el = el;
  }

  show() {
    this._ensureElement();
    if (!this.visible) {
      this.el.style.opacity = "1";
      this.visible = true;
    }
  }

  hide() {
    if (!this.el) return;
    this.el.style.opacity = "0";
    this.visible = false;
  }

  moveTo(x, y) {
    this._ensureElement();
    this.x = x;
    this.y = y;
    this.el.style.left = `${x}px`;
    this.el.style.top = `${y}px`;
    this.show();
  }

  pulse(color = "rgba(61, 215, 194, 0.95)") {
    this._ensureElement();
    const original = this.el.style.background;
    this.el.style.background = `radial-gradient(circle, ${color} 0%, ${color} 50%, transparent 80%)`;
    setTimeout(() => {
      this.el.style.background = original;
    }, 220);
  }
}

function findVideo() {
  const videos = Array.from(document.querySelectorAll("video, audio"));
  if (videos.length === 0) return null;
  // 优先返回正在播放或最大的一个
  const playing = videos.find((v) => !v.paused);
  if (playing) return playing;
  videos.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return br.width * br.height - ar.width * ar.height;
  });
  return videos[0];
}

function elementUnder(x, y) {
  return document.elementFromPoint(x, y);
}

export class InPageController {
  constructor() {
    this.cursor = new GhostCursor();
    this.cooldowns = new Map();
    this.lastClickAt = 0;
    this.dragActive = false;
    this.dragTarget = null;
    this.fullscreen = false;
    this.pinchStartedAt = null;
  }

  _cooldownReady(key, ms) {
    const now = performance.now();
    const until = this.cooldowns.get(key) || 0;
    if (now < until) return false;
    this.cooldowns.set(key, now + ms);
    return true;
  }

  /**
   * 入口：根据动作名 + 上下文执行真实网页内动作。
   * 返回 { ok, message } 给 UI 显示气泡。
   */
  handle(action, ctx = {}) {
    switch (action) {
      case "move_cursor":
        return this._move(ctx.anchor);
      case "click_or_drag":
        return this._pinch(ctx.anchor);
      case "release":
        return this._release();
      case "press_escape":
        return this._escape();
      case "alt_tab":
        return this._toggleFullscreen();
      case "media_playpause":
        return this._togglePlay();
      case "volume_up":
        return this._volume(+0.1);
      case "volume_down":
        return this._volume(-0.1);
      case "media_next":
        return this._mediaSeek(+10);
      case "media_prev":
        return this._mediaSeek(-10);
      case "show_desktop":
        return this._scrollTop();
      case "minimize":
        return this._scrollTop();
      case "close_window":
        return this._historyBack();
      case "scroll_up":
        return this._scroll(-SCROLL_STEP_PX);
      case "scroll_down":
        return this._scroll(+SCROLL_STEP_PX);
      default:
        return { ok: false, message: `不支持的动作 ${action}` };
    }
  }

  _move(anchor) {
    if (!anchor) return { ok: false, message: "缺少坐标" };
    const x = anchor.x * window.innerWidth;
    const y = anchor.y * window.innerHeight;
    this.cursor.moveTo(x, y);
    if (this.dragActive && this.dragTarget) {
      // 模拟拖拽期间的滚动
      const dy = y - (this._lastDragY || y);
      window.scrollBy(0, -dy * 0.5);
      this._lastDragY = y;
    }
    return { ok: true, message: `移动到 (${Math.round(x)}, ${Math.round(y)})` };
  }

  _pinch(anchor) {
    if (!anchor) return { ok: false, message: "缺少坐标" };
    const x = anchor.x * window.innerWidth;
    const y = anchor.y * window.innerHeight;
    this.cursor.moveTo(x, y);
    const now = performance.now();
    if (this.pinchStartedAt === null) {
      this.pinchStartedAt = now;
      return { ok: true, message: "捏合中" };
    }
    const duration = now - this.pinchStartedAt;
    if (duration > 420 && !this.dragActive) {
      this.dragActive = true;
      this.dragTarget = elementUnder(x, y);
      this._lastDragY = y;
      return { ok: true, message: "进入拖拽（沿 y 滚动）" };
    }
    return { ok: true, message: "捏合保持" };
  }

  _release() {
    if (this.dragActive) {
      this.dragActive = false;
      this.dragTarget = null;
      this._lastDragY = null;
    }
    if (this.pinchStartedAt !== null) {
      const duration = performance.now() - this.pinchStartedAt;
      this.pinchStartedAt = null;
      // 短捏 = 单击命中元素
      if (duration > 60 && duration < 420 && this._cooldownReady("click", 250)) {
        const target = elementUnder(this.cursor.x, this.cursor.y);
        if (target) {
          target.dispatchEvent(
            new MouseEvent("click", {
              bubbles: true,
              cancelable: true,
              clientX: this.cursor.x,
              clientY: this.cursor.y,
              view: window,
            }),
          );
          this.cursor.pulse();
          return { ok: true, message: `已点击 ${target.tagName.toLowerCase()}` };
        }
      }
    }
    this.cursor.hide();
    return { ok: true, message: "释放" };
  }

  _scroll(deltaY) {
    if (!this._cooldownReady("scroll", 60)) return { ok: false, message: "滚动冷却" };
    window.scrollBy({ top: deltaY, behavior: "smooth" });
    return { ok: true, message: deltaY > 0 ? "向下滚动" : "向上滚动" };
  }

  _scrollTop() {
    if (!this._cooldownReady("top", 800)) return { ok: false, message: "冷却中" };
    window.scrollTo({ top: 0, behavior: "smooth" });
    return { ok: true, message: "回到页首" };
  }

  _togglePlay() {
    if (!this._cooldownReady("play", 600)) return { ok: false, message: "播放冷却" };
    const v = findVideo();
    if (!v) return { ok: false, message: "页面没有视频/音频" };
    if (v.paused) {
      v.play().catch(() => {});
      return { ok: true, message: "已播放" };
    }
    v.pause();
    return { ok: true, message: "已暂停" };
  }

  _volume(delta) {
    if (!this._cooldownReady("volume", 200)) return { ok: false, message: "音量冷却" };
    const v = findVideo();
    if (!v) return { ok: false, message: "页面没有媒体" };
    v.volume = Math.max(0, Math.min(1, (v.volume || 0) + delta));
    return { ok: true, message: `音量 ${Math.round(v.volume * 100)}%` };
  }

  _mediaSeek(deltaSec) {
    if (!this._cooldownReady("seek", 400)) return { ok: false, message: "切歌冷却" };
    const v = findVideo();
    if (!v) return { ok: false, message: "页面没有媒体" };
    v.currentTime = Math.max(0, (v.currentTime || 0) + deltaSec);
    return { ok: true, message: deltaSec > 0 ? "快进 10s" : "倒退 10s" };
  }

  _toggleFullscreen() {
    if (!this._cooldownReady("fs", 1100)) return { ok: false, message: "全屏冷却" };
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.().catch(() => {});
      this.fullscreen = true;
      return { ok: true, message: "已进入全屏" };
    }
    document.exitFullscreen?.().catch(() => {});
    this.fullscreen = false;
    return { ok: true, message: "已退出全屏" };
  }

  _escape() {
    if (!this._cooldownReady("esc", 800)) return { ok: false, message: "Esc 冷却" };
    if (document.fullscreenElement) {
      document.exitFullscreen?.().catch(() => {});
      return { ok: true, message: "退出全屏" };
    }
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", code: "Escape", bubbles: true }),
    );
    return { ok: true, message: "Esc 已发送给当前页" };
  }

  _historyBack() {
    if (!this._cooldownReady("back", 1500)) return { ok: false, message: "后退冷却" };
    history.back();
    return { ok: true, message: "浏览器后退" };
  }
}
