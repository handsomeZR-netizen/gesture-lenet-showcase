"""FastAPI + pyautogui bridge for gesture-driven desktop control."""

DEFAULT_BINDINGS = {
    "open_palm": {"action": "release", "enabled": True},
    "point": {"action": "move_cursor", "enabled": True},
    "pinch": {"action": "click_or_drag", "enabled": True},
    "fist": {"action": "press_escape", "enabled": True},
    "victory": {"action": "alt_tab", "enabled": True},
    "ok": {"action": "media_playpause", "enabled": True},
    "thumbs_up": {"action": "volume_up", "enabled": True},
    "thumbs_down": {"action": "volume_down", "enabled": True},
    "three": {"action": "show_desktop", "enabled": True},
    "call": {"action": "media_next", "enabled": True},
    "swipe_up": {"action": "scroll_up", "enabled": True},
    "swipe_down": {"action": "scroll_down", "enabled": True},
    "swipe_left": {"action": "media_prev", "enabled": False},
    "swipe_right": {"action": "media_next", "enabled": False},
}

ACTION_LABELS_CN = {
    "release": "释放（无动作）",
    "move_cursor": "移动鼠标",
    "click_or_drag": "单击 / 长捏拖拽",
    "press_escape": "Esc / 取消",
    "alt_tab": "切换窗口（Alt+Tab）",
    "media_playpause": "播放 / 暂停",
    "volume_up": "音量 +",
    "volume_down": "音量 -",
    "show_desktop": "显示桌面",
    "media_next": "下一首",
    "media_prev": "上一首",
    "scroll_up": "向上滚动",
    "scroll_down": "向下滚动",
    "minimize": "最小化窗口",
    "close_window": "关闭窗口（Alt+F4）",
    "noop": "禁用",
}

AVAILABLE_ACTIONS = list(ACTION_LABELS_CN.keys())
