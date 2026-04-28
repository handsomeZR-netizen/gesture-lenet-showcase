"""Lightweight 21-keypoint MLP gesture classifier."""

GESTURE_LABELS = [
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
]

GESTURE_LABELS_CN = {
    "open_palm": "张开手掌",
    "point": "食指指向",
    "pinch": "捏合",
    "fist": "握拳",
    "victory": "V 字",
    "ok": "OK 圈",
    "thumbs_up": "拇指向上",
    "thumbs_down": "拇指向下",
    "three": "三指",
    "call": "电话手势",
}

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(GESTURE_LABELS)}
NUM_CLASSES = len(GESTURE_LABELS)
FEATURE_DIM = 63
