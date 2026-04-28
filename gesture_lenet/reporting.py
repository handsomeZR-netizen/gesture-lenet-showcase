"""Reporting helpers for course-facing experiment summaries."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

from gesture_lenet.data import build_metadata, load_sign_mnist_csv

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def load_json(path: str | Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def classification_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-class rows from sklearn's classification_report output."""
    report = metrics["classification_report"]
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if not isinstance(values, dict) or "support" not in values:
            continue
        if label.endswith("avg"):
            continue
        rows.append(
            {
                "label": str(label),
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return rows


def weakest_classes(metrics: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    rows = classification_rows(metrics)
    return sorted(rows, key=lambda row: (row["f1_score"], row["support"]))[:limit]


def strongest_classes(metrics: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    rows = classification_rows(metrics)
    return sorted(rows, key=lambda row: row["f1_score"], reverse=True)[:limit]


def save_per_class_metrics_csv(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = classification_rows(metrics)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "precision", "recall", "f1_score", "support"],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_confusions(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    pairs: list[dict[str, Any]] = []
    for true_index, row in enumerate(matrix):
        row_total = int(row.sum())
        if row_total == 0:
            continue
        for predicted_index, count in enumerate(row):
            if true_index == predicted_index or count == 0:
                continue
            pairs.append(
                {
                    "true": class_names[true_index],
                    "predicted": class_names[predicted_index],
                    "count": int(count),
                    "share_of_true": float(count / row_total),
                }
            )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:limit]


def plot_per_class_f1(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(classification_rows(metrics), key=lambda row: row["f1_score"])
    labels = [row["label"] for row in rows]
    values = [row["f1_score"] for row in rows]
    colors = ["#d95f02" if value < 0.75 else "#1b9e77" for value in values]

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(labels, values, color=colors)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Class")
    axis.set_ylabel("F1-score")
    axis.set_title("Per-class F1-score on Test Set")
    axis.grid(axis="y", alpha=0.25)
    for label in axis.get_xticklabels():
        label.set_rotation(0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_sample_grid(csv_path: str | Path, output_path: str | Path, max_classes: int = 24) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_labels, images = load_sign_mnist_csv(csv_path)
    metadata = build_metadata(raw_labels)
    selected_labels = metadata.raw_labels[:max_classes]
    columns = 6
    rows = math.ceil(len(selected_labels) / columns)

    figure, axes = plt.subplots(rows, columns, figsize=(10, 6.8))
    flat_axes = np.array(axes).reshape(-1)
    for axis in flat_axes:
        axis.axis("off")

    for axis, raw_label, class_name in zip(flat_axes, selected_labels, metadata.class_names):
        sample_indices = np.flatnonzero(raw_labels == raw_label)
        if len(sample_indices) == 0:
            continue
        image = images[int(sample_indices[0])]
        axis.imshow(image, cmap="gray")
        axis.set_title(class_name, fontsize=10)
        axis.axis("off")

    figure.suptitle("Sign Language MNIST Class Samples", y=0.98, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_scoreboard(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = metrics["classification_report"]
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    best_val_accuracy = float(summary["best_val_accuracy"])
    test_accuracy = float(metrics["accuracy"])
    generalization_gap = best_val_accuracy - test_accuracy

    figure = plt.figure(figsize=(14, 8))
    grid = figure.add_gridspec(2, 3, height_ratios=[0.9, 1.1], hspace=0.4, wspace=0.3)
    figure.suptitle("Gesture Recognition Course Showcase", fontsize=18, fontweight="bold")

    axis_cards = figure.add_subplot(grid[0, :])
    axis_cards.axis("off")
    cards = [
        ("Test Accuracy", test_accuracy, "#1b9e77"),
        ("Macro F1", float(macro["f1-score"]), "#7570b3"),
        ("Weighted F1", float(weighted["f1-score"]), "#66a61e"),
        ("Best Val Accuracy", best_val_accuracy, "#e6ab02"),
        ("Val-Test Gap", generalization_gap, "#d95f02"),
        ("Classes", float(summary["num_classes"]), "#1f78b4"),
    ]
    for index, (title, value, color) in enumerate(cards):
        x = 0.02 + index * 0.163
        axis_cards.add_patch(
            plt.Rectangle((x, 0.15), 0.145, 0.68, color="#f7f7f7", ec=color, lw=2)
        )
        display_value = f"{value:.2%}" if title != "Classes" else f"{int(value)}"
        axis_cards.text(x + 0.0725, 0.57, display_value, ha="center", fontsize=18, color=color)
        axis_cards.text(x + 0.0725, 0.34, title, ha="center", fontsize=10, color="#333333")

    epochs = range(1, len(history["train_accuracy"]) + 1)
    axis_curve = figure.add_subplot(grid[1, :2])
    axis_curve.plot(epochs, history["train_accuracy"], label="train acc", color="#1b9e77", lw=2)
    axis_curve.plot(epochs, history["val_accuracy"], label="val acc", color="#d95f02", lw=2)
    axis_curve.set_xlabel("Epoch")
    axis_curve.set_ylabel("Accuracy")
    axis_curve.set_ylim(0, 1.03)
    axis_curve.set_title("Training Progress")
    axis_curve.grid(alpha=0.25)
    axis_curve.legend()

    rows = sorted(classification_rows(metrics), key=lambda row: row["f1_score"])
    axis_f1 = figure.add_subplot(grid[1, 2])
    axis_f1.barh([row["label"] for row in rows], [row["f1_score"] for row in rows], color="#7570b3")
    axis_f1.set_xlim(0, 1.05)
    axis_f1.set_xlabel("F1-score")
    axis_f1.set_title("Class Stability")
    axis_f1.grid(axis="x", alpha=0.25)

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_course_report(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = metrics["classification_report"]
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    best_val_accuracy = float(summary["best_val_accuracy"])
    test_accuracy = float(metrics["accuracy"])
    generalization_gap = best_val_accuracy - test_accuracy
    final_train_accuracy = float(history["train_accuracy"][-1])
    final_val_accuracy = float(history["val_accuracy"][-1])
    architecture = str(summary.get("architecture", "lenet"))
    model_name = "改进 CNN" if architecture == "improved" else "LeNet 风格 CNN"
    weak = weakest_classes(metrics)
    strong = strongest_classes(metrics)

    weak_text = "\n".join(
        f"- `{row['label']}`: F1={row['f1_score']:.3f}, "
        f"precision={row['precision']:.3f}, recall={row['recall']:.3f}, support={row['support']}"
        for row in weak
    )
    strong_text = "\n".join(
        f"- `{row['label']}`: F1={row['f1_score']:.3f}" for row in strong
    )

    content = f"""# 计算机视觉课程实践展示报告

## 项目定位

本项目是一个手势识别课程实践，核心链路覆盖数据集读取、CNN 训练、测试集评估、混淆矩阵分析、单图推理和摄像头实时推理。当前 Python/CV 主线使用 Sign Language MNIST 的 24 类静态手势进行 {model_name} 分类，`J` 和 `Z` 因需要动态轨迹不在静态标签空间内。

## 当前实验结果

- 类别数：`{summary['num_classes']}`
- 模型结构：`{architecture}`
- 最佳验证准确率：`{best_val_accuracy:.2%}`
- 测试集准确率：`{test_accuracy:.2%}`
- Macro F1：`{float(macro['f1-score']):.3f}`
- Weighted F1：`{float(weighted['f1-score']):.3f}`
- 最后一轮训练准确率：`{final_train_accuracy:.2%}`
- 最后一轮验证准确率：`{final_val_accuracy:.2%}`
- 验证到测试泛化差距：`{generalization_gap:.2%}`

## 可展示亮点

- 完整训练闭环：`CSV -> Dataset -> CNN -> Checkpoint -> Metrics -> Confusion Matrix`。
- 有可量化结果：测试集准确率、Macro/Weighted F1、逐类别 precision/recall/F1 都已产出。
- 有失败分析：不仅展示准确率，也指出弱类别和潜在混淆来源，适合课程答辩说明改进方向。
- 有实时推理入口：`infer_camera.py` 结合 MediaPipe 手部检测做摄像头 ROI 过滤，避免背景帧被强行分类。
- 有可视化资产：训练曲线、混淆矩阵、类别样本网格和课程展示看板可直接放入 PPT。

## 强类别

{strong_text}

## 需要重点解释的弱类别

{weak_text}

这些类别通常更容易受到手型相似、低分辨率灰度输入、静态图片与真实摄像头画面差异的影响。答辩时建议把这部分作为“模型误差分析”和“后续优化方向”，比只报一个准确率更完整。

## 生成文件

- `course_showcase.png`：课程展示总览图
- `per_class_f1.png`：逐类别 F1 可视化
- `sample_grid.png`：24 类样本展示
- 评估输出目录中的 `confusion_matrix.png`：混淆矩阵
- 评估输出目录中的 `per_class_metrics.csv`：逐类别指标表
- 评估输出目录中的 `confusion_pairs.json`：主要易混类别

## 后续优化建议

- 增加真实摄像头采集数据，做课程场景自己的微调或少样本校准。
- 加入数据增强、Dropout/BatchNorm 或更轻量的现代 CNN，与 LeNet 做对照实验。
- 把静态分类和动态手势分开评估：静态字母分类看准确率，交互控制看响应延迟、误触发率和稳定性。
"""
    output_path.write_text(content, encoding="utf-8")
