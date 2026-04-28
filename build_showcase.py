#!/usr/bin/env python3
"""Build course-facing visual assets and summary report for the CV project."""

from __future__ import annotations

import argparse
from pathlib import Path

from gesture_lenet.reporting import (
    load_json,
    plot_per_class_f1,
    plot_sample_grid,
    plot_scoreboard,
    render_course_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/improved_train/summary.json")
    parser.add_argument("--metrics", default="outputs/improved_eval/metrics.json")
    parser.add_argument("--history", default="outputs/improved_train/history.json")
    parser.add_argument("--sample-csv", default="data/raw/sign_mnist_test.csv")
    parser.add_argument("--output-dir", default="outputs/showcase")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(args.summary)
    metrics = load_json(args.metrics)
    history = load_json(args.history)

    plot_scoreboard(
        summary=summary,
        metrics=metrics,
        history=history,
        output_path=output_dir / "course_showcase.png",
    )
    plot_per_class_f1(metrics=metrics, output_path=output_dir / "per_class_f1.png")
    plot_sample_grid(csv_path=args.sample_csv, output_path=output_dir / "sample_grid.png")
    render_course_report(
        summary=summary,
        metrics=metrics,
        history=history,
        output_path=output_dir / "course_report.md",
    )

    print(f"Showcase assets written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
