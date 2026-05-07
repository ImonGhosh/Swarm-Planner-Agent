#!/usr/bin/env python3
"""Generate Iteration-vs-Accuracy or Iteration-vs-Quality graph from metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from aco_tot.plotting import plot_iteration_vs_accuracy, plot_iteration_vs_quality


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    framework_root = project_root / "Aco-ToT-multi-agent-framework"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        choices=["accuracy", "quality"],
        default="accuracy",
        help="Metric to plot against iteration.",
    )
    parser.add_argument(
        "--metrics_csv_path",
        default=str(framework_root / "metrics" / "train_iteration_metrics.csv"),
        help="Path to training metrics CSV produced during training.",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help=(
            "Output PNG path for the graph. If omitted, uses "
            "metrics/iteration_vs_accuracy.png or metrics/iteration_vs_quality.png."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    framework_root = project_root / "Aco-ToT-multi-agent-framework"

    if args.metric == "accuracy":
        output_path = args.out_path or str(
            framework_root / "metrics" / "iteration_vs_accuracy.png"
        )
        title = args.title or "Iteration vs Accuracy"
        output = plot_iteration_vs_accuracy(
            metrics_csv_path=args.metrics_csv_path,
            output_path=output_path,
            title=title,
        )
    else:
        output_path = args.out_path or str(
            framework_root / "metrics" / "iteration_vs_quality.png"
        )
        title = args.title or "Iteration vs Quality"
        output = plot_iteration_vs_quality(
            metrics_csv_path=args.metrics_csv_path,
            output_path=output_path,
            title=title,
        )
    print(output)


if __name__ == "__main__":
    main()
