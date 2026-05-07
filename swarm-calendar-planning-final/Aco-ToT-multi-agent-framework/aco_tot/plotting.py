"""Utilities for plotting training metrics."""

from __future__ import annotations

import csv
from pathlib import Path


def _read_metric_series(
    metrics_csv_path: str,
    y_column: str,
) -> tuple[list[int], list[float]]:
    iterations: list[int] = []
    y_values: list[float] = []

    with Path(metrics_csv_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iterations.append(int(row["iteration"]))
            y_values.append(float(row[y_column]))

    if not iterations:
        raise ValueError("Metrics CSV is empty; cannot build plot.")
    return iterations, y_values


def plot_iteration_vs_accuracy(
    metrics_csv_path: str,
    output_path: str,
    *,
    title: str = "Iteration vs Accuracy",
) -> str:
    """
    Plot cumulative accuracy over training iterations.

    Expects CSV columns:
      - iteration
      - cumulative_accuracy
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required to generate the training accuracy plot. "
            "Install it in your environment: pip install matplotlib"
        ) from exc

    iterations, cumulative_accuracy = _read_metric_series(
        metrics_csv_path=metrics_csv_path,
        y_column="cumulative_accuracy",
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, cumulative_accuracy, color="#1f77b4", linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()
    return str(output_file)


def plot_iteration_vs_quality(
    metrics_csv_path: str,
    output_path: str,
    *,
    title: str = "Iteration vs Quality",
) -> str:
    """
    Plot cumulative mean quality over training iterations.

    Expects CSV columns:
      - iteration
      - cumulative_mean_quality
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required to generate the training quality plot. "
            "Install it in your environment: pip install matplotlib"
        ) from exc

    iterations, cumulative_mean_quality = _read_metric_series(
        metrics_csv_path=metrics_csv_path,
        y_column="cumulative_mean_quality",
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, cumulative_mean_quality, color="#2ca02c", linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel("Quality")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()
    return str(output_file)
