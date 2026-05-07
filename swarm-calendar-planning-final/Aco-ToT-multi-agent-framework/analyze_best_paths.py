#!/usr/bin/env python3
"""Analyze best-quality ToT paths from a training run JSONL log."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EXACT_THRESHOLD = 0.999999


@dataclass
class Aggregate:
    count: int = 0
    quality_sum: float = 0.0
    max_quality: float = 0.0
    path_score_sum: float = 0.0
    exact_count: int = 0

    def update(self, *, quality: float, path_score: float) -> None:
        self.count += 1
        self.quality_sum += quality
        self.max_quality = max(self.max_quality, quality)
        self.path_score_sum += path_score
        if quality >= EXACT_THRESHOLD:
            self.exact_count += 1

    def mean_quality(self) -> float:
        return self.quality_sum / float(self.count) if self.count else 0.0

    def mean_path_score(self) -> float:
        return self.path_score_sum / float(self.count) if self.count else 0.0

    def exact_hit_rate(self) -> float:
        return self.exact_count / float(self.count) if self.count else 0.0


def _find_latest_train_log(logs_dir: Path) -> Path:
    candidates = sorted(logs_dir.glob("train_run_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No train logs found under: {logs_dir} (expected train_run_*.jsonl)"
        )
    return candidates[-1]


def _parse_level_and_branch(edge: str) -> Tuple[int, str]:
    # Expected edge format: L<level>:<branch_id>
    left, branch = edge.split(":", 1)
    level = int(left[1:])
    return level, branch


def _iter_train_task_results(log_path: Path) -> Iterable[dict]:
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") != "task_result":
                continue
            if row.get("mode") != "train":
                continue
            yield row


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _try_make_plots(
    *,
    output_dir: Path,
    top_path_rows: List[dict],
    branch_rows: List[dict],
    top_k: int,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_paths: List[str] = []

    # Plot 1: top-k paths by mean quality
    selected = top_path_rows[:top_k]
    if selected:
        labels = [row["path"] for row in selected]
        values = [float(row["mean_quality"]) for row in selected]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(values)), values, color="#1f77b4")
        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Mean Quality")
        plt.title(f"Top {len(selected)} Paths by Mean Quality")
        plt.tight_layout()
        path_plot = output_dir / "top_paths_mean_quality.png"
        plt.savefig(path_plot, dpi=160)
        plt.close()
        plot_paths.append(str(path_plot))

    # Plot 2: branch quality by level (one subplot per level)
    by_level: Dict[int, List[dict]] = defaultdict(list)
    for row in branch_rows:
        by_level[int(row["level"])].append(row)
    if by_level:
        levels = sorted(by_level.keys())
        fig, axes = plt.subplots(
            nrows=len(levels),
            ncols=1,
            figsize=(11, max(3, 2.6 * len(levels))),
            squeeze=False,
        )
        for idx, level in enumerate(levels):
            ax = axes[idx][0]
            rows = sorted(by_level[level], key=lambda r: float(r["mean_quality_when_used"]), reverse=True)
            labels = [row["branch_id"] for row in rows]
            values = [float(row["mean_quality_when_used"]) for row in rows]
            ax.bar(range(len(values)), values, color="#2ca02c")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Mean Q")
            ax.set_title(f"Level {level} Branch Quality")
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=0, fontsize=9)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        branch_plot = output_dir / "branch_quality_by_level.png"
        fig.savefig(branch_plot, dpi=160)
        plt.close(fig)
        plot_paths.append(str(branch_plot))

    return plot_paths


def analyze_best_paths(
    *,
    log_path: Path,
    output_dir: Path,
    top_k: int,
    make_plots: bool,
) -> dict:
    path_stats: Dict[str, Aggregate] = defaultdict(Aggregate)
    branch_stats: Dict[str, Aggregate] = defaultdict(Aggregate)
    tasks_seen = 0
    traces_seen = 0

    for task_row in _iter_train_task_results(log_path):
        tasks_seen += 1
        traces = task_row.get("traces", [])
        for trace in traces:
            traces_seen += 1
            quality = float(trace.get("quality_score", 0.0))
            path_score = float(trace.get("path_score", 0.0))
            path = trace.get("path", [])
            if not isinstance(path, list) or not path:
                continue
            path_str = " -> ".join(path)
            path_stats[path_str].update(quality=quality, path_score=path_score)

            for edge in path:
                branch_stats[str(edge)].update(quality=quality, path_score=path_score)

    path_rows: List[dict] = []
    for path, agg in path_stats.items():
        path_rows.append(
            {
                "path": path,
                "count": agg.count,
                "mean_quality": round(agg.mean_quality(), 6),
                "max_quality": round(agg.max_quality, 6),
                "mean_path_score": round(agg.mean_path_score(), 6),
                "exact_hit_rate": round(agg.exact_hit_rate(), 6),
            }
        )

    path_rows.sort(
        key=lambda row: (
            -float(row["mean_quality"]),
            -float(row["exact_hit_rate"]),
            -int(row["count"]),
            row["path"],
        )
    )

    branch_rows: List[dict] = []
    for edge, agg in branch_stats.items():
        level, branch_id = _parse_level_and_branch(edge)
        branch_rows.append(
            {
                "edge": edge,
                "level": level,
                "branch_id": branch_id,
                "usage_count": agg.count,
                "mean_quality_when_used": round(agg.mean_quality(), 6),
                "exact_hit_rate_when_used": round(agg.exact_hit_rate(), 6),
            }
        )

    branch_rows.sort(
        key=lambda row: (
            int(row["level"]),
            -float(row["mean_quality_when_used"]),
            -float(row["exact_hit_rate_when_used"]),
            -int(row["usage_count"]),
            row["branch_id"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path_csv = output_dir / "path_leaderboard.csv"
    branch_csv = output_dir / "branch_leaderboard.csv"
    top_json = output_dir / "top_paths.json"

    _write_csv(
        path_csv,
        [
            "path",
            "count",
            "mean_quality",
            "max_quality",
            "mean_path_score",
            "exact_hit_rate",
        ],
        path_rows,
    )
    _write_csv(
        branch_csv,
        [
            "edge",
            "level",
            "branch_id",
            "usage_count",
            "mean_quality_when_used",
            "exact_hit_rate_when_used",
        ],
        branch_rows,
    )

    top_payload = {
        "log_path": str(log_path),
        "tasks_seen": tasks_seen,
        "traces_seen": traces_seen,
        "top_k": top_k,
        "top_paths": path_rows[:top_k],
    }
    with top_json.open("w", encoding="utf-8") as handle:
        json.dump(top_payload, handle, ensure_ascii=False, indent=2)

    plot_files: List[str] = []
    if make_plots:
        plot_files = _try_make_plots(
            output_dir=output_dir,
            top_path_rows=path_rows,
            branch_rows=branch_rows,
            top_k=top_k,
        )

    return {
        "log_path": str(log_path),
        "tasks_seen": tasks_seen,
        "traces_seen": traces_seen,
        "path_leaderboard_csv": str(path_csv),
        "branch_leaderboard_csv": str(branch_csv),
        "top_paths_json": str(top_json),
        "plot_files": plot_files,
        "top_paths": path_rows[:top_k],
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    framework_root = project_root / "Aco-ToT-multi-agent-framework"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_path",
        default=None,
        help="Path to train_run_*.jsonl. If omitted, latest file in logs/ is used.",
    )
    parser.add_argument(
        "--logs_dir",
        default=str(framework_root / "logs"),
        help="Directory containing train_run_*.jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(framework_root / "analysis"),
        help="Directory to write analysis outputs.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top paths to print/store in top_paths.json.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip optional matplotlib plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    log_path = Path(args.log_path) if args.log_path else _find_latest_train_log(logs_dir)
    output_dir = Path(args.output_dir)

    result = analyze_best_paths(
        log_path=log_path,
        output_dir=output_dir,
        top_k=args.top_k,
        make_plots=not args.no_plots,
    )

    print("\nTop paths:")
    for rank, row in enumerate(result["top_paths"], start=1):
        print(
            f"{rank:>2}. mean_q={row['mean_quality']:.4f} "
            f"exact_rate={row['exact_hit_rate']:.4f} "
            f"count={row['count']} | {row['path']}"
        )

    print("\nOutputs:")
    print(f"- path leaderboard : {result['path_leaderboard_csv']}")
    print(f"- branch leaderboard: {result['branch_leaderboard_csv']}")
    print(f"- top paths json   : {result['top_paths_json']}")
    if result["plot_files"]:
        for plot_path in result["plot_files"]:
            print(f"- plot            : {plot_path}")
    else:
        print("- plot            : none")


if __name__ == "__main__":
    main()

