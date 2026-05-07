#!/usr/bin/env python3
"""Train or run inference for ACO-ToT calendar scheduling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from absl import app, flags

from dotenv import load_dotenv

load_dotenv()

FLAGS = flags.FLAGS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODE = flags.DEFINE_enum(
    "mode",
    "infer",
    ["train", "infer", "solve"],
    "Execution mode. train updates pheromones, infer writes pred_5shot_pro outputs, solve runs per-instance ACO.",
)
DATA_PATH = flags.DEFINE_string(
    "data_path",
    str(PROJECT_ROOT / "data" / "calendar_scheduling_input.json"),
    "Path to input dataset JSON.",
)
OUT_PATH = flags.DEFINE_string(
    "out_path",
    str(PROJECT_ROOT / "data" / "calendar_scheduling_output.json"),
    "Output JSON path (used in infer mode).",
)
PHEROMONE_PATH = flags.DEFINE_string(
    "pheromone_path",
    str(PROJECT_ROOT / "Aco-ToT-multi-agent-framework" / "pheromones.json"),
    "Pheromone state JSON path.",
)
MODEL = flags.DEFINE_string(
    "model",
    "qwen/qwen-2.5-7b-instruct",
    "Model name to use via OpenRouter ChatOpenAI.",
)
AGENTS_PER_TASK = flags.DEFINE_integer(
    "agents_per_task",
    8,
    "Number of agents per task.",
)
EPOCHS = flags.DEFINE_integer(
    "epochs",
    1,
    "Number of training epochs in train mode.",
)
ITERATIONS_PER_TASK = flags.DEFINE_integer(
    "iterations_per_task",
    3,
    "Number of per-task ACO iterations in solve mode.",
)
ALPHA = flags.DEFINE_float("alpha", 1.0, "ACO alpha (pheromone importance).")
BETA = flags.DEFINE_float("beta", 2.0, "ACO beta (heuristic importance).")
EPSILON = flags.DEFINE_float("epsilon", 0.001, "ACO epsilon smoothing constant.")
RHO = flags.DEFINE_float("rho", 0.1, "Pheromone evaporation rate.")
SEED = flags.DEFINE_integer("seed", 42, "Random seed.")
ACO = flags.DEFINE_boolean(
    "aco",
    True,
    "If true, use ACO-guided selection; if false, use random ToT selection and disable pheromone updates in training.",
)
METRICS_OUT_PATH = flags.DEFINE_string(
    "metrics_out_path",
    str(
        PROJECT_ROOT
        / "Aco-ToT-multi-agent-framework"
        / "metrics"
        / "train_iteration_metrics.csv"
    ),
    "CSV path for per-iteration training metrics (train mode).",
)


def _import_framework_api():
    framework_root = (
        Path(__file__).resolve().parents[1] / "Aco-ToT-multi-agent-framework"
    )
    framework_path = str(framework_root)
    if framework_path not in sys.path:
        sys.path.insert(0, framework_path)

    from aco_tot.invoke import infer_dataset, solve_dataset, train_dataset

    return train_dataset, infer_dataset, solve_dataset


def main(_: list[str]) -> None:
    train_dataset, infer_dataset, solve_dataset = _import_framework_api()

    if MODE.value == "train":
        summary = train_dataset(
            data_path=DATA_PATH.value,
            out_pheromone_path=PHEROMONE_PATH.value,
            model=MODEL.value,
            agents_per_task=AGENTS_PER_TASK.value,
            epochs=EPOCHS.value,
            alpha=ALPHA.value,
            beta=BETA.value,
            epsilon=EPSILON.value,
            rho=RHO.value,
            seed=SEED.value,
            aco_enabled=ACO.value,
            metrics_out_path=METRICS_OUT_PATH.value,
        )
        print(json.dumps(summary, indent=2))
        return

    if MODE.value == "solve":
        summary = solve_dataset(
            data_path=DATA_PATH.value,
            out_path=OUT_PATH.value,
            model=MODEL.value,
            agents_per_task=AGENTS_PER_TASK.value,
            iterations_per_task=ITERATIONS_PER_TASK.value,
            alpha=ALPHA.value,
            beta=BETA.value,
            epsilon=EPSILON.value,
            rho=RHO.value,
            seed=SEED.value,
            aco_enabled=ACO.value,
        )
        print(json.dumps(summary, indent=2))
        return

    infer_dataset(
        data_path=DATA_PATH.value,
        pheromone_path=PHEROMONE_PATH.value,
        out_path=OUT_PATH.value,
        model=MODEL.value,
        agents_per_task=AGENTS_PER_TASK.value,
        alpha=ALPHA.value,
        beta=BETA.value,
        epsilon=EPSILON.value,
        rho=RHO.value,
        seed=SEED.value,
        aco_enabled=ACO.value,
    )
    print(f"Inference complete. Output written to: {OUT_PATH.value}")


if __name__ == "__main__":
    app.run(main)
