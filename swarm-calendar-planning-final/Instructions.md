# ACO-ToT CLI Instructions

This file consolidates the current command-line workflows for training, inference, logs, plotting, and best-path analysis for the calendar scheduling ACO-ToT framework.

Run location:
- Run all commands below from below location

PowerShell:

```powershell
cd ".\Planner-Agent\swarm-calendar-planning-final\Aco-ToT-multi-agent-framework"
```

## 1. Training CLI

Use [train_aco_tot_calendar.py](</C:/IMON/Masters/DKE Course/Semester 4/Scientific Project/Planner-Agent/swarm-calendar-planning-final/Aco-ToT-multi-agent-framework/train_aco_tot_calendar.py>).

```powershell
python train_aco_tot_calendar.py --mode train --data_path ../data/calendar_scheduling_input.json --out_path ./pheromones.json --pheromone_path ./pheromones.json --model qwen/qwen-2.5-7b-instruct --agents_per_task 8 --epochs 1 --alpha 1.0 --beta 2.0 --epsilon 0.001 --rho 0.1 --seed 42 --aco true --metrics_out_path ./metrics/train_iteration_metrics.csv
```

Relevant args:
- `--data_path`: input dataset JSON
- `--out_path`: output pheromone JSON path
- `--pheromone_path`: optional override; if omitted, `out_path` is used
- `--model`
- `--agents_per_task`
- `--epochs`
- `--alpha`, `--beta`, `--epsilon`, `--rho`
- `--seed`
- `--aco`: `true/false`, also accepts `1/0`, `yes/no`, `y/n`
- `--metrics_out_path`: CSV for training metrics

Expected outputs:
- Printed JSON summary on stdout
- Pheromone file: `./pheromones.json`
- Metrics CSV: `./metrics/train_iteration_metrics.csv`
- JSONL run log: `./logs/train_run_<timestamp>.jsonl`

The training summary JSON includes:
- `mode`
- `tasks`
- `epochs`
- `agents_per_task`
- `aco_enabled`
- `pheromone_updates_applied`
- `quality_proxy_mean`
- `pheromone_path`
- `iteration_metrics_path`
- `final_cumulative_accuracy`
- `final_cumulative_mean_quality`
- `run_log_path`

Mode matrix:
- `train + aco=true`: ACO-guided traversal with pheromone updates across tasks.
- `train + aco=false`: random ToT traversal, no pheromone updates.
- `infer + aco=true`: ACO-guided traversal using loaded pheromones.
- `infer + aco=false`: random ToT traversal, loaded pheromones ignored.
- `solve + aco=true`: per-task local pheromones, `n` agents for `T` iterations, deterministic scoring, final prediction from final iteration.
- `solve + aco=false`: same `n*T` budget, random ToT only, deterministic scoring, no pheromone updates.

## 2. Inference / Solve CLI

Use [infer_aco_tot_calendar.py](</C:/IMON/Masters/DKE Course/Semester 4/Scientific Project/Planner-Agent/swarm-calendar-planning-final/Aco-ToT-multi-agent-framework/infer_aco_tot_calendar.py>).

Inference (legacy global-pheromone inference):

```powershell
python infer_aco_tot_calendar.py --mode infer --data_path ../data/calendar_scheduling_input.json --out_path ../data/calendar_scheduling_output.json --pheromone_path ./pheromones.json --model qwen/qwen-2.5-7b-instruct --agents_per_task 8 --alpha 1.0 --beta 2.0 --epsilon 0.001 --rho 0.1 --seed 42 --aco true
```

Solve (recommended per-instance ACO):

```powershell
python infer_aco_tot_calendar.py --mode solve --data_path ../data/calendar_scheduling_input.json --out_path ../data/calendar_scheduling_output.json --model qwen/qwen-2.5-7b-instruct --agents_per_task 8 --iterations_per_task 3 --alpha 1.0 --beta 2.0 --epsilon 0.001 --rho 0.1 --seed 42 --aco true
```

Relevant args:
- `--mode`: `infer` or `solve`
- `--data_path`
- `--out_path`
- `--pheromone_path` (used for `infer`; ignored by `solve`)
- `--model`
- `--agents_per_task`
- `--iterations_per_task` (only for `solve`)
- `--alpha`, `--beta`, `--epsilon`, `--rho`
- `--seed`
- `--aco`

Expected outputs:
- Output dataset JSON at `--out_path`
- JSONL run log:
- `infer`: `./logs/infer_run_<timestamp>.jsonl`
- `solve`: `<out_path parent>/logs/solve_run_<timestamp>.jsonl`

Per dataset item, inference writes:
- `pred_5shot_pro`
- `pred_model`
- `aco_tot_agent_count`

Per dataset item, solve additionally writes:
- `aco_tot_mode=solve`
- `aco_tot_iterations_per_task`
- `aco_tot_scoring_mode=deterministic_task_based`
- `aco_tot_selection_policy=final_iteration_best`
- `aco_tot_aco_enabled`
- `aco_tot_final_iteration_best_quality`
- `aco_tot_final_iteration_best_slot_rank`
- `aco_tot_pheromone_updates_applied`

Solve behavior contract:
- generation prompt: `prompt_5shot`
- deterministic scorer prompt: `prompt_0shot` (fallback to current-task extraction from `prompt_5shot`)
- official prediction: best trace from **final iteration only**
- `--aco false`: random ToT traversal, no pheromone updates (`pheromone_updates_applied=0`)

## 3. JSONL Logs

Default log location:
- Training logs: [logs/train_run_*.jsonl](</C:/IMON/Masters/DKE Course/Semester 4/Scientific Project/Planner-Agent/swarm-calendar-planning-final/Aco-ToT-multi-agent-framework/logs>)
- Inference logs: [logs/infer_run_*.jsonl](</C:/IMON/Masters/DKE Course/Semester 4/Scientific Project/Planner-Agent/swarm-calendar-planning-final/Aco-ToT-multi-agent-framework/logs>)

Each log contains structured JSONL events:
- `run_start`
- `task_result`
- `run_end`

Training task rows include traces, prediction, quality statistics, prompt context, and `aco_enabled`.
Inference task rows include traces, prediction, and `aco_enabled`.

## 4. Accuracy and Quality Plots

The plotting utilities live in [aco_tot/plotting.py](</C:/IMON/Masters/DKE Course/Semester 4/Scientific Project/Planner-Agent/swarm-calendar-planning-final/Aco-ToT-multi-agent-framework/aco_tot/plotting.py>), and the CLI wrapper is [plot_training_accuracy.py](</C:/IMON/Masters/DKE Course/Semester 4/Scientific Project/Planner-Agent/swarm-calendar-planning-final/Aco-ToT-multi-agent-framework/plot_training_accuracy.py>).

Accuracy plot:

```powershell
python plot_training_accuracy.py --metric accuracy --metrics_csv_path ./metrics/train_iteration_metrics.csv --out_path ./metrics/iteration_vs_accuracy.png --title "Iteration vs Accuracy"
```

Quality plot:

```powershell
python plot_training_accuracy.py --metric quality --metrics_csv_path ./metrics/train_iteration_metrics.csv --out_path ./metrics/iteration_vs_quality.png --title "Iteration vs Quality"
```

Expected outputs:
- PNG file at `--out_path`
- Printed output path on stdout

Note:
- `plotting.py` itself is a utility module, not a standalone CLI script.
- `matplotlib` must be installed for plot generation.

## 5. Best-Path Analysis

Use [analyze_best_paths.py](</C:/IMON/Masters/DKE Course/Semester 4/Scientific Project/Planner-Agent/swarm-calendar-planning-final/Aco-ToT-multi-agent-framework/analyze_best_paths.py>).

Default run against the latest training log:

```powershell
python analyze_best_paths.py
```

Explicit run:

```powershell
python analyze_best_paths.py --log_path ./logs/train_run_YYYYMMDD_HHMMSS_microseconds.jsonl --logs_dir ./logs --output_dir ./analysis --top_k 10
```

Disable plots:

```powershell
python analyze_best_paths.py --no_plots
```

Expected outputs in `./analysis`:
- `path_leaderboard.csv`
- `branch_leaderboard.csv`
- `top_paths.json`
- optionally:
- `top_paths_mean_quality.png`
- `branch_quality_by_level.png`

Expected console output:
- `Top paths:` ranked summary
- `Outputs:` section listing generated files
