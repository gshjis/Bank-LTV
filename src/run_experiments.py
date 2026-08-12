"""Ночной последовательный запуск серии CatBoost-экспериментов.

Каждая конфигурация запускается отдельным процессом и отдельным MLflow run.
Ошибки одного запуска не останавливают всю серию, если не передан --stop-on-error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict


class ExperimentConfig(TypedDict):
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    auto_class_weights: str
    iterations: int
    early_stopping_rounds: int


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a hyperparameter experiment batch")
    parser.add_argument("--train-path", type=Path, default=project_dir / "data" / "response_train.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=project_dir / "artifacts" / "experiments")
    parser.add_argument("--experiment-name", default="kaggle-ltv-nightly")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=0, help="0 — запустить все конфигурации")
    parser.add_argument("--start-index", type=int, default=0, help="Индекс для продолжения прерванной серии")
    parser.add_argument("--shuffle", action="store_true", help="Перемешать конфигурации детерминированно")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_grid() -> list[ExperimentConfig]:
    """Умеренная сетка: 72 запуска, подходящих для ночного batch."""
    grid = itertools.product(
        [3, 4, 5, 6],
        [0.003, 0.005, 0.01],
        [5.0, 10.0, 20.0],
        ["Balanced", "None"],
    )
    return [
        {
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "auto_class_weights": auto_class_weights,
            "iterations": 1500,
            "early_stopping_rounds": 100,
        }
        for depth, learning_rate, l2_leaf_reg, auto_class_weights in grid
    ]


def command_for(
    config: ExperimentConfig, args: argparse.Namespace, index: int
) -> list[str]:
    train_script = Path(__file__).with_name("train.py")
    command = [
        sys.executable,
        str(train_script),
        "--train-path",
        str(args.train_path),
        "--artifacts-dir",
        str(args.artifacts_dir / f"run_{index:03d}"),
        "--experiment-name",
        args.experiment_name,
        "--run-name",
        f"grid-{index:03d}",
        "--seed",
        str(args.seed + index),
    ]
    if args.tracking_uri:
        command.extend(["--tracking-uri", args.tracking_uri])
    for name, value in config.items():
        option = "--" + name.replace("_", "-")
        command.extend([option, str(value)])
    return command


def main() -> None:
    args = parse_args()
    configs = build_grid()
    if args.shuffle:
        random.Random(args.seed).shuffle(configs)
    if args.start_index < 0 or args.start_index >= len(configs):
        raise ValueError(f"start-index должен быть в диапазоне [0, {len(configs) - 1}]")

    selected = configs[args.start_index :]
    if args.max_runs > 0:
        selected = selected[: args.max_runs]
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.artifacts_dir / "batch_summary.jsonl"

    print(f"Всего конфигураций: {len(configs)}; к запуску: {len(selected)}")
    for offset, config in enumerate(selected):
        index = args.start_index + offset
        command = command_for(config, args, index)
        print(f"\n[{index + 1}/{len(configs)}] {' '.join(command)}")
        if args.dry_run:
            continue

        started_at = time.time()
        completed = subprocess.run(command, check=False)
        record = {
            "index": index,
            "config": config,
            "return_code": completed.returncode,
            "duration_seconds": round(time.time() - started_at, 2),
        }
        with summary_path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if completed.returncode != 0 and args.stop_on_error:
            raise RuntimeError(f"Эксперимент {index} завершился с кодом {completed.returncode}")

    print(f"\nСводка запусков: {summary_path}")


if __name__ == "__main__":
    main()
