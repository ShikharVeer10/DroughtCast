"""Run fixed seeds across [42, 101, 2024, 2025, 2026], validation selection, held-out metrics, and baseline comparisons."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run(*args):
    subprocess.run([sys.executable, *args], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--results-dir", default="results/maharashtra/seeds")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 101, 2024, 2025, 2026]
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_path = Path(config["data"]["path"])
    if not args.skip_build and not dataset_path.exists():
        run("-m", "data.build_dataset", "--config", args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        checkpoint = str(results_dir / f"model_seed_{seed}.pt")
        report = str(results_dir / f"metrics_seed_{seed}.json")
        print(f"\n=================== RUNNING SEED {seed} ===================")
        run(
            "train.py",
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--checkpoint",
            checkpoint,
        )
        run(
            "evaluate.py",
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--checkpoint",
            checkpoint,
            "--output",
            report,
        )
    run(
        "-m",
        "experiments.summarize_runs",
        "--config",
        args.config,
        "--results-dir",
        str(results_dir),
        "--seeds",
        *[str(x) for x in args.seeds],
    )
