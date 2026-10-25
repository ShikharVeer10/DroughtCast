"""Run fixed seeds across [42, 101, 2024, 2025, 2026], validation selection, held-out metrics, and baseline comparisons."""
import argparse
import subprocess
import sys
from pathlib import Path


def run(*args):
    subprocess.run([sys.executable, *args], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024, 2025, 2026])
    args = parser.parse_args()
    if not args.skip_build and not Path("data/monthly_sites.csv").exists():
        run("-m", "data.build_dataset", "--start", "1981", "--end", "2024")
    Path("results/seeds").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        checkpoint = f"results/seeds/model_seed_{seed}.pt"
        report = f"results/seeds/metrics_seed_{seed}.json"
        print(f"\n=================== RUNNING SEED {seed} ===================")
        run("train.py", "--seed", str(seed), "--checkpoint", checkpoint)
        run("evaluate.py", "--seed", str(seed), "--checkpoint", checkpoint, "--output", report)
    run("-m", "experiments.summarize_runs", *[str(x) for x in args.seeds])
