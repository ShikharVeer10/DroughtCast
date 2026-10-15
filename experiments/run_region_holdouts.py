"""Run a resumable leave-one-region-out grid with source-only fitting."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    regions = args.regions or sorted({site["region"] for site in config["sites"].values()})
    root = Path("results/region_holdouts")
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"seeds": args.seeds, "regions": regions,
        "protocol": "leave-region-out; train/validation on source regions only; heldout region test only"}, indent=2), encoding="utf-8")
    known = {site["region"] for site in config["sites"].values()}
    unknown = set(regions) - known
    if unknown:
        raise ValueError(f"Unknown region(s): {sorted(unknown)}")
    for region in regions:
        folder = root / region
        folder.mkdir(exist_ok=True)
        for seed in args.seeds:
            checkpoint, report = folder / f"model_seed_{seed}.pt", folder / f"metrics_seed_{seed}.json"
            if checkpoint.exists() and report.exists() and not args.force:
                print(f"skip completed region={region} seed={seed}")
                continue
            subprocess.run([sys.executable, "train.py", "--seed", str(seed), "--checkpoint", str(checkpoint),
                            "--holdout-region", region], check=True)
            subprocess.run([sys.executable, "evaluate.py", "--seed", str(seed), "--checkpoint", str(checkpoint),
                            "--output", str(report), "--holdout-region", region], check=True)
    subprocess.run([sys.executable, "-m", "experiments.summarize_ablations", str(root)], check=True)


if __name__ == "__main__":
    main()
