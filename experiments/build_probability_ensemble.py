from __future__ import annotations
import subprocess
import sys
import hashlib
from pathlib import Path

import pandas as pd
import torch


KEYS = ["site", "input_start_month", "input_end_month", "target_month", "scale", "lead", "drought_threshold_spei"]


def aggregate_seed_probabilities(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one seed prediction frame is required")
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby(KEYS, as_index=False).agg(
        calibrated_probability_mean=("calibrated_probability", "mean"),
        calibrated_probability_min=("calibrated_probability", "min"),
        calibrated_probability_max=("calibrated_probability", "max"),
        n_seeds=("calibrated_probability", "count"),
    )
    return summary


def main() -> None:
    checkpoint_dir = Path("results/ablations/quantile_cqr")
    checkpoints = sorted(checkpoint_dir.rglob("model_seed_*.pt"))
    data_path = Path("data/monthly_sites.csv")
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    compatible = []
    seen_seeds = set()
    for checkpoint in checkpoints:
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        seed = int(metadata.get("seed", checkpoint.stem.removeprefix("model_seed_")))
        if metadata.get("artifact_metadata", {}).get("dataset_sha256") == data_hash and seed not in seen_seeds:
            compatible.append(checkpoint)
            seen_seeds.add(seed)
    if len(compatible) < 2:
        print(f"Only {len(compatible)} compatible checkpoint(s) match the current dataset hash {data_hash}.")
        stale_seeds = []
        for checkpoint in checkpoints:
            metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
            seed = int(metadata.get("seed", checkpoint.stem.removeprefix("model_seed_")))
            if seed not in seen_seeds and seed not in stale_seeds:
                stale_seeds.append(seed)
        for seed in stale_seeds:
            destination = checkpoint_dir / "current_dataset" / f"model_seed_{seed}.pt"
            print(f".\\.venv\\Scripts\\python.exe train.py --config results\\ablations\\quantile_cqr\\config.yaml --seed {seed} --checkpoint {destination}")
        raise SystemExit(1)
    run_dir = Path("results/probability_seed_runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for checkpoint in compatible:
        output = run_dir / f"{checkpoint.stem}_probabilities.csv"
        command = [
            sys.executable, "inference.py", "--site", "all",
            "--checkpoint", str(checkpoint), "--output-csv", str(output),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        frame = pd.read_csv(output)
        frame["seed"] = checkpoint.stem.removeprefix("model_seed_")
        frames.append(frame)
    result = aggregate_seed_probabilities(frames)
    result["calibration"] = "validation-only Platt scaling"
    output = Path("results/inference_probabilities_ensemble.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Checkpoints: {len(checkpoints)} ({', '.join(path.stem for path in checkpoints)})")
    print(f"Saved ensemble probabilities to {output}")
    print("Probability ranges by scale/lead:")
    for (scale, lead), group in result.groupby(["scale", "lead"], sort=True):
        spread = group.calibrated_probability_max - group.calibrated_probability_min
        print(f"{scale}/{lead}: mean seed range width={spread.mean():.4f}, min={group.calibrated_probability_min.min():.4f}, max={group.calibrated_probability_max.max():.4f}")


if __name__ == "__main__":
    main()
