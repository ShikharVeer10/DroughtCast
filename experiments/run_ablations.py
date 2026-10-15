"""Run architecture-near ablations using the same frozen chronological protocol.

Examples:
  python -m experiments.run_ablations --seeds 11 22 33 44 55 --variants no_attention gru
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


BASE_FEATURES = ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN",
                 "oni", "dmi", "nao", "pdo", "lag_spei_3", "lag_spei_6", "lag_spei_12"]
VARIANTS = {
    "no_attention": {"model": {"attention": "none"}},
    "head_specific_attention": {"model": {"attention": "head_specific"}},
    "gru": {"model": {"encoder": "gru"}},
    "transformer": {"model": {"encoder": "transformer"}},
    "variable_gate": {"model": {"variable_gate": True}},
    "window_3": {"data": {"seq_len": 3}},
    "window_12": {"data": {"seq_len": 12}},
    "window_24": {"data": {"seq_len": 24}},
    "without_indices": {"data": {"features": BASE_FEATURES[:5] + BASE_FEATURES[9:]}},
    "without_lagged_spei": {"data": {"features": BASE_FEATURES[:9]}},
    "climate_only": {"data": {"features": BASE_FEATURES[:5]}},
}


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["no_attention", "head_specific_attention", "gru", "transformer", "without_indices", "without_lagged_spei", "window_3", "window_12", "window_24"])
    args = parser.parse_args()
    base = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    out_dir = Path("results/ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": args.seeds, "variants": args.variants, "primary_architecture": "shared LSTM attention; ablations only"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for variant in args.variants:
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        config = merge(copy.deepcopy(base), VARIANTS[variant])
        config_path = variant_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for seed in args.seeds:
            checkpoint = variant_dir / f"model_seed_{seed}.pt"
            report = variant_dir / f"metrics_seed_{seed}.json"
            subprocess.run([sys.executable, "train.py", "--config", str(config_path), "--seed", str(seed),
                            "--checkpoint", str(checkpoint)], check=True)
            subprocess.run([sys.executable, "evaluate.py", "--config", str(config_path), "--seed", str(seed),
                            "--checkpoint", str(checkpoint), "--output", str(report)], check=True)
    subprocess.run([sys.executable, "-m", "experiments.summarize_ablations", str(out_dir)], check=True)
