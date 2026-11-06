import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.config_schema import validate_config

BASE_FEATURES = [
    "PRECTOTCORR",
    "T2M",
    "T2MDEW",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "oni",
    "dmi",
    "nao",
    "pdo",
    "spei_3_obs",
    "spei_6_obs",
    "spei_12_obs",
]
VARIANTS = {
    "full_model": {},
    "no_attention": {"model": {"attention": "none"}},
    "head_specific_attention": {"model": {"attention": "head_specific"}},
    "separate_encoders": {"model": {"separate_encoders": True}},
    "gru": {"model": {"encoder": "gru"}},
    "transformer": {"model": {"encoder": "transformer"}},
    "variable_gate": {"model": {"variable_gate": True}},
    "fixed_task_weights": {"training": {"task_weighting": "fixed"}},
    "uncertainty_task_weights": {"training": {"task_weighting": "uncertainty"}},
    "no_extreme_weighting": {"training": {"extreme_weight": 1.0}},
    "extreme_weight_2": {"training": {"extreme_weight": 2.0}},
    "huber_point_loss": {"training": {"loss": "huber"}},
    "quantile_cqr": {"model": {"quantiles": [0.1, 0.5, 0.9]}},
    "window_3": {"data": {"seq_len": 3}},
    "window_12": {"data": {"seq_len": 12}},
    "window_24": {"data": {"seq_len": 24}},
    "without_indices": {"data": {"features": BASE_FEATURES[:5] + BASE_FEATURES[9:]}},
    "without_observed_spei": {"data": {"features": BASE_FEATURES[:9]}},
    "climate_only": {"data": {"features": BASE_FEATURES[:5]}},
    "without_climate": {"data": {"features": BASE_FEATURES[5:]}},
    "single_site_aurangabad": {"data": {"site_filter": ["aurangabad"]}},
}


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load_base_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Keep dataset-only project metadata out of the strict training schema.
    config.pop("study_area", None)
    if "data" not in config or "path" not in config["data"]:
        canonical_path = (
            Path(__file__).resolve().parents[1] / "configs" / "base_diagram.yaml"
        )
        canonical = yaml.safe_load(canonical_path.read_text(encoding="utf-8")) or {}
        config = merge(canonical, config)
    if isinstance(config.get("sites"), list):
        config["sites"] = {
            site.get("id", site.get("name", "").lower().replace(" ", "_")): site
            for site in config["sites"]
        }
    model = config.setdefault("model", {})
    if model.get("attention") == "multi_horizon":
        model["attention"] = "shared"
    training = config.setdefault("training", {})
    if training.get("loss") == "uncertainty_weighted":
        training["loss"] = "mse"
        training["task_weighting"] = "uncertainty"
    validate_config(config)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Base training configuration",
    )
    parser.add_argument(
        "--output-dir",
        default="results/maharashtra/ablations",
        help="Keep Maharashtra ablation artifacts separate from global benchmark runs",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS)
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if both checkpoint and report already exist",
    )
    parser.add_argument(
        "--probability-calibration",
        action="store_true",
        help="After the quantile_cqr variant, fit validation-only event calibration for each seed",
    )
    args = parser.parse_args()
    if args.probability_calibration and "quantile_cqr" not in args.variants:
        parser.error("--probability-calibration requires --variants quantile_cqr")
    base = load_base_config(Path(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seeds": args.seeds,
        "variants": args.variants,
        "dataset_path": base["data"]["path"],
        "primary_architecture": "shared LSTM attention; ablations only",
        "baseline_variant": "full_model",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for variant in args.variants:
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        config = merge(copy.deepcopy(base), VARIANTS[variant])
        config_path = variant_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        for seed in args.seeds:
            checkpoint = variant_dir / f"model_seed_{seed}.pt"
            report = variant_dir / f"metrics_seed_{seed}.json"
            if report.exists() and checkpoint.exists() and not args.force:
                print(f"skip completed {variant} seed={seed}")
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "train.py"),
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--checkpoint",
                    str(checkpoint),
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "evaluate.py"),
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(report),
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.summarize_ablations",
            str(out_dir.resolve()),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if args.probability_calibration:
        calibration_dir = out_dir.parent / "probability_calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.evaluate_event_calibration",
                    "--config",
                    str(out_dir / "quantile_cqr" / "config.yaml"),
                    "--checkpoint",
                    str(out_dir / "quantile_cqr" / f"model_seed_{seed}.pt"),
                    "--output",
                    str((calibration_dir / f"calibration_seed_{seed}.json").resolve()),
                    "--predictions-output",
                    str((calibration_dir / f"predictions_seed_{seed}.csv").resolve()),
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.summarize_probability_calibration",
                "--input-dir",
                str(calibration_dir.resolve()),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
