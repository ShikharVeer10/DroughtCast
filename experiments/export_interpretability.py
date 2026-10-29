"""Export deterministic attention and perturbation diagnostics on validation windows only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.loader import load_site_table
from models.forecaster import MultiScaleForecaster
from utils.artifacts import validate_checkpoint_metadata

SCALES = ("spei_3", "spei_6", "spei_12")
INDICES = {"oni", "dmi", "nao", "pdo"}


def _season(meta, sites):
    month = int(meta["target_dates"][0][5:7])
    north = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    south = {"DJF": (6, 7, 8), "MAM": (9, 10, 11), "JJA": (12, 1, 2), "SON": (3, 4, 5)}
    hemi = "north" if float(sites.get(meta["site"], {}).get("lat", 0)) >= 0 else "south"
    seasons = north if hemi == "north" else south
    return next(name for name, months in seasons.items() if month in months)


def _predict(model, x, last_spei, static):
    with torch.no_grad():
        predictions, aux = model(x, last_spei=last_spei, static=static)
    return {
        scale: value.squeeze(-1).cpu().numpy()
        if value.ndim == 3
        else value.cpu().numpy()
        for scale, value in predictions.items()
    }, aux["attention"].cpu().numpy()


def _mse(prediction, targets, mask, samples=None):
    values = {}
    for scale_index, scale in enumerate(SCALES):
        for lead_index in range(targets.shape[-1]):
            valid = mask[:, scale_index, lead_index].copy()
            if samples is not None:
                valid &= samples
            valid &= np.isfinite(prediction[scale][:, lead_index])
            values[(scale, lead_index + 1)] = (
                float(
                    np.mean(
                        (
                            prediction[scale][valid, lead_index]
                            - targets[valid, scale_index, lead_index]
                        )
                        ** 2
                    )
                )
                if valid.any()
                else float("nan")
            )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_diagram.yaml")
    parser.add_argument("--checkpoint", default="results/seeds/model_seed_42.pt")
    parser.add_argument("--output", default="results/interpretability_validation.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    partitions = load_site_table(
        cfg["data"]["path"],
        cfg["data"].get("features"),
        cfg["data"]["seq_len"],
        cfg["data"]["forecast_steps"],
        cfg["data"].get("metadata_path", "data/meta.json"),
        input_extras=cfg["model"].get("input_extras"),
        static_context=cfg["model"].get("static_context", False),
        residual_heads=cfg["model"].get("heads", "direct") == "residual",
        site_specs=cfg.get("sites"),
    )
    val = partitions["val"]
    if len(val) < 2:
        raise RuntimeError("Validation partition needs at least two windows")
    cfg["data"]["input_dim"] = val.X.shape[-1]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state, cfg)
    model = MultiScaleForecaster(cfg)
    model.load_state_dict(state.get("model", state))
    model.eval()
    seed = int(state.get("seed", 42))
    feature_names = list(cfg["data"]["features"])
    targets, mask = val.Y.numpy(), val.mask.numpy()
    seasons = np.asarray([_season(meta, cfg.get("sites", {})) for meta in val.metadata])
    base, attention = _predict(model, val.X, val.last_spei, val.static)
    base_mse = _mse(base, targets, mask)
    if attention.ndim == 4:
        attention_scale = {
            scale: attention[:, idx, :, 0] for idx, scale in enumerate(SCALES)
        }
    elif attention.ndim == 3:
        shared = attention[:, :, 0]
        attention_scale = {scale: shared for scale in SCALES}
    else:
        raise ValueError(f"Unexpected attention shape {attention.shape}")

    lag_count, sample_count, _ = (
        val.X.shape[1],
        val.X.shape[0],
        val.X.shape[2],
    )
    occluded_predictions = {}
    for lag in range(lag_count):
        changed = val.X.clone()
        changed[:, lag, :] = val.X[:, lag, :].mean(dim=0, keepdim=True)
        occluded_predictions[lag], _ = _predict(
            model, changed, val.last_spei, val.static
        )

    attention_rows = []
    for season in ("DJF", "MAM", "JJA", "SON"):
        sample_mask = seasons == season
        if not sample_mask.any():
            continue
        baseline = _mse(base, targets, mask, sample_mask)
        for lag in range(lag_count):
            changed_mse = _mse(occluded_predictions[lag], targets, mask, sample_mask)
            for scale in SCALES:
                weights = attention_scale[scale][sample_mask].mean(axis=0)
                for lead in range(targets.shape[-1]):
                    attention_rows.append(
                        {
                            "seed": seed,
                            "split": "validation",
                            "scale": scale,
                            "lead": f"T+{lead + 1}",
                            "season": season,
                            "input_step": f"t-{lag_count - lag}",
                            "attention": float(weights[lag]),
                            "occlusion_delta_validation_mse": changed_mse[
                                (scale, lead + 1)
                            ]
                            - baseline[(scale, lead + 1)],
                            "n_validation_windows": int(sample_mask.sum()),
                        }
                    )

    # Rotate sample indices by a fixed half-period. This permutation is deterministic and uses validation data only.
    order = torch.roll(torch.arange(sample_count), shifts=max(1, sample_count // 2))
    feature_rows = []
    for feature_index, feature in enumerate(feature_names):
        lowered = feature.lower()
        group = (
            "SPEI state"
            if lowered.startswith("spei_")
            else ("ocean-teleconnection" if lowered in INDICES else "local climate")
        )
        for lag in range(lag_count):
            changed = val.X.clone()
            changed[:, lag, feature_index] = val.X[order, lag, feature_index]
            changed_prediction, _ = _predict(model, changed, val.last_spei, val.static)
            permuted_mse = _mse(changed_prediction, targets, mask)
            for scale in SCALES:
                for lead in range(targets.shape[-1]):
                    feature_rows.append(
                        {
                            "seed": seed,
                            "split": "validation",
                            "feature": feature,
                            "feature_group": group,
                            "input_step": f"t-{lag_count - lag}",
                            "scale": scale,
                            "lead": f"T+{lead + 1}",
                            "delta_validation_mse": permuted_mse[(scale, lead + 1)]
                            - base_mse[(scale, lead + 1)],
                            "n_validation_windows": sample_count,
                        }
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    attention_path = output.parent / "attention_occlusion_validation.csv"
    feature_path = output.parent / "feature_importance_validation.csv"
    pd.DataFrame(attention_rows).to_csv(attention_path, index=False)
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    output.write_text(
        json.dumps(
            {
                "split": "validation only",
                "seed": seed,
                "checkpoint": args.checkpoint,
                "checkpoint_sha256": hashlib.sha256(
                    Path(args.checkpoint).read_bytes()
                ).hexdigest(),
                "dataset_sha256": hashlib.sha256(
                    Path(cfg["data"]["path"]).read_bytes()
                ).hexdigest(),
                "config_sha256": hashlib.sha256(
                    Path(args.config).read_bytes()
                ).hexdigest(),
                "config": args.config,
                "attention_csv": str(attention_path),
                "feature_importance_csv": str(feature_path),
                "season_definition": "meteorological seasons assigned by each site's hemisphere",
                "occlusion": "replace all feature values at one input lag with that lag's validation mean",
                "permutation": "fixed cyclic sample permutation for one variable and lag; deterministic",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        pd.DataFrame(attention_rows)
        .groupby(["scale", "season"], as_index=False)
        .attention.mean()
        .to_string(index=False)
    )
    print(
        pd.DataFrame(feature_rows)
        .groupby(["feature_group", "feature"], as_index=False)
        .delta_validation_mse.mean()
        .to_string(index=False)
    )
    print(
        f"Exported validation-only diagnostics to {attention_path} and {feature_path}"
    )


if __name__ == "__main__":
    main()
