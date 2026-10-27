from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.loader import load_site_table, validate_dataset_protocol
from models.forecaster import MultiScaleForecaster
from utils.artifacts import architecture_fingerprint, validate_checkpoint_metadata
from utils.probability import fit_isotonic_calibrator, quantile_risk_below_threshold

SCALES = ("spei_3", "spei_6", "spei_12")


class DroughtCastPredictor:
    def __init__(self,checkpoint_path: str | Path,config_path: str | Path | None = None,device: str = "cpu",):
        self.device = torch.device(device)
        chk = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_cfg = None
        if isinstance(chk, dict) and "config" in chk and chk["config"]:
            checkpoint_cfg = chk["config"]
        elif config_path is not None:
            checkpoint_cfg = yaml.safe_load(
                Path(config_path).read_text(encoding="utf-8")
            )
        else:
            raise ValueError("Configuration must be provided or present in the checkpoint.")
        if isinstance(chk, dict):
            validate_checkpoint_metadata(chk, checkpoint_cfg)
        if config_path is not None and isinstance(chk, dict) and chk.get("config"):
            supplied_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            supplied_cfg.setdefault("data", {})
            if not supplied_cfg["data"].get("input_dim") and supplied_cfg["data"].get(
                "features"
            ):
                supplied_cfg["data"]["input_dim"] = len(
                    supplied_cfg["data"]["features"]
                )
            checkpoint_arch = chk["artifact_metadata"]["architecture_fingerprint"]
            if architecture_fingerprint(supplied_cfg) != checkpoint_arch:
                raise ValueError(
                    "The supplied config does not match the checkpoint architecture. "
                    "Use the config saved in the checkpoint or retrain."
                )
        self.cfg = checkpoint_cfg
        self.model = MultiScaleForecaster(self.cfg)
        state_dict = chk.get("model", chk)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, x: np.ndarray | torch.Tensor
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if isinstance(x, np.ndarray):
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_t = x.to(device=self.device, dtype=torch.float32)

        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)

        with torch.no_grad():
            preds_t, aux_t = self.model(x_t)

        preds = {k: v.cpu().numpy() for k, v in preds_t.items()}

        def to_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy()
            if isinstance(value, list):
                return [to_numpy(item) for item in value]
            if isinstance(value, tuple):
                return tuple(to_numpy(item) for item in value)
            return value

        aux = {k: to_numpy(v) for k, v in aux_t.items()}
        return preds, aux


def _load_recent_input(
    input_csv: Path, site: str, features: list[str], seq_len: int, metadata_path: Path
) -> tuple[np.ndarray, list[pd.Timestamp], dict]:
    """Read six real monthly observations and apply the saved train-only scaler."""
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input data not found: {input_csv}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Saved training normalizers not found: {metadata_path}. Rebuild the dataset and rerun training."
        )
    frame = pd.read_csv(input_csv, parse_dates=["date"])
    required = {"date", *features}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    if "site" in frame.columns:
        frame = frame.loc[frame["site"].astype(str) == site].copy()
    if frame.empty:
        raise ValueError(f"No input rows found for site '{site}' in {input_csv}")
    frame = frame.sort_values("date")
    if frame["date"].duplicated().any():
        raise ValueError(f"Input CSV has duplicate monthly dates for site '{site}'")
    recent = frame.tail(seq_len).copy()
    if len(recent) != seq_len:
        raise ValueError(
            f"Site '{site}' needs {seq_len} real monthly rows; found {len(recent)}"
        )
    dates = list(pd.to_datetime(recent["date"]))
    expected = list(pd.date_range(end=dates[-1], periods=seq_len, freq="MS"))
    if dates != expected:
        raise ValueError(
            f"Site '{site}' input rows must be consecutive month starts: {dates}"
        )
    raw = recent[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError(
            "The latest six input rows contain missing or nonnumeric feature values"
        )
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    normalizer = saved.get("normalizers", {}).get(site)
    if not normalizer:
        raise ValueError(
            f"No train-only normalizer for site '{site}' in {metadata_path}"
        )
    mean = np.asarray(normalizer.get("mean"), dtype=float)
    std = np.asarray(normalizer.get("std"), dtype=float)
    if (
        mean.shape != (len(features),)
        or std.shape != (len(features),)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError(
            f"Invalid saved normalizer for site '{site}' in {metadata_path}"
        )
    return ((raw - mean) / std).astype(np.float32), dates, saved


def main():
    parser = argparse.ArgumentParser(description="Print validation-calibrated drought-event probabilities from DroughtCast."
    )
    parser.add_argument(
        "--checkpoint",
        default="results/ablations/quantile_cqr/model_seed_42.pt",
        help="Dataset-matched quantile-model checkpoint",
    )
    parser.add_argument(
        "--config",
        help="Optional config; by default use the config saved in the checkpoint",
    )
    parser.add_argument(
        "--input-csv",
        help="CSV with date, site (optional), and configured raw feature columns; defaults to the configured monthly dataset",
    )
    parser.add_argument(
        "--site",
        required=True,
        help="Configured site name or 'all' for every configured site",
    )
    parser.add_argument(
        "--output-csv",
        default="results/inference_probabilities_seed_42.csv",
        help="Write the exact forecasts shown in the terminal to this CSV",
    )
    parser.add_argument(
        "--drought-threshold",
        "--threshold",
        dest="drought_threshold",
        type=float,
        default=-1.0,
        help="SPEI event threshold whose probability is calibrated on validation data (default: -1.0)",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run inference on (cpu, cuda)"
    )
    args = parser.parse_args()

    chk_path = Path(args.checkpoint)
    if not chk_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {chk_path}. Train a model with `python train.py --config configs/base_diagram.yaml`."
        )
    predictor = DroughtCastPredictor(chk_path, config_path=args.config, device=args.device)
    cfg = predictor.cfg
    features = list(cfg["data"]["features"])
    seq_len = int(cfg["data"].get("seq_len", 6))
    input_csv = Path(args.input_csv or cfg["data"]["path"])
    metadata_path = Path(cfg["data"].get("metadata_path", "data/meta.json"))

    quantiles = list(cfg.get("model", {}).get("quantiles") or [])
    if quantiles != [0.1, 0.5, 0.9]:
        raise ValueError(
            "Calibrated event probabilities require the q10/q50/q90 checkpoint. "
            "Use results/ablations/quantile_cqr/model_seed_42.pt."
        )
    training_data_path = Path(cfg["data"]["path"])
    validate_dataset_protocol()
    checkpoint = torch.load(chk_path, map_location="cpu", weights_only=False)
    checkpoint_meta = checkpoint.get("artifact_metadata", {})
    training_hash = hashlib.sha256(training_data_path.read_bytes()).hexdigest()
    if checkpoint_meta.get("dataset_sha256") != training_hash:
        raise ValueError("Quantile checkpoint dataset hash does not match the current training dataset")
    if (
        input_csv.resolve() == training_data_path.resolve()
        and hashlib.sha256(input_csv.read_bytes()).hexdigest() != training_hash
    ):
        raise ValueError("Input dataset changed after checkpoint training; use a separate --input-csv for new observations")

    calibration_data = load_site_table(
        training_data_path,
        features,
        seq_len,
        cfg["data"]["forecast_steps"],
        metadata_path,
        [],
        cfg["data"].get("site_filter"),
        cfg["model"].get("input_extras"),
        cfg["model"].get("static_context", False),
        False,
        cfg.get("sites"),
    )
    validation = calibration_data["val"]
    if not len(validation):
        raise RuntimeError("No validation samples are available to fit probability calibration")
    validation_predictions, _ = predictor.predict(validation.X)
    configured_sites = cfg.get("sites", {})
    if not isinstance(configured_sites, dict) or not configured_sites:
        raise ValueError(
            "The checkpoint configuration has no named sites under 'sites'"
        )
    selected_sites = (
        list(configured_sites) if args.site.lower() == "all" else [args.site]
    )
    unknown = [site for site in selected_sites if site not in configured_sites]
    if unknown:
        raise ValueError(
            f"Unknown site(s) {unknown}. Configured sites: {', '.join(configured_sites)}"
        )
    site_inputs = {}
    for site in selected_sites:
        site_inputs[site] = _load_recent_input(
            input_csv, site, features, seq_len, metadata_path
        )
    predictions, _ = predictor.predict(
        np.stack([site_inputs[site][0] for site in selected_sites])
    )
    calibrators = {}
    for scale_index, scale in enumerate(SCALES):
        for lead in range(int(cfg["data"]["forecast_steps"])):
            valid = validation.mask[:, scale_index, lead].numpy().copy()
            valid &= np.isfinite(validation_predictions[scale][..., lead, :]).all(
                axis=-1
            )
            val_quantiles = validation_predictions[scale][valid, lead, :]
            val_events = (
                validation.Y[:, scale_index, lead].numpy()[valid]
                <= args.drought_threshold
            ).astype(int)
            raw_val_probability = quantile_risk_below_threshold(
                val_quantiles, args.drought_threshold
            )
            calibrators[(scale_index, lead)] = (
                fit_isotonic_calibrator(raw_val_probability, val_events),
                int(valid.sum()),
            )

    print("DroughtCast calibrated drought-event forecast")
    print(f"Sites: {', '.join(selected_sites)}")
    print(f"Checkpoint: {chk_path}")
    print(f"Input data: {input_csv}")
    print(f"Event: SPEI <= {args.drought_threshold:.1f}")
    print("Calibration: validation-only isotonic regression, fitted separately by scale and lead")
    rows = []
    for site_index, site in enumerate(selected_sites):
        _, input_dates, _ = site_inputs[site]
        last_input = input_dates[-1]
        print(
            f"\nSite: {site} | input: {input_dates[0]:%Y-%m} through {last_input:%Y-%m}"
        )
        print("Scale   Lead  Target month  q10     q50     q90     P(drought)  n_val")
        print("------  ----  ------------  ------  ------  ------  -----------  -----")
        for scale_index, scale in enumerate(SCALES):
            for lead in range(int(cfg["data"]["forecast_steps"])):
                calibrator, n_val = calibrators[(scale_index, lead)]
                forecast_quantiles = predictions[scale][site_index, lead, :]
                raw_probability = float(
                    quantile_risk_below_threshold(
                        forecast_quantiles, args.drought_threshold
                    )
                )
                calibrated_probability = float(
                    calibrator.predict(np.asarray([raw_probability]))[0]
                )
                target_month = last_input + pd.DateOffset(months=lead + 1)
                q10, q50, q90 = (float(v) for v in forecast_quantiles)
                print(
                    f"{scale.upper():<7} T+{lead + 1:<2}  {target_month:%Y-%m}       {q10:>5.2f}   {q50:>5.2f}   {q90:>5.2f}       {calibrated_probability:>6.4f}     {n_val:>3}"
                )
                rows.append(
                    {
                        "site": site,
                        "input_start_month": input_dates[0].strftime("%Y-%m"),
                        "input_end_month": last_input.strftime("%Y-%m"),
                        "target_month": target_month.strftime("%Y-%m"),
                        "scale": scale.upper(),
                        "lead": f"T+{lead + 1}",
                        "drought_threshold_spei": args.drought_threshold,
                        "q10": q10,
                        "q50": q50,
                        "q90": q90,
                        "raw_probability": raw_probability,
                        "calibrated_probability": calibrated_probability,
                        "calibration": "validation-only isotonic",
                        "n_validation": n_val,
                        "checkpoint": str(chk_path),
                        "dataset_sha256": checkpoint_meta.get("dataset_sha256", ""),
                    }
                )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} forecasts to {output_path}")
    print("Probabilities are calibrated from validation data for this checkpoint and threshold; they are not guarantees of future occurrence.")


if __name__ == "__main__":
    main()
