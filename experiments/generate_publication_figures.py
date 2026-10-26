from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from data.loader import load_site_table
from models.forecaster import MultiScaleForecaster
from utils.artifacts import validate_checkpoint_metadata

SCALES = ("spei_3", "spei_6", "spei_12")
LEADS = ("T+1", "T+2", "T+3", "T+4")
COLORS = {"droughtcast": "#0072B2", "persistence": "#E69F00",
          "ridge": "#CC79A7", "gru": "#AA4499", "tcn": "#332288"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "savefig.dpi": 300, "figure.dpi": 150,
})


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.eps", bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model_predictions(cfg: dict, checkpoint: Path):
    data = load_site_table(
        cfg["data"]["path"], cfg["data"].get("features"), cfg["data"]["seq_len"],
        cfg["data"]["forecast_steps"], cfg["data"].get("metadata_path"),
        [], cfg["data"].get("site_filter"), cfg["model"].get("input_extras"),
        cfg["model"].get("static_context", False),
        cfg["model"].get("heads", cfg["model"].get("head_mode", "direct")) == "residual",
        cfg.get("sites"),
    )
    cfg["data"]["input_dim"] = data["test"].X.shape[-1]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state, cfg)
    model = MultiScaleForecaster(cfg)
    model.load_state_dict(state.get("model", state))
    model.eval()
    with torch.no_grad():
        output, _ = model(data["test"].X, last_spei=data["test"].last_spei,
                          static=data["test"].static)
    pred = np.stack([output[k].numpy() for k in SCALES], axis=1)
    return data, pred


def figure_study_network(cfg: dict, out: Path, elevation: pd.DataFrame | None) -> str:
    sites = cfg["sites"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if elevation is None:
        for site, spec in sites.items():
            ax.scatter(spec["lon"], spec["lat"], s=75, edgecolor="black",
                       label=spec.get("region", "unknown"))
            ax.annotate(site.replace("_", " ").title(), (spec["lon"], spec["lat"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title("DroughtCast study sites and configured regions")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_xlabel("Longitude (degrees)")
        ax.text(0.01, 0.01, "Site network map; no elevation data supplied",
                transform=ax.transAxes, fontsize=8, color="#555555")
    else:
        required = {"site", "elevation_m"}
        if not required.issubset(elevation.columns):
            raise ValueError("--elevation-csv must contain site and elevation_m columns")
        joined = elevation.set_index("site")
        vals = [float(joined.loc[s, "elevation_m"]) for s in sites]
        sc = ax.scatter([v["lon"] for v in sites.values()],
                        [v["lat"] for v in sites.values()], c=vals, s=90,
                        cmap="terrain", edgecolor="black")
        for site, spec in sites.items():
            ax.annotate(site.replace("_", " ").title(), (spec["lon"], spec["lat"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        fig.colorbar(sc, ax=ax, label="Elevation (m)")
        ax.set_title("DroughtCast site elevation map")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_xlabel("Longitude (degrees)")
    ax.grid(alpha=.25)
    name = "figure01_study_sites" if elevation is None else "figure01_elevation_map"
    save(fig, out, name)
    return name


def figure_target_distributions(csv: Path, out: Path) -> str:
    frame = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, scale in zip(axes, SCALES):
        vals = pd.to_numeric(frame[scale], errors="coerce").dropna()
        ax.hist(vals, bins=35, color="#4C78A8", alpha=.85, edgecolor="white")
        ax.axvline(-1.0, color="#D55E00", ls="--", lw=1, label="Moderate drought")
        ax.axvline(-1.5, color="#882255", ls=":", lw=1, label="Severe drought")
        ax.set_title(scale.upper().replace("_", "-"))
        ax.set_xlabel("SPEI")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Monthly site observations")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle("Observed target distributions used by the forecasting task")
    fig.tight_layout()
    save(fig, out, "figure02_target_distributions")
    return "figure02_target_distributions"


def figure_scatter(data: dict, pred: np.ndarray, out: Path) -> str:
    y = data["test"].Y.numpy()
    mask = data["test"].mask.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    for i, (ax, scale) in enumerate(zip(axes, SCALES)):
        valid = mask[:, i, 0] & np.isfinite(pred[:, i, 0])
        yt, yp = y[valid, i, 0], pred[valid, i, 0]
        ax.scatter(yt, yp, s=8, alpha=.32, color=COLORS["droughtcast"],
                   rasterized=True)
        lim = max(3.0, float(np.nanmax(np.abs(np.r_[yt, yp]))))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="1:1")
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        bias = float(np.mean(yp - yt))
        ax.text(.04, .96, f"n={len(yt)}\nRMSE={rmse:.3f}\nBias={bias:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "0.7"})
        ax.set_title(f"{scale.upper().replace('_', '-')} | T+1")
        ax.set_xlabel("Observed SPEI")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("DroughtCast prediction")
    fig.suptitle("Observed versus predicted drought index on the untouched test split")
    fig.tight_layout()
    save(fig, out, "figure03_observed_predicted")
    return "figure03_observed_predicted"


def figure_metric_heatmap(report: Path, out: Path) -> str:
    report_data = json.loads(report.read_text(encoding="utf-8"))
    methods = ["droughtcast", "persistence", "ridge", "gru", "tcn"]
    rows, labels = [], []
    for scale in SCALES:
        for lead in LEADS:
            labels.append(f"{scale.upper().replace('_', '-')}\n{lead}")
            rows.append([report_data["methods"].get(m, {}).get(scale, {}).get(lead, {}).get("rmse", np.nan)
                         for m in methods])
    matrix = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 7.0))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(methods)), [m.title() for m in methods], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Test RMSE across accumulation scales and forecast leads")
    ax.set_xlabel("Method")
    ax.set_ylabel("Target scale and forecast lead")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if np.isfinite(matrix[r, c]):
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center",color="white" if matrix[r, c] > np.nanmedian(matrix) else "black", fontsize=7)
    fig.colorbar(image, ax=ax, label="RMSE (lower is better)")
    fig.tight_layout()
    save(fig, out, "figure04_test_rmse_heatmap")
    return "figure04_test_rmse_heatmap"


def figure_residuals(data: dict, pred: np.ndarray, out: Path) -> str:
    y, mask = data["test"].Y.numpy(), data["test"].mask.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for i, (ax, scale) in enumerate(zip(axes, SCALES)):
        valid = mask[:, i, 0] & np.isfinite(pred[:, i, 0])
        residual = pred[valid, i, 0] - y[valid, i, 0]
        ax.hist(residual, bins=30, color=COLORS["droughtcast"], alpha=.85,edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{scale.upper().replace('_', '-')} | T+1")
        ax.set_xlabel("Prediction error")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Test-window count")
    fig.suptitle("Test residual distributions (descriptive, not a calibration claim)")
    fig.tight_layout()
    save(fig, out, "figure05_residual_distributions")
    return "figure05_residual_distributions"


def figure_drought_map(cfg: dict, data: dict, pred: np.ndarray, out: Path,scale: str = "spei_3", threshold: float = -1.0,target_date: str | None = None) -> str:
    scale_idx = SCALES.index(scale)
    metadata = data["test"].metadata
    ymask = data["test"].mask.numpy()[:, scale_idx, :]
    records = []
    for row_idx, meta in enumerate(metadata):
        for lead_idx, date in enumerate(meta["target_dates"]):
            if ymask[row_idx, lead_idx] and np.isfinite(pred[row_idx, scale_idx, lead_idx]):
                records.append({
                    "site": meta["site"], "date": date,
                    "lead": lead_idx + 1, "prediction": pred[row_idx, scale_idx, lead_idx],
                })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No finite predictions available for drought map")
    if target_date:
        chosen_date = target_date
    else:
        complete_dates = (
            frame.groupby("date")["lead"].nunique()
            .loc[lambda values: values == len(LEADS)]
        )
        chosen_date = str(complete_dates.index.max() if len(complete_dates) else frame["date"].max())
    snapshot = frame[frame["date"] == chosen_date].copy()
    if snapshot.empty:
        raise ValueError(f"No test prediction exists for target date {chosen_date!r}")
    frequency = (frame.assign(event=frame["prediction"] <= threshold)
                 .groupby(["site", "lead"])["event"].mean().rename("frequency")
                 .reset_index())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=False, sharey=False)
    sites = cfg["sites"]
    lon = [spec["lon"] for spec in sites.values()]
    lat = [spec["lat"] for spec in sites.values()]
    pad_x = max((max(lon) - min(lon)) * .12, 5)
    pad_y = max((max(lat) - min(lat)) * .12, 5)
    for lead_idx, ax in enumerate(axes.flat, start=1):
        current = snapshot[snapshot["lead"] == lead_idx].merge(
            frequency[frequency["lead"] == lead_idx], on="site", how="left")
        current["lon"] = current["site"].map({k: v["lon"] for k, v in sites.items()})
        current["lat"] = current["site"].map({k: v["lat"] for k, v in sites.items()})
        drought = current["prediction"] <= threshold
        sizes = 90 + 260 * current["frequency"].fillna(0).to_numpy()
        colors = current["prediction"].to_numpy()
        scatter = ax.scatter(current["lon"], current["lat"], c=colors, s=sizes,
                             cmap="RdYlBu", vmin=-2.5, vmax=2.5,
                             edgecolor=np.where(drought, "#7F0000", "black"),
                             linewidth=np.where(drought, 2.0, .7), zorder=3)
        for _, row in current.iterrows():
            ax.annotate(row["site"].replace("_", " ").title(),
                        (row["lon"], row["lat"]), xytext=(5, 5),
                        textcoords="offset points", fontsize=7)
        ax.set_xlim(min(lon) - pad_x, max(lon) + pad_x)
        ax.set_ylim(min(lat) - pad_y, max(lat) + pad_y)
        ax.set_title(f"T+{lead_idx} | {chosen_date}")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(alpha=.25)
    fig.colorbar(scatter, ax=axes, fraction=.025, pad=.02, label="Predicted SPEI")
    fig.suptitle(
        f"Site-level {scale.upper().replace('_', '-')} drought forecast map\n"
        f"Marker size = empirical test frequency of prediction ≤ {threshold:.1f}; "
        "red edge = drought at selected date",
        y=.99,
    )
    fig.tight_layout()
    name = f"figure06_{scale}_drought_map"
    save(fig, out, name)
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/current_model.pt")
    parser.add_argument("--metrics", default="results/full_eval.json")
    parser.add_argument("--out-dir", default="results/publication_figures")
    parser.add_argument("--elevation-csv", help="Real CSV with site,elevation_m columns")
    parser.add_argument("--map-scale", choices=SCALES, default="spei_3")
    parser.add_argument("--drought-threshold", type=float, default=-1.0)
    parser.add_argument("--map-date", help="Test target date (YYYY-MM-DD); defaults to latest available")
    args = parser.parse_args()
    cfg_path, data_path, checkpoint = map(Path, [args.config, "data/monthly_sites.csv", args.checkpoint])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data, pred = load_model_predictions(cfg, checkpoint)
    elevation = pd.read_csv(args.elevation_csv) if args.elevation_csv else None
    figures = [
        figure_study_network(cfg, out, elevation),
        figure_target_distributions(data_path, out),
        figure_scatter(data, pred, out),
        figure_metric_heatmap(Path(args.metrics), out),
        figure_residuals(data, pred, out),
        figure_drought_map(cfg, data, pred, out, args.map_scale,
                           args.drought_threshold, args.map_date),
    ]
    manifest = {
        "figures": figures,
        "dataset": str(data_path), "dataset_sha256": file_sha256(data_path),
        "checkpoint": str(checkpoint), "metrics": str(args.metrics),
        "elevation": str(args.elevation_csv) if args.elevation_csv else None,
        "map_definition": {
            "scale": args.map_scale, "threshold": args.drought_threshold,
            "date": args.map_date or "latest available test target date",
            "marker_size": "empirical frequency of predicted SPEI at or below threshold",
            "probability_claim": False,
        },
        "omitted_unsupported_evidence": ["learning_curve", "elevation_surface"] if elevation is None else ["learning_curve"],
        "note": "All plotted predictions are generated by the compatible checkpoint on the chronological test split.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
