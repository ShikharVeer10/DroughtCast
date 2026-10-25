"""Publication-grade visualisations from real evaluation artefacts.

Every figure reads files under results/ and fails loudly if they are missing.
No figure input is synthesised with np.random.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from utils.visualisation import plot_hexbin_extreme_calibration, plot_multihorizon_seasonal_attention

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "droughtcast": "#0072B2",
    "persistence": "#E69F00",
    "seasonal_climatology": "#009E73",
    "climatology": "#009E73",
    "ridge": "#CC79A7",
    "hist_gradient_boosting": "#D55E00",
    "xgboost": "#882255",
    "var": "#44AA99",
    "gru": "#AA4499",
    "tcn": "#332288",
}
SCALES = ["spei_3", "spei_6", "spei_12"]
SITE_COORDS = {
    "guntur": (79.74, 15.91),
    "jaipur": (75.78, 26.91),
    "nagpur": (79.08, 21.14),
    "murray_darling": (143.0, -34.0),
    "addis_ababa": (38.74, 9.03),
    "phoenix": (-112.07, 33.44),
    "seville": (-5.98, 37.38),
    "pretoria": (28.22, -25.74),
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required results artefact missing: {path}. "
            "Run training/evaluation (or the interpretability export) before make_figures."
        )
    return path


def load_metrics(path: Path | None = None) -> dict:
    candidates = [
        path,
        Path("results/test_metrics.json"),
        Path("results/test_metrics_seed42.json"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "No evaluation metrics found under results/test_metrics.json or "
        "results/test_metrics_seed42.json. Run evaluate.py first."
    )


def save_fig(fig, out_folder: Path, stem: str):
    out_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_folder / f"{stem}.png", dpi=300)
    fig.savefig(out_folder / f"{stem}.pdf")
    fig.savefig(out_folder / f"{stem}.eps")
    plt.close(fig)


def generate_figure1_architecture(out_dir: Path, config_path: Path | None = None):
    """Figure 1: architecture schematic. Prefer config-driven generation when available."""
    config_path = config_path or Path("configs/base_diagram.yaml")
    if config_path.is_file():
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        m = cfg.get("model", {})
        h1 = m.get("hidden_dim_1", 64)
        h2 = m.get("hidden_dim_2", 32)
        drop = m.get("dropout", 0.2)
        encoder_label = f"LSTM({h1}) -> Drop({drop}) -> LSTM({h2})"
    else:
        encoder_label = "LSTM(64) -> Drop(0.2) -> LSTM(32)"

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    phases = [
        ("Phase 1: Input Tensor\n[B, T=6, D=12]\n12 climate, ocean and\ndrought-state variables",
         0.03, 0.45, 0.14, 0.35, "#E8F0FE", "#1A73E8"),
        (f"Phase 2: Shared Encoder\nPyramidal LSTM\n{encoder_label}",
         0.21, 0.45, 0.16, 0.35, "#E6F4EA", "#137333"),
        ("Phase 3: Temporal Attention\nscore = v^T tanh(W h_t)\nSoftmax over T=6",
         0.41, 0.45, 0.15, 0.35, "#FEF7E0", "#F9AB00"),
        ("Phase 4: Shared Context\nContext vector c in R^32\nSame c for all heads",
         0.60, 0.45, 0.15, 0.35, "#FCE8E6", "#C5221F"),
    ]
    for title, x, y, w, h, bg, border in phases:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                      fc=bg, ec=border, lw=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=9, fontweight="bold", transform=ax.transAxes)

    heads = [
        ("SPEI-3 accumulation\nscale head\nT+1..T+4", 0.80, 0.70),
        ("SPEI-6 accumulation\nscale head\nT+1..T+4", 0.80, 0.45),
        ("SPEI-12 accumulation\nscale head\nT+1..T+4", 0.80, 0.20),
    ]
    for title, x, y in heads:
        rect = patches.FancyBboxPatch((x, y), 0.18, 0.20, boxstyle="round,pad=0.015",
                                      fc="#E8EAED", ec="#5F6368", lw=1.8, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.10, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", transform=ax.transAxes)

    for x0, x1 in ((0.17, 0.21), (0.37, 0.41), (0.56, 0.60)):
        ax.annotate("", xy=(x1, 0.625), xytext=(x0, 0.625), xycoords="axes fraction",
                    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=6))
    for y in (0.80, 0.55, 0.30):
        ax.annotate("", xy=(0.80, y), xytext=(0.75, 0.625), xycoords="axes fraction",
                    arrowprops=dict(facecolor="#1A73E8", shrink=0.05, width=1.5, headwidth=6))

    ax.text(0.5, 0.95, "Figure 1: DroughtCast Multi-Scale SPEI Forecasting Architecture",
            ha="center", va="center", fontsize=12, fontweight="bold", transform=ax.transAxes)
    save_fig(fig, out_dir, "figure1_architecture_schematic")


def _site_lead_means(metrics: dict, method: str, scale: str, lead: str = "T+1"):
    by_site = metrics["methods"][method][scale].get("by_site", {})
    if not by_site:
        raise KeyError(
            f"metrics for {method}/{scale} lack by_site entries required for Figure 2"
        )
    rows = []
    for site, leads in by_site.items():
        if lead not in leads:
            continue
        if site not in SITE_COORDS:
            continue
        lon, lat = SITE_COORDS[site]
        rows.append({"site": site, "lon": lon, "lat": lat, "rmse": leads[lead]["rmse"],
                     "mae": leads[lead]["mae"]})
    if not rows:
        raise RuntimeError(f"No geolocated site metrics for {method}/{scale}/{lead}")
    return pd.DataFrame(rows)


def generate_figure2_spatiotemporal_tracking(out_dir: Path, metrics: dict):
    """Figure 2: site-level RMSE maps from real evaluation metrics (not synthetic)."""
    if "droughtcast" not in metrics.get("methods", {}):
        raise KeyError("Figure 2 requires methods.droughtcast in the evaluation JSON")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True)
    scale_labels = ["SPEI-3", "SPEI-6", "SPEI-12"]
    for i, scale in enumerate(SCALES):
        frame = _site_lead_means(metrics, "droughtcast", scale, "T+1")
        ax0, ax1 = axes[i, 0], axes[i, 1]
        sc0 = ax0.scatter(frame["lon"], frame["lat"], c=frame["rmse"], cmap="viridis",
                          s=120, edgecolors="k", vmin=0.5, vmax=2.0)
        ax0.set_title(f"{scale_labels[i]}: DroughtCast RMSE (T+1)", fontweight="bold")
        ax0.set_ylabel("Latitude (deg)")
        ax0.grid(alpha=0.3)
        fig.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.04, label="RMSE")

        sc1 = ax1.scatter(frame["lon"], frame["lat"], c=frame["mae"], cmap="plasma",
                          s=120, edgecolors="k", vmin=0.3, vmax=1.5)
        ax1.set_title(f"{scale_labels[i]}: DroughtCast MAE (T+1)", fontweight="bold")
        ax1.grid(alpha=0.3)
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="MAE")
        if i == 2:
            ax0.set_xlabel("Longitude (deg)")
            ax1.set_xlabel("Longitude (deg)")
    plt.tight_layout()
    save_fig(fig, out_dir, "figure2_spatiotemporal_tracking_bias")


def generate_figure3_hexbin(out_dir: Path, predictions_path: Path | None = None):
    """Figure 3: hexbin calibration from saved prediction arrays."""
    path = predictions_path or Path("results/predictions_test.npz")
    require_file(path)
    payload = np.load(path)
    required = ("y_true", "y_pred")
    for key in required:
        if key not in payload:
            raise KeyError(f"{path} must contain '{key}' with shape [N, 3, L]")
    y_true, y_pred = payload["y_true"], payload["y_pred"]
    titles = ["SPEI-3", "SPEI-6", "SPEI-12"]
    y_true_dict = {titles[i]: y_true[:, i, :].ravel() for i in range(3)}
    y_pred_dict = {titles[i]: y_pred[:, i, :].ravel() for i in range(3)}
    plot_hexbin_extreme_calibration(y_true_dict, y_pred_dict,
                                    save_path=str(out_dir / "figure3_hexbin_calibration.pdf"))
    plot_hexbin_extreme_calibration(y_true_dict, y_pred_dict,
                                    save_path=str(out_dir / "figure3_hexbin_calibration.png"))


def generate_figure4_taylor_and_decay(out_dir: Path, metrics: dict):
    """Figure 4: Taylor-style summary and lead degradation from real metrics."""
    methods = metrics.get("methods", {})
    if "droughtcast" not in methods:
        raise KeyError("Figure 4 requires methods.droughtcast")

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    taylor_methods = [m for m in ("droughtcast", "gru", "tcn", "xgboost", "ridge", "persistence")
                      if m in methods]
    if len(taylor_methods) < 2:
        raise RuntimeError("Figure 4 needs at least two methods with metrics for the Taylor panel")

    for method in taylor_methods:
        block = methods[method]["spei_3"].get("T+2")
        if not block:
            continue
        # Approximate correlation proxy from NSE when r is unavailable: clip to [0, 1].
        nse = float(block.get("nse", 0.0))
        corr = float(np.clip(np.sqrt(max(nse, 0.0)), 0.0, 0.999))
        rmse = float(block["rmse"])
        # Standard-deviation proxy: use RMSE relative to persistence when available.
        radius = max(0.2, min(1.5, rmse))
        ax1.plot(np.arccos(corr), radius, "o", markersize=9,
                 color=PALETTE.get(method, "#333333"), label=method)
    ax1.plot(0, 1.0, "k*", markersize=12, label="reference")
    ax1.set_thetamin(0)
    ax1.set_thetamax(90)
    ax1.set_title("(a) Multi-Model Polar Taylor Proxy (SPEI-3, T+2)", fontweight="bold", pad=15)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), frameon=True)

    ax2 = fig.add_subplot(1, 2, 2)
    leads = [1, 2, 3, 4]
    for method in ("droughtcast", "xgboost", "ridge", "persistence"):
        if method not in methods:
            continue
        rmse = []
        for lead in leads:
            block = methods[method]["spei_3"].get(f"T+{lead}")
            if block is None:
                break
            rmse.append(float(block["rmse"]))
        if len(rmse) == len(leads):
            ax2.plot(leads, rmse, "o-", color=PALETTE.get(method, "#333333"),
                     lw=2.0, label=method)
    ax2.set_xlabel("Forecast Horizon Lead Time (Months)")
    ax2.set_ylabel("Root Mean Squared Error (RMSE)")
    ax2.set_xticks(leads)
    ax2.set_xticklabels(["T+1", "T+2", "T+3", "T+4"])
    ax2.set_title("(b) Multi-Step Lead Degradation (SPEI-3)", fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=True)
    plt.tight_layout()
    save_fig(fig, out_dir, "figure4_polar_taylor_and_decay")


def generate_figure5_attention(out_dir: Path):
    interp_path = require_file(Path("results/interpretability.json"))
    data = json.loads(interp_path.read_text(encoding="utf-8"))
    if "attn_3x6" not in data or "seasonal_12x6" not in data:
        raise KeyError("results/interpretability.json must contain attn_3x6 and seasonal_12x6")
    attn_3x6 = np.asarray(data["attn_3x6"], dtype=float)
    seasonal_12x6 = np.asarray(data["seasonal_12x6"], dtype=float)
    plot_multihorizon_seasonal_attention(
        attn_3x6, seasonal_12x6, save_path=str(out_dir / "figure5_attention_heatmaps.pdf"))
    plot_multihorizon_seasonal_attention(
        attn_3x6, seasonal_12x6, save_path=str(out_dir / "figure5_attention_heatmaps.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, help="Path to evaluation JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--skip-hexbin", action="store_true",
                        help="Skip Figure 3 when results/predictions_test.npz is unavailable")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.metrics)

    print("Generating Figure 1: Architecture Schematic...")
    generate_figure1_architecture(out_dir)
    print("Generating Figure 2: Spatiotemporal Tracking from metrics...")
    generate_figure2_spatiotemporal_tracking(out_dir, metrics)
    if args.skip_hexbin:
        print("Skipping Figure 3 (--skip-hexbin)")
    else:
        print("Generating Figure 3: Hexbin Joint Calibration...")
        generate_figure3_hexbin(out_dir)
    print("Generating Figure 4: Polar Taylor Diagram & Decay Curves...")
    generate_figure4_taylor_and_decay(out_dir, metrics)
    print("Generating Figure 5: Multi-Horizon Attention Heatmaps...")
    generate_figure5_attention(out_dir)
    print(f"\nPublication figures written to {out_dir}")


if __name__ == "__main__":
    main()
