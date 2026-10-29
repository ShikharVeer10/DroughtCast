"""Publication-quality figure generation for continuous spatial drought evidence.

Renders all required conference-ready figures in PNG (300 DPI), vector PDF, and EPS:
1. Elevation map
2. Observed SPEI map
3. Predicted SPEI map
4. Calibrated drought probability map
5. Uncertainty / CQR interval width map
6. Prediction error map
7. Threshold comparison map (-1.0, -1.5, -2.0)
8. Reliability diagrams (empirical frequency vs predicted probability)
9. Regional summary map with administrative boundary overlays
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from data.spatial import SCALES
from utils.probability import reliability_table

try:
    import geopandas as gpd
except ImportError:
    gpd = None


def save_multiformat(fig: plt.Figure, base_path: Path) -> None:
    """Save figure to PNG (300 dpi), PDF, and EPS."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(base_path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def overlay_boundaries(ax, boundaries_gpkg: Path | str | None) -> None:
    """Overlay vector administrative boundaries if geopandas and file available."""
    if gpd is not None and boundaries_gpkg is not None:
        p = Path(boundaries_gpkg)
        if p.is_file():
            try:
                gdf = gpd.read_file(p)
                gdf.boundary.plot(ax=ax, color="black", linewidth=0.7, alpha=0.6)
            except (OSError, ValueError) as exc:
                warnings.warn(f"Skipping boundary overlay {p}: {exc}", stacklevel=2)


def plot_elevation_map(
    ds: xr.Dataset,
    output_dir: Path,
    boundaries: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    elev = ds["elevation"].values
    im = ax.pcolormesh(ds["lon"], ds["lat"], elev, cmap="terrain", shading="auto")
    ax.set_title("Copernicus DEM GLO-30 Elevation (m)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25)
    overlay_boundaries(ax, boundaries)
    fig.colorbar(im, ax=ax, label="Elevation (meters)")
    save_multiformat(fig, output_dir / "figure01_spatial_elevation")


def plot_observed_predicted_error(
    ds: xr.Dataset,
    output_dir: Path,
    scale: str = "spei_6",
    time_idx: int = -1,
    boundaries: Path | None = None,
) -> None:
    t_val = ds["time"].values[time_idx]
    date_str = str(t_val)[:10]

    obs = ds[f"obs_{scale}"].isel(time=time_idx).values
    pred = ds[f"pred_{scale}"].isel(time=time_idx).values
    err = ds[f"error_{scale}"].isel(time=time_idx).values

    # 1. Observed SPEI Map
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.pcolormesh(
        ds["lon"], ds["lat"], obs, cmap="RdYlBu", vmin=-3.0, vmax=3.0, shading="auto"
    )
    ax.set_title(
        f"Observed {scale.upper().replace('_', '-')} ({date_str})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25)
    overlay_boundaries(ax, boundaries)
    fig.colorbar(im, ax=ax, label="Observed SPEI")
    save_multiformat(fig, output_dir / f"figure02_observed_{scale}")

    # 2. Predicted SPEI Map
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.pcolormesh(
        ds["lon"], ds["lat"], pred, cmap="RdYlBu", vmin=-3.0, vmax=3.0, shading="auto"
    )
    ax.set_title(
        f"DroughtCast Predicted {scale.upper().replace('_', '-')} ({date_str})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25)
    overlay_boundaries(ax, boundaries)
    fig.colorbar(im, ax=ax, label="Predicted SPEI")
    save_multiformat(fig, output_dir / f"figure03_predicted_{scale}")

    # 3. Prediction Error Map
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.pcolormesh(
        ds["lon"], ds["lat"], err, cmap="coolwarm", vmin=-2.0, vmax=2.0, shading="auto"
    )
    ax.set_title(
        f"Prediction Error (Predicted − Observed) {scale.upper().replace('_', '-')}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25)
    overlay_boundaries(ax, boundaries)
    fig.colorbar(im, ax=ax, label="SPEI Residual Error")
    save_multiformat(fig, output_dir / f"figure04_prediction_error_{scale}")


def plot_probability_and_uncertainty(
    ds: xr.Dataset,
    output_dir: Path,
    scale: str = "spei_6",
    time_idx: int = -1,
    boundaries: Path | None = None,
) -> None:
    t_val = ds["time"].values[time_idx]
    date_str = str(t_val)[:10]

    prob = ds[f"prob_{scale}"].isel(time=time_idx).values
    cqr_lo = ds[f"cqr_lower_{scale}"].isel(time=time_idx).values
    cqr_hi = ds[f"cqr_upper_{scale}"].isel(time=time_idx).values
    width = cqr_hi - cqr_lo

    # 1. Calibrated drought probability map
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.pcolormesh(
        ds["lon"], ds["lat"], prob, cmap="YlOrRd", vmin=0.0, vmax=1.0, shading="auto"
    )
    ax.set_title(
        f"Calibrated P(SPEI ≤ -1.0) Drought Probability ({date_str})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25)
    overlay_boundaries(ax, boundaries)
    fig.colorbar(im, ax=ax, label="Drought Event Probability")
    save_multiformat(fig, output_dir / f"figure05_calibrated_probability_{scale}")

    # 2. Uncertainty / interval width map
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.pcolormesh(
        ds["lon"], ds["lat"], width, cmap="magma_r", vmin=0.5, vmax=3.0, shading="auto"
    )
    ax.set_title(
        f"Conformal Uncertainty: 80% CQR Interval Width ({date_str})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25)
    overlay_boundaries(ax, boundaries)
    fig.colorbar(im, ax=ax, label="80% CQR Interval Width")
    save_multiformat(fig, output_dir / f"figure06_uncertainty_interval_width_{scale}")


def plot_threshold_comparisons(
    ds: xr.Dataset,
    output_dir: Path,
    scale: str = "spei_6",
    time_idx: int = -1,
    boundaries: Path | None = None,
) -> None:
    pred = ds[f"pred_{scale}"].isel(time=time_idx).values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    thresholds = [
        (-1.0, "Moderate (≤ -1.0)"),
        (-1.5, "Severe (≤ -1.5)"),
        (-2.0, "Extreme (≤ -2.0)"),
    ]

    for ax, (th, title) in zip(axes, thresholds):
        binary = (pred <= th).astype(float)
        ax.pcolormesh(
            ds["lon"],
            ds["lat"],
            binary,
            cmap="OrRd",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
        )
        ax.set_title(f"Drought Regime: {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(alpha=0.25)
        overlay_boundaries(ax, boundaries)

    fig.suptitle(
        f"Continuous Spatial Drought Delineation Across Severity Regimes ({scale.upper()})",
        fontsize=14,
        fontweight="bold",
    )
    save_multiformat(fig, output_dir / f"figure07_threshold_comparison_{scale}")


def plot_spatial_reliability(
    ds: xr.Dataset,
    output_dir: Path,
    scale: str = "spei_6",
) -> None:
    obs = ds[f"obs_{scale}"].values.flatten()
    prob = ds[f"prob_{scale}"].values.flatten()
    valid = np.isfinite(obs) & np.isfinite(prob)
    if not valid.any():
        return

    events = (obs[valid] <= -1.0).astype(int)
    probs = np.clip(prob[valid], 0.0, 1.0)
    rel_rows = reliability_table(probs, events, bins=10)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    if rel_rows:
        x = [r["mean_predicted"] for r in rel_rows]
        y = [r["observed_frequency"] for r in rel_rows]
        ax.plot(
            x,
            y,
            "s-",
            color="#1f77b4",
            label=f"DroughtCast ({scale.upper()})",
            linewidth=2,
            markersize=7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Forecast Probability", fontsize=11)
    ax.set_ylabel("Observed Drought Frequency", fontsize=11)
    ax.set_title(
        f"Spatial Event Reliability Diagram ({scale.upper()})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save_multiformat(fig, output_dir / f"figure08_reliability_diagram_{scale}")


def plot_regional_summary(
    ds: xr.Dataset,
    output_dir: Path,
    scale: str = "spei_6",
    time_idx: int = -1,
    boundaries: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    elev = ds["elevation"].values
    pred = ds[f"pred_{scale}"].isel(time=time_idx).values

    # Elevation with boundaries
    im0 = axes[0].pcolormesh(ds["lon"], ds["lat"], elev, cmap="terrain", shading="auto")
    axes[0].set_title(
        "Regional Topography & Administrative Boundaries",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    overlay_boundaries(axes[0], boundaries)
    fig.colorbar(im0, ax=axes[0], label="Elevation (m)")

    # Drought forecast with boundaries
    im1 = axes[1].pcolormesh(
        ds["lon"], ds["lat"], pred, cmap="RdYlBu", vmin=-3.0, vmax=3.0, shading="auto"
    )
    axes[1].set_title(
        "Continuous Drought Forecast & Administrative Boundaries",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_ylabel("Latitude (°N)")
    overlay_boundaries(axes[1], boundaries)
    fig.colorbar(im1, ax=axes[1], label=f"{scale.upper()} Forecast")

    save_multiformat(fig, output_dir / f"figure09_regional_summary_{scale}")


def generate_all_spatial_figures(
    evidence_nc: Path | str,
    output_dir: Path | str = "results/spatial_pipeline/figures",
    boundaries_gpkg: Path | str | None = "data/raw/spatial/gadm/IND.gpkg",
) -> None:
    ds = xr.open_dataset(evidence_nc)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b_path = (
        Path(boundaries_gpkg)
        if boundaries_gpkg and Path(boundaries_gpkg).is_file()
        else None
    )

    plot_elevation_map(ds, out_dir, boundaries=b_path)
    for scale in SCALES:
        plot_observed_predicted_error(ds, out_dir, scale=scale, boundaries=b_path)
        plot_probability_and_uncertainty(ds, out_dir, scale=scale, boundaries=b_path)
        plot_threshold_comparisons(ds, out_dir, scale=scale, boundaries=b_path)
        plot_spatial_reliability(ds, out_dir, scale=scale)
        plot_regional_summary(ds, out_dir, scale=scale, boundaries=b_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready spatial figures"
    )
    parser.add_argument("evidence_nc", help="Spatial evidence NetCDF file")
    parser.add_argument("--output-dir", default="results/spatial_pipeline/figures")
    parser.add_argument("--boundaries", default="data/raw/spatial/gadm/IND.gpkg")
    args = parser.parse_args()

    generate_all_spatial_figures(args.evidence_nc, args.output_dir, args.boundaries)
    print(f"Generated all spatial figures in {args.output_dir} (PNG, PDF, EPS)")


if __name__ == "__main__":
    main()
