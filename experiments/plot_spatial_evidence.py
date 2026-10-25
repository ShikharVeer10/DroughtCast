"""Plot observed, predicted, and error drought fields from a validated grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.spatial import load_spatial_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", help="Validated NetCDF with obs_* and pred_* variables")
    parser.add_argument("--scale", choices=("spei_3", "spei_6", "spei_12"), default="spei_6")
    parser.add_argument("--date", help="ISO target date; defaults to the latest date")
    parser.add_argument("--output-dir", default="results/spatial_figures")
    parser.add_argument("--threshold", type=float, default=-1.0)
    args = parser.parse_args()
    grid = load_spatial_grid(args.grid)
    dates = grid.dataset[grid.time].values
    date = np.datetime64(args.date) if args.date else dates[-1]
    observed = grid.dataset[f"obs_{args.scale}"].sel({grid.time: date})
    predicted = grid.dataset[f"pred_{args.scale}"].sel({grid.time: date})
    error = predicted - observed
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    fields = [(observed, "Observed SPEI"), (predicted, "Predicted SPEI"), (error, "Prediction error")]
    for ax, (field, title) in zip(axes, fields):
        image = ax.pcolormesh(grid.dataset[grid.lon], grid.dataset[grid.lat], field,
                              cmap="RdYlBu" if title != "Prediction error" else "coolwarm",
                              shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(alpha=.2)
        fig.colorbar(image, ax=ax, label="SPEI" if title != "Prediction error" else "Error")
    fig.suptitle(
        f"{args.scale.upper().replace('_', '-')} spatial drought evidence | {str(date)[:10]}\n"
        f"Drought threshold: SPEI ≤ {args.threshold:.1f}; common observed/predicted grid",
    )
    for ext in ("png", "pdf", "eps"):
        fig.savefig(out / f"{args.scale}_{str(date)[:10]}_spatial_evidence.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
