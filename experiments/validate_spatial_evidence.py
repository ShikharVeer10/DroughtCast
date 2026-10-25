"""Validate a gridded spatial evidence NetCDF and write paper-safe metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.spatial import load_spatial_grid, spatial_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", help="NetCDF containing obs_* and pred_* fields")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--output", default="results/spatial_metrics.json")
    args = parser.parse_args()
    grid = load_spatial_grid(args.grid)
    metrics = spatial_metrics(grid, args.threshold)
    report = {
        "grid": str(Path(args.grid)),
        "dimensions": {name: int(size) for name, size in grid.dataset.sizes.items()},
        "latitude_coordinate": grid.lat,
        "longitude_coordinate": grid.lon,
        "time_coordinate": grid.time,
        "metrics": metrics,
        "status": "validated observed/predicted pairs on a common grid",
        "warning": "Metrics are not evidence of geographic generalization unless the grid test dates were held out from training.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
