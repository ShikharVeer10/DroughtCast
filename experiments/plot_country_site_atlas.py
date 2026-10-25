"""Plot real country boundaries with DroughtCast site-level risk overlays.

This is a geographic atlas, not a continuous spatial prediction surface.
Continuous drought color fields require gridded predictors and observed
targets and are intentionally rejected by this command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import yaml


COUNTRY_FILES = {
    "india": "IND.gpkg", "australia": "AUS.gpkg", "ethiopia": "ETH.gpkg",
    "usa": "USA.gpkg", "spain": "ESP.gpkg", "south_africa": "ZAF.gpkg",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries", default="data/raw/spatial/gadm")
    parser.add_argument("--risk-report", default="results/probabilistic_figures/probability_map_report.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="results/spatial_figures")
    args = parser.parse_args()

    risk = json.loads(Path(args.risk_report).read_text(encoding="utf-8"))
    if risk.get("probability_calibration_claim") is not False:
        raise ValueError("Risk report must explicitly distinguish quantile-derived risk from calibrated probability")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    snapshot_date = risk["date"]
    # The site-level report does not contain coordinates; use the real config.
    site_rows = pd.DataFrame([
        {"site": name, "lon": spec["lon"], "lat": spec["lat"]}
        for name, spec in cfg["sites"].items()
    ])
    fig, ax = plt.subplots(figsize=(11, 7))
    boundary_dir = Path(args.boundaries)
    loaded = 0
    for country, filename in COUNTRY_FILES.items():
        path = boundary_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing boundary layer for {country}: {path}")
        layer = gpd.read_file(path, layer=0)
        layer.boundary.plot(ax=ax, linewidth=.25, color="#555555")
        loaded += 1
    ax.scatter(site_rows["lon"], site_rows["lat"], s=75, color="#D55E00",
               edgecolor="black", zorder=3)
    for _, row in site_rows.iterrows():
        ax.annotate(row["site"].replace("_", " ").title(),
                    (row["lon"], row["lat"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(
        f"DroughtCast multi-country site atlas | target date {snapshot_date}\n"
        "Country boundaries and forecast sites; no continuous drought raster claimed"
    )
    ax.grid(alpha=.25)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(out / f"country_site_atlas_{snapshot_date}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"countries_loaded": loaded, "sites": len(site_rows),
                      "date": snapshot_date, "continuous_surface": False}, indent=2))


if __name__ == "__main__":
    main()
