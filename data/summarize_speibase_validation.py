"""Regenerate metrics from already matched SPEIbase points without rereading grids."""

from pathlib import Path

import pandas as pd

from data.compare_speibase_maharashtra import _metrics


def main() -> None:
    folder = Path("results/maharashtra/speibase_validation")
    points = pd.read_csv(folder / "matched_monthly_points.csv", parse_dates=["date"])
    rows = [
        {"site": site, "scale": scale, **_metrics(group, scale)}
        for site, group in points.groupby("site", sort=True)
        for scale in (3, 6, 12)
    ]
    rows.extend(
        {"site": "ALL", "scale": scale, **_metrics(points, scale)}
        for scale in (3, 6, 12)
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(folder / "site_scale_metrics.csv", index=False)
    print(metrics[metrics["site"] == "ALL"].to_string(index=False))


if __name__ == "__main__":
    main()
