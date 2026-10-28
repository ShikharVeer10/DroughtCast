"""Read-only input and observed-SPEI diagnostics from monthly_sites.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    frame = pd.read_csv("data/monthly_sites.csv", parse_dates=["date"])
    spei_columns = [f"spei_{scale}_obs" for scale in (3, 6, 12)]
    missing = [column for column in spei_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing observed SPEI columns: {missing}")
    print(f"Observed SPEI columns: {spei_columns}")
    selected = frame.loc[frame.date.eq(pd.Timestamp("2024-12-01")), ["site", *spei_columns]]
    print("Per-site observed SPEI on 2024-12-01:")
    print(selected.sort_values("site").to_string(index=False))

    cfg = yaml.safe_load(Path("results/ablations/quantile_cqr/config.yaml").read_text(encoding="utf-8"))
    features = list(cfg["data"]["features"])
    missing_features = [feature for feature in features if feature not in frame.columns]
    if missing_features:
        raise ValueError(f"Model input features absent from dataset: {missing_features}")
    recent = {}
    for site in ("phoenix", "pretoria"):
        recent[site] = (frame.loc[frame.site.eq(site)].sort_values("date").tail(6)
                        .set_index("date")[features])
    common_dates = recent["phoenix"].index.intersection(recent["pretoria"].index)
    differences = (recent["phoenix"].loc[common_dates] - recent["pretoria"].loc[common_dates]).abs()
    print("Absolute feature differences for the last six Phoenix/Pretoria input rows:")
    print(differences.to_string())

    print("Per-site observed SPEI counts:")
    for site, group in frame.groupby("site", sort=True):
        metrics = []
        for column in spei_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            mean, std = values.mean(), values.std(ddof=0)
            z_count = int((((values - mean) / std).abs() > 3).sum()) if std else 0
            baseline = group.loc[group.date.between("1981-01-01", "2010-12-31"), column]
            later = group.loc[group.date >= "2017-01-01", column]
            lo, hi = baseline.min(), baseline.max()
            outside_count = int(((later < lo) | (later > hi)).sum())
            metrics.append(f"{column}: |z|>3={z_count}, 2017+ outside 1981-2010 range={outside_count}")
        print(f"{site}: " + "; ".join(metrics))


if __name__ == "__main__":
    main()
