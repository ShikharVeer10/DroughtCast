from pathlib import Path

import pandas as pd


def test_november_to_february_forecast_origin_has_incomplete_features():
    frame = pd.read_csv(
        Path("data/maharashtra/monthly_sites.csv"), parse_dates=["date"]
    )
    window = frame[
        frame["date"].dt.strftime("%Y-%m").isin(
            ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
        )
    ]
    assert window[["dmi", "pdo"]].isna().any().any()
