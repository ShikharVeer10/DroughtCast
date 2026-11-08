from pathlib import Path

import pandas as pd


def test_incomplete_future_data_blocks_evaluation():
    frame = pd.read_csv(
        Path("data/maharashtra/monthly_sites.csv"), parse_dates=["date"]
    )
    required = {"2026-11", "2026-12", "2027-01", "2027-02"}
    present = set(
        frame.loc[
            frame["date"].dt.strftime("%Y-%m").isin(required), "date"
        ].dt.strftime("%Y-%m")
    )
    assert present != required
