from pathlib import Path

import pandas as pd


def test_required_future_months_are_not_present_in_canonical_data():
    frame = pd.read_csv(
        Path("data/maharashtra/monthly_sites.csv"), parse_dates=["date"]
    )
    required = {"2026-11", "2026-12", "2027-01", "2027-02"}
    present = set(frame.loc[frame["date"].dt.strftime("%Y-%m").isin(required), "date"].dt.strftime("%Y-%m"))
    assert present == set()


def test_freeze_requires_all_five_origins():
    origins = ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10")
    target_ends = ("2026-10", "2026-11", "2026-12", "2027-01", "2027-02")
    assert len(origins) == len(target_ends) == 5
    assert target_ends[-1] == "2027-02"
