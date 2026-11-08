"""Print frozen Phase 3.14 forecasts in a lead-ordered terminal table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="droughtcast_neural_selected_seed_101")
    args = parser.parse_args()
    path = Path("results/maharashtra/phase3/phase3_14_prospective_forecasts/site_level_forecasts.csv")
    data = pd.read_csv(path)
    data = data[data["model"] == args.model].copy()
    if data.empty:
        raise SystemExit(f"Model not found: {args.model}")
    data = data.sort_values(["site", "scale", "target_month"])
    data["lead"] = data.groupby(["site", "scale"]).cumcount().map(
        {0: "T+1", 1: "T+2", 2: "T+3", 3: "T+4"}
    )
    data["target_month"] = pd.to_datetime(data["target_month"]).dt.strftime("%b %Y")
    table = data.pivot(index=["site", "scale"], columns="lead", values="prediction")
    table = table.reindex(columns=["T+1", "T+2", "T+3", "T+4"])
    months = (
        data.drop_duplicates("lead")
        .set_index("lead")["target_month"]
        .reindex(["T+1", "T+2", "T+3", "T+4"])
    )
    table.columns = [f"{lead} • {months[lead]}" for lead in table.columns]
    print(table.to_string(float_format=lambda value: f"{value: .4f}"))


if __name__ == "__main__":
    main()
