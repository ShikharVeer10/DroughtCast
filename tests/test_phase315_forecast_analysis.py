from pathlib import Path

import pandas as pd


def test_forecast_analysis_outputs_match_phase314_records():
    root = Path("results/maharashtra/phase3/phase3_15_forecast_analysis")
    assert len(pd.read_csv(root / "location_wise_summary.csv")) == 144
    assert len(pd.read_csv(root / "scale_comparison.csv")) == 12
    assert pd.read_csv(root / "model_comparison.csv")["mean_prediction"].notna().all()
    assert len(list((root / "figures").glob("*.png"))) == 4
