from pathlib import Path

import pandas as pd


def test_prospective_forecast_outputs_have_expected_keys_and_finite_values():
    root = Path("results/maharashtra/phase3/phase3_14_prospective_forecasts")
    classical = pd.read_csv(root / "classical_forecasts.csv")
    neural = pd.read_csv(root / "neural_forecasts.csv")
    assert len(classical) == 720
    assert len(neural) == 2160
    assert classical["site"].nunique() == neural["site"].nunique() == 12
    assert classical["target_month"].nunique() == neural["target_month"].nunique() == 4
    assert classical["scale"].nunique() == neural["scale"].nunique() == 3
    assert classical["prediction"].notna().all()
    assert neural["prediction"].notna().all()
    assert sorted(neural["quantile"].unique()) == [0.1, 0.5, 0.9]
