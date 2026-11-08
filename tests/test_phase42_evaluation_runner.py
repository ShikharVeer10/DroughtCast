import hashlib
import json

import pandas as pd
import pytest

from evaluation.independent import run_independent_evaluation


def test_runner_requires_accepted_frozen_dataset(tmp_path):
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    pd.DataFrame(
        [{"site": "akola", "forecast_origin": "2026-10", "target_month": "2026-11",
          "scale": "spei_3", "lead": "T+1", "prediction": 0.0, "model": "ridge"}]
    ).to_csv(forecast_dir / "classical_forecasts.csv", index=False)
    pd.DataFrame(columns=["site", "target_month", "scale", "prediction", "model", "quantile"]).to_csv(
        forecast_dir / "neural_forecasts.csv", index=False
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "pending"}), encoding="utf-8")
    dataset = tmp_path / "observed.csv"
    pd.DataFrame([{"site": "akola", "target_month": "2026-11",
                   "scale": "spei_3", "observed": 0.0}]).to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="not frozen"):
        run_independent_evaluation(forecast_dir, dataset, manifest)


def test_runner_matches_frozen_dataset_and_writes_metrics(tmp_path):
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    rows = [{"site": "akola", "forecast_origin": "2026-10", "target_month": "2026-11",
             "scale": "spei_3", "lead": "T+1", "prediction": 1.0, "model": "ridge", "quantile": ""}]
    for name in ("classical_forecasts.csv", "neural_forecasts.csv"):
        pd.DataFrame(rows if name.startswith("classical") else
                     [{"site": "akola", "forecast_origin": "2026-10", "target_month": "2026-11",
                       "scale": "spei_3", "lead": "T+1", "prediction": 1.0,
                       "model": "droughtcast_neural", "quantile": 0.5}]
                     ).to_csv(forecast_dir / name, index=False)
    dataset = tmp_path / "observed.csv"
    pd.DataFrame([{"site": "akola", "target_month": "2026-11",
                   "scale": "spei_3", "observed": 0.0}]).to_csv(dataset, index=False)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "accepted", "dataset_sha256": digest}), encoding="utf-8")
    result = run_independent_evaluation(forecast_dir, dataset, manifest, output_dir=tmp_path / "out")
    assert set(result["model"]) == {"ridge", "droughtcast_neural"}
    assert (tmp_path / "out" / "overall_metrics.csv").is_file()
