from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.loader import load_site_table
from data.preprocessing import ClimateNormalizer
from inference import _load_recent_input


def test_normalizer_imputes_with_training_means_only():
    training = np.array([[1.0, 10.0], [3.0, 20.0], [np.nan, 30.0]])
    normalizer = ClimateNormalizer().fit(training)
    changed_future = np.array([[1e9, -1e9]])
    other = ClimateNormalizer().fit(np.vstack([training[:2], changed_future]))
    assert np.allclose(normalizer.mean, [2.0, 20.0])
    assert not np.allclose(normalizer.mean, other.mean)
    transformed = normalizer.transform(np.array([[np.nan, 25.0], [np.inf, np.nan]]))
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] == 0.0
    assert transformed[1, 0] == 0.0
    assert transformed[1, 1] == 0.0


def test_october_2026_inference_preprocessing_imputes_stale_indices(tmp_path):
    with open("configs/config.yaml", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    features = cfg["data"]["features"]
    frame = pd.read_csv(cfg["data"]["path"], parse_dates=["date"])
    site = "aurangabad"
    site_rows = frame.loc[frame.site.eq(site)]
    assert site_rows.date.max() == pd.Timestamp("2026-10-01")
    assert site_rows.iloc[-1][["dmi", "pdo"]].isna().all()
    assert site_rows.iloc[-1][["oni", "nao"]].notna().all()

    site_dataset = tmp_path / "aurangabad_monthly.csv"
    site_rows.to_csv(site_dataset, index=False)
    source_metadata = Path(__file__).parents[1] / "data/maharashtra/dataset_metadata.json"
    (tmp_path / "dataset_metadata.json").write_text(
        source_metadata.read_text(encoding="utf-8"), encoding="utf-8"
    )
    metadata_path = tmp_path / "preprocessing.json"
    datasets = load_site_table(
        site_dataset,
        features=features,
        seq_len=6,
        forecast_steps=cfg["data"]["forecast_steps"],
        metadata_path=metadata_path,
        site_filter=[site],
        split_config=cfg["data"]["split_boundaries"],
    )
    assert len(datasets["test"]) > 0
    x, dates, _ = _load_recent_input(
        Path(__file__).parents[1] / cfg["data"]["path"], site, features, 6, metadata_path
    )
    assert x.shape == (6, 15)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert dates[-1] == pd.Timestamp("2026-10-01")
    assert np.all(x[-1, features.index("dmi")] == 0.0)
    assert np.all(x[:, features.index("pdo")] == 0.0)
