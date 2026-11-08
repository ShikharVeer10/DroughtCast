from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data.preprocessing import validate_normalizer_metadata
from data.protocol import DATASET_PROTOCOL_VERSION, MODEL_FEATURES, SPLIT_BOUNDARIES
from experiments.run_phase2_1_baselines import (
    CATEGORY_NAMES,
    _metric_tables,
    _validate_windows,
    candidate_configs,
    classification_metrics,
    climatology_predictions,
    drought_categories,
    make_estimator,
    persistence_predictions,
    regression_metrics,
    select_estimator,
)


class DummyDataset(SimpleNamespace):
    def __len__(self):
        return len(self.metadata)


def _sample(site="aurangabad", start="2010-01-01"):
    dates = pd.date_range(start, periods=6, freq="MS")
    targets = pd.date_range(dates[-1] + pd.offsets.MonthBegin(1), periods=4, freq="MS")
    history = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8]])
    return {
        "site": site,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "target_dates": [d.strftime("%Y-%m-%d") for d in targets],
        "spei_history": history.tolist(),
        "last_observed_spei": {"spei_3": 0.6, "spei_6": 0.7, "spei_12": 0.8},
        "index_provenance_by_index": [
            {
                name: {
                    "source_month": (d - pd.offsets.MonthBegin(2 if name == "oni" else 1)).strftime("%Y-%m-%d"),
                    "effective_month": d.strftime("%Y-%m-%d"),
                    "observation_age_months": 2 if name == "oni" else 1,
                    "is_stale": False,
                    "is_missing": False,
                }
                for name in ("oni", "dmi", "nao", "pdo")
            }
            for d in dates
        ],
    }


def test_persistence_uses_latest_finite_input_spei_only():
    metadata = _sample()
    metadata["spei_history"][4][0] = np.nan
    metadata["spei_history"][5][0] = np.nan
    dataset = DummyDataset(metadata=[metadata])
    predictions = persistence_predictions(dataset)
    np.testing.assert_allclose(predictions[0, 0], [0.4] * 4)
    np.testing.assert_allclose(predictions[0, 1], [0.7] * 4)
    np.testing.assert_allclose(predictions[0, 2], [0.8] * 4)


def test_climatology_uses_only_unique_training_rows_and_calendar_month():
    frame = pd.DataFrame(
        {
            "site": ["aurangabad", "aurangabad", "aurangabad", "aurangabad"],
            "date": pd.to_datetime(["1990-02-01", "1991-02-01", "1990-03-01", "2012-02-01"]),
            "spei_3": [1.0, 3.0, 10.0, 10000.0],
            "spei_6": [2.0, 4.0, 20.0, 10000.0],
            "spei_12": [3.0, 5.0, 30.0, 10000.0],
        }
    )
    target = SimpleNamespace(
        Y=torch.zeros((1, 3, 4), dtype=torch.float32),
        metadata=[{**_sample(), "target_dates": ["2012-02-01", "2012-02-01", "2012-02-01", "2012-02-01"]}],
    )
    predictions, report = climatology_predictions(SimpleNamespace(), target, frame)
    np.testing.assert_allclose(predictions[0, :, :], np.asarray([[2.0] * 4, [3.0] * 4, [4.0] * 4]))
    assert report["unique_training_rows"] == 3
    assert report["fallback_counts"] == {}


def test_calendar_fallback_uses_training_site_mean_then_global_mean():
    frame = pd.DataFrame(
        {
            "site": ["aurangabad", "beed"],
            "date": pd.to_datetime(["1990-01-01", "1990-01-01"]),
            "spei_3": [1.0, 3.0],
            "spei_6": [2.0, 4.0],
            "spei_12": [3.0, 5.0],
        }
    )
    target = DummyDataset(Y=torch.zeros((2, 3, 4)), metadata=[_sample("aurangabad"), _sample("unknown")])
    target.metadata[0]["target_dates"] = ["2012-02-01"] * 4
    target.metadata[1]["target_dates"] = ["2012-02-01"] * 4
    predictions, report = climatology_predictions(SimpleNamespace(), target, frame)
    np.testing.assert_allclose(predictions[0, :, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(predictions[1, :, 0], [2.0, 3.0, 4.0])
    assert report["fallback_counts"] == {"site_overall": 12, "pooled_overall": 12}


@pytest.mark.parametrize(
    "values,expected",
    [
        ([-2.0, -1.5, -1.0, -0.99, 0.99, 1.0, 1.5, 2.0], [0, 1, 2, 3, 3, 4, 5, 6]),
        ([np.nan], [-1]),
    ],
)
def test_documented_drought_category_boundaries(values, expected):
    assert drought_categories(np.asarray(values)).tolist() == expected


def test_regression_and_classification_metrics_exclude_missing_targets():
    regression = regression_metrics(np.asarray([1.0, 2.0, np.nan]), np.asarray([1.0, 3.0, 100.0]))
    assert regression["n_valid"] == 2
    assert regression["mae"] == 0.5
    classification, matrix = classification_metrics(np.asarray([-2.0, -1.2, 0.0]), np.asarray([-2.2, -0.8, 0.1]))
    assert classification["category_accuracy"] == pytest.approx(2 / 3)
    assert classification["severe_drought_recall"] == 0.0
    assert classification["severe_or_extreme_event_recall_le_minus_1_5"] == 1.0
    assert np.asarray(matrix).shape == (7, 7)


def test_ridge_validation_selection_fits_train_and_scores_validation():
    x_train = np.arange(80, dtype=float).reshape(20, 4)
    y_train = x_train[:, 0] * 0.4 - x_train[:, 1] * 0.1
    x_val = np.arange(40, 64, dtype=float).reshape(6, 4)
    y_val = x_val[:, 0] * 0.4 - x_val[:, 1] * 0.1
    estimator, info = select_estimator("ridge", x_train, y_train, x_val, y_val, 42)
    assert info["selection_split"] == "validation"
    assert info["fit_rows"] == 20
    assert info["selection_rows"] == 6
    assert np.isfinite(estimator.predict(x_val)).all()


def test_random_forest_fit_and_fixed_seed_reproducibility():
    x = np.arange(80, dtype=float).reshape(20, 4)
    y = x[:, 0] * 0.2 + x[:, 2]
    params = candidate_configs("random_forest")[0]
    first = make_estimator("random_forest", params, 42).fit(x, y).predict(x)
    second = make_estimator("random_forest", params, 42).fit(x, y).predict(x)
    np.testing.assert_allclose(first, second)


def test_xgboost_runs_or_reports_an_explicit_dependency_failure():
    try:
        estimator = make_estimator("xgboost", candidate_configs("xgboost")[0], 42)
    except RuntimeError as exc:
        assert "XGBoost import failed" in str(exc)
        return
    x = np.arange(80, dtype=float).reshape(20, 4)
    y = x[:, 0] * 0.2 - x[:, 2]
    estimator.fit(x, y)
    assert np.isfinite(estimator.predict(x[:3])).all()


def test_window_audit_checks_target_alignment_and_index_availability():
    datasets = {
        "train": DummyDataset(metadata=[_sample()]),
        "val": DummyDataset(metadata=[_sample(start="2012-01-01")]),
        "test": DummyDataset(metadata=[_sample(start="2017-01-01")]),
    }
    report = _validate_windows(datasets, ["aurangabad"])
    assert report["no_future_targets_in_input_windows"]
    assert report["no_cross_split_window_detected"]
    assert report["no_future_climate_index_dates"]
    assert report["date_ranges"]["train"]["input_start"] == "2010-01"
    assert report["date_ranges"]["val"]["target_start"] == "2012-07"
    assert report["date_ranges"]["test"]["target_start"] == "2017-07"


def test_window_audit_detects_future_climate_index_source():
    sample = _sample()
    sample["index_provenance_by_index"][0]["oni"]["source_month"] = "2010-02-01"
    datasets = {
        "train": DummyDataset(metadata=[sample]),
        "val": DummyDataset(metadata=[_sample(start="2012-01-01")]),
        "test": DummyDataset(metadata=[_sample(start="2017-01-01")]),
    }
    report = _validate_windows(datasets, ["aurangabad"])
    assert not report["no_future_climate_index_dates"]
    assert report["future_source_month_records"] == 1


def test_metric_table_has_overall_and_per_site_rows_and_all_category_names():
    metadata = [_sample("aurangabad"), _sample("beed")]
    dataset = DummyDataset(
        Y=torch.zeros((2, 3, 4), dtype=torch.float32),
        mask=torch.ones((2, 3, 4), dtype=torch.bool),
        metadata=metadata,
    )
    predictions = {("persistence", "all_features"): np.zeros((2, 3, 4))}
    tables, _, matrices = _metric_tables(predictions, {"train": dataset, "val": dataset, "test": dataset}, ["aurangabad", "beed"])
    assert set(tables["site"]) == {"aurangabad", "beed", "ALL"}
    assert len(CATEGORY_NAMES) == 7
    assert len(matrices) == 3 * 4 * 3


def test_verified_normalizer_metadata_is_accepted_and_legacy_is_rejected():
    import json
    from pathlib import Path

    metadata = json.loads(Path("data/normalizers/maharashtra_protocol10.json").read_text())
    validate_normalizer_metadata(
        metadata,
        MODEL_FEATURES,
        expected_protocol_version=DATASET_PROTOCOL_VERSION,
        expected_split_boundaries=SPLIT_BOUNDARIES,
        expected_seq_len=6,
        expected_forecast_steps=4,
    )
    legacy = json.loads(Path("data/meta.json").read_text())
    with pytest.raises(ValueError, match="schema"):
        validate_normalizer_metadata(legacy, MODEL_FEATURES)
