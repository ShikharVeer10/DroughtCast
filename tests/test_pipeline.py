import numpy as np
import pandas as pd
import torch

from data.loader import split_for_date, load_site_table
from data.preprocessing import ClimateNormalizer
from evaluation.baselines import persistence
from models.forecaster import MultiScaleForecaster


def test_split_dates_and_gap():
    assert split_for_date(pd.Timestamp("2010-12-01")) == "train"
    assert split_for_date(pd.Timestamp("2011-06-01")) is None
    assert split_for_date(pd.Timestamp("2012-01-01")) == "val"
    assert split_for_date(pd.Timestamp("2016-06-01")) is None
    assert split_for_date(pd.Timestamp("2017-01-01")) == "test"


def test_normalizer_uses_given_training_rows_and_preserves_nan():
    values = np.array([[1.0, 2.0], [3.0, np.nan]])
    normalizer = ClimateNormalizer().fit(values)
    assert np.allclose(normalizer.mean, [2.0, 2.0])
    assert np.isnan(normalizer.transform(np.array([[np.nan, 2.0]]))[0, 0])


def test_date_aligned_windows_mask_missing_targets(tmp_path):
    dates = pd.date_range("2009-01-01", "2010-12-01", freq="MS")
    frame = pd.DataFrame({"date": dates, "site": "x", "f": np.arange(len(dates), dtype=float),
                          "spei_3": np.arange(len(dates), dtype=float),
                          "spei_6": np.arange(len(dates), dtype=float),
                          "spei_12": np.arange(len(dates), dtype=float)})
    frame.loc[20, "spei_6"] = np.nan
    path = tmp_path / "monthly.csv"
    frame.to_csv(path, index=False)
    data = load_site_table(path, ["f"], seq_len=3, forecast_steps=2)
    assert len(data["train"]) > 0
    assert data["train"].Y.shape[1:] == (3, 2)
    assert not data["train"].mask.all()
    assert len(data["val"]) == 0 and len(data["test"]) == 0


def test_leave_region_out_keeps_heldout_only_in_test(tmp_path):
    dates = pd.date_range("2008-01-01", "2018-12-01", freq="MS")
    rows = []
    for site, region, offset in (("a", "source", 0.0), ("b", "held", 100.0)):
        for i, date in enumerate(dates):
            rows.append({"date": date, "site": site, "region": region,
                         "f": offset + i, "spei_3": np.sin(i),
                         "spei_6": np.cos(i), "spei_12": np.sin(i / 2)})
    path = tmp_path / "regions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    data = load_site_table(path, ["f"], seq_len=3, forecast_steps=2,
                           holdout_regions=["held"])
    assert data["train"].metadata and all(m["region"] == "source" for m in data["train"].metadata)
    assert data["val"].metadata and all(m["region"] == "source" for m in data["val"].metadata)
    assert data["test"].metadata and all(m["region"] == "held" for m in data["test"].metadata)


def test_persistence_uses_last_spei_per_scale():
    metadata = [{"last_observed_spei": {"spei_3": -1.2, "spei_6": .3, "spei_12": -.8}}]
    result = persistence(metadata, forecast_steps=2)
    assert np.allclose(result, [[[-1.2, -1.2], [.3, .3], [-.8, -.8]]])


def test_model_shapes_variants_and_attention():
    for encoder in ("lstm", "gru", "transformer"):
        for attention in ("head_specific", "shared", "none"):
            config = {"data": {"input_dim": 7, "forecast_steps": 4},
                      "model": {"encoder": encoder, "attention": attention,
                                "hidden_dim_1": 8, "hidden_dim_2": 8, "dropout": 0.0,
                                "variable_gate": True}}
            model = MultiScaleForecaster(config)
            prediction, extra = model(torch.randn(2, 6, 7))
            assert all(prediction[k].shape == (2, 4) for k in ("spei_3", "spei_6", "spei_12"))
            if attention == "shared":
                assert extra["attention"].shape == (2, 6, 1)
                assert torch.allclose(extra["attention"].sum(dim=1), torch.ones(2, 1), atol=1e-5)
            else:
                assert extra["attention"].shape == (2, 3, 6, 1)
                assert torch.allclose(extra["attention"].sum(dim=2), torch.ones(2, 3, 1), atol=1e-5)


def test_model_quantile_shape_for_encoder_variants():
    for encoder in ("lstm", "gru", "transformer"):
        config = {"data": {"input_dim": 5, "forecast_steps": 4},
                  "model": {"encoder": encoder, "attention": "shared", "quantiles": [0.1, 0.5, 0.9],
                            "hidden_dim_1": 8, "hidden_dim_2": 8, "dropout": 0.0}}
        model = MultiScaleForecaster(config)
        output, _ = model(torch.randn(2, 6, 5))
        assert all(output[k].shape == (2, 4, 3) for k in ("spei_3", "spei_6", "spei_12"))


def test_model_separate_encoders_variant():
    config = {"data": {"input_dim": 3, "forecast_steps": 2},
              "model": {"encoder": "lstm", "separate_encoders": True,
                        "attention": "shared", "hidden_dim_1": 4, "hidden_dim_2": 4}}
    prediction, aux = MultiScaleForecaster(config)(torch.randn(2, 5, 3))
    assert all(value.shape == (2, 2) for value in prediction.values())
    assert aux["attention"].shape == (2, 3, 5, 1)
    assert torch.allclose(aux["attention"].sum(dim=2), torch.ones(2, 3, 1), atol=1e-5)
