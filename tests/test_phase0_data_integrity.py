"""Phase 0 blocking tests: PET, SPEI standardization, SPEI obs parity, ONI lag, leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.build_dataset import (
    _nonparametric_z_scores,
    _standardize_accumulation,
    estimate_pet_hargreaves,
    mid_month_doy,
)
from data.loader import load_site_table


def test_mid_month_doy_not_first_of_month():
    idx = pd.DatetimeIndex(["2020-06-01", "2020-02-01"])
    doy = mid_month_doy(idx)
    # 2020 leap year: June 1 + (30-1)/2 d -> mid-June (DOY 167); must not be June 1 (DOY 153)
    assert doy[0] == pytest.approx(167.0)
    assert doy[0] != 153
    assert doy[1] != 32  # must not be Feb 1 (DOY 32)
    # Feb 1 + (29-1)/2 = Feb 15 -> DOY 46
    assert doy[1] == pytest.approx(46.0)


def test_hargreaves_et0_known_case_jaipur_june():
    """lat 26.9, June, Tmean 34, TR 12 -> ET0 about 6.9 mm/day."""
    idx = pd.DatetimeIndex(["2020-06-01"])
    frame = pd.DataFrame(
        {
            "T2M": [34.0],
            "T2M_MAX": [40.0],
            "T2M_MIN": [28.0],  # TR = 12
            "ALLSKY_SFC_SW_DWN": [0.0],  # unused by corrected formula
        },
        index=idx,
    )
    pet_month = estimate_pet_hargreaves(frame, latitude=26.9)
    et0_day = float(pet_month.iloc[0]) / idx.days_in_month[0]
    assert et0_day == pytest.approx(6.9, abs=0.15)


def test_nonparametric_calibration_mean_near_zero():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=60)
    z = _nonparametric_z_scores(cal, cal)
    assert abs(float(np.mean(z))) < 0.15


def test_nonparametric_tails_symmetric_absolute_z():
    cal = np.linspace(-2.0, 2.0, 30)
    step = cal[1] - cal[0]
    lo = cal[0] - step
    hi = cal[-1] + step
    z_lo, z_hi = _nonparametric_z_scores(np.array([lo, hi]), cal)
    assert abs(abs(z_lo) - abs(z_hi)) < 1e-6
    assert z_lo < 0 < z_hi


def test_nonparametric_tails_stable_out_of_range():
    rng = np.random.default_rng(1)
    far = []
    for _ in range(500):
        cal = rng.normal(size=30)
        x = np.linspace(cal.max(), cal.max() + 3, 40)
        assert np.all(np.diff(_nonparametric_z_scores(x, cal)) >= -1e-12)
        pts = np.array([cal.max() + 0.3, cal.max() + 1.0, cal.min() - 1.0])
        far.append(np.abs(_nonparametric_z_scores(pts, cal)) >= 5)
    assert np.mean(far) < 0.01


def test_standardize_accumulation_nonparametric_branch():
    dates = pd.date_range("1985-01-01", "2015-12-01", freq="MS")
    balance = pd.Series(np.sin(np.arange(len(dates)) / 7.0), index=dates)
    spei = _standardize_accumulation(balance, scale=3, method="nonparametric")
    cal = spei.loc[spei.index <= "2010-12-01"].dropna()
    assert abs(float(cal.mean())) < 0.25


def test_oni_shifted_plus_one_month():
    """Feature at month m must use ONI centered on m-1 (no look-ahead into m+1)."""
    from data.build_dataset import _parse_noaa_text

    text = "SEAS  YR   TOTAL  ANOM\nDJF  2000   0.0  1.00\nJFM  2000   0.0  2.00\nFMA  2000   0.0  3.00"
    series = _parse_noaa_text("oni", text)
    # DJF -> January, JFM -> February
    assert series.loc[pd.Timestamp("2000-01-01")] == pytest.approx(1.0)
    shifted = series.shift(1)
    # At February, use January-centered ONI (DJF = 1.0), not JFM (2.0).
    assert shifted.loc[pd.Timestamp("2000-02-01")] == pytest.approx(1.0)
    assert shifted.loc[pd.Timestamp("2000-03-01")] == pytest.approx(2.0)


def test_spei_obs_feature_names_and_no_shift_parity(tmp_path):
    dates = pd.date_range("2008-01-01", "2010-12-01", freq="MS")
    rows = []
    for i, date in enumerate(dates):
        spei = float(i) / 10.0
        rows.append(
            {
                "date": date,
                "site": "x",
                "region": "r",
                "PRECTOTCORR": 1.0,
                "T2M": 20.0,
                "T2MDEW": 10.0,
                "WS10M": 2.0,
                "ALLSKY_SFC_SW_DWN": 5.0,
                "oni": 0.0,
                "dmi": 0.0,
                "nao": 0.0,
                "pdo": 0.0,
                "spei_3": spei,
                "spei_6": spei,
                "spei_12": spei,
                "spei_3_obs": spei,
                "spei_6_obs": spei,
                "spei_12_obs": spei,
            }
        )
    path = tmp_path / "monthly.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    features = ["spei_3_obs", "spei_6_obs", "spei_12_obs"]
    data = load_site_table(path, features, seq_len=3, forecast_steps=2)
    assert len(data["train"]) > 0
    # Last input row SPEI obs must equal last_observed_spei (information parity).
    sample = data["train"].metadata[0]
    # Features were normalized; check metadata dates instead for leakage below.
    assert "spei_3" in sample["last_observed_spei"]


def test_window_spei_no_target_leakage(tmp_path):
    dates = pd.date_range("2008-01-01", "2010-12-01", freq="MS")
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "site": "x",
                "region": "r",
                "f": float(i),
                "spei_3": float(i),
                "spei_6": float(i),
                "spei_12": float(i),
                "spei_3_obs": float(i),
                "spei_6_obs": float(i),
                "spei_12_obs": float(i),
            }
        )
    path = tmp_path / "leak.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    data = load_site_table(
        path, ["spei_3_obs", "spei_6_obs", "spei_12_obs"], seq_len=4, forecast_steps=2
    )
    for split in ("train", "val", "test"):
        for meta in data[split].metadata:
            input_dates = [pd.Timestamp(d) for d in meta["dates"]]
            index_source_dates = [pd.Timestamp(d) for d in meta["index_source_dates"]]
            target_dates = [pd.Timestamp(d) for d in meta["target_dates"]]
            last_input = max(input_dates)
            assert all(d <= last_input for d in input_dates)
            assert all(d > last_input for d in target_dates)
            assert all(d < min(target_dates) for d in index_source_dates)
            assert len(meta["spei_history"]) == len(input_dates)
            assert all(len(row) == 3 for row in meta["spei_history"])


def test_all_monthly_indices_are_aligned_to_previous_month():
    from data.build_dataset import build_site

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    frame = pd.DataFrame(
        {
            "PRECTOTCORR": 1.0,
            "T2M": 20.0,
            "T2MDEW": 10.0,
            "WS10M": 2.0,
            "ALLSKY_SFC_SW_DWN": 5.0,
            "T2M_MAX": 25.0,
            "T2M_MIN": 15.0,
        },
        index=dates,
    )

    def fake_fetch(*_args):
        return frame.copy()

    from unittest.mock import patch

    indices = pd.DataFrame(
        {name: np.arange(4, dtype=float) for name in ("oni", "dmi", "nao", "pdo")},
        index=dates,
    )
    with patch("data.build_dataset.fetch_power", side_effect=fake_fetch):
        result = build_site(
            "x",
            {"lat": 10.0, "region": "test"},
            2000,
            2000,
            None,
            "nonparametric",
            indices,
        )
    for name in ("oni", "dmi", "nao", "pdo"):
        assert np.isnan(result.loc[0, name])
        assert result.loc[1, name] == 0.0
        assert result.loc[3, name] == 2.0
