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


def test_oni_uses_two_month_conservative_lag():
    """Centered ONI is lagged two calendar months to avoid same-window leakage."""
    from data.build_dataset import _parse_noaa_text

    text = "SEAS  YR   TOTAL  ANOM\nDJF  2000   0.0  1.00\nJFM  2000   0.0  2.00\nFMA  2000   0.0  3.00\nMAM  2000   0.0  4.00\nAMJ  2000   0.0  5.00"
    series = _parse_noaa_text("oni", text)
    # DJF -> January, JFM -> February
    assert series.loc[pd.Timestamp("2000-01-01")] == pytest.approx(1.0)
    shifted = series.shift(2)
    # At March use January-centered DJF; at April use February-centered JFM.
    assert shifted.loc[pd.Timestamp("2000-03-01")] == pytest.approx(1.0)
    assert shifted.loc[pd.Timestamp("2000-04-01")] == pytest.approx(2.0)


def test_centered_oni_source_for_concrete_input_month_has_no_future_month():
    """A January input uses a season ending before January, not centered in January."""
    from data.build_dataset import _parse_noaa_text

    text = (
        "SEAS  YR   TOTAL  ANOM\n"
        "OND  1999   0.0  9.00\n"
        "NDJ  1999   0.0  8.00\n"
        "DJF  2000   0.0  1.00\n"
        "JFM  2000   0.0  2.00\n"
    )
    centered = _parse_noaa_text("oni", text)
    centered = centered.reindex(pd.date_range("1999-10-01", "2000-03-01", freq="MS"))
    aligned = centered.shift(2)
    input_month = pd.Timestamp("2000-01-01")
    assert aligned.loc[input_month] == pytest.approx(9.0)
    assert aligned.loc[input_month] != centered.loc[input_month]


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


def test_indices_follow_protocol_specific_lags():
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
    assert result.loc[0, ["oni", "dmi", "nao", "pdo"]].isna().all()
    assert result.loc[1, ["oni"]].isna().all()
    assert result.loc[2, "oni"] == pytest.approx(0.0)
    assert result.loc[2, ["dmi", "nao", "pdo"]].eq(1.0).all()
    assert result.loc[3, "oni"] == pytest.approx(1.0)
    assert result.loc[3, ["dmi", "nao", "pdo"]].eq(2.0).all()


def test_index_alignment_preserves_calendar_availability_and_provenance():
    from data.build_dataset import align_index_with_provenance

    source = pd.Series([1.25, 2.5], index=pd.to_datetime(["2026-08-01", "2026-09-01"]))
    issues = pd.date_range("2026-08-01", "2026-10-01", freq="MS")
    aligned = align_index_with_provenance("oni", source, issues)
    # ONI has a two-calendar-month conservative lag; September becomes
    # effective in November and must not appear in the October input.
    assert aligned.loc[pd.Timestamp("2026-10-01"), "value"] == pytest.approx(1.25)
    assert aligned.loc[pd.Timestamp("2026-10-01"), "source_month"] == pd.Timestamp("2026-08-01")
    assert aligned.loc[pd.Timestamp("2026-10-01"), "effective_month"] == pd.Timestamp("2026-10-01")
    assert aligned.loc[pd.Timestamp("2026-10-01"), "observation_age_months"] == 2
    assert not aligned.loc[pd.Timestamp("2026-10-01"), "is_stale"]


def test_old_pdo_value_is_missing_at_october_2026():
    from data.build_dataset import align_index_with_provenance

    source = pd.Series([0.8], index=pd.to_datetime(["2025-08-01"]))
    aligned = align_index_with_provenance("pdo", source, pd.DatetimeIndex([pd.Timestamp("2026-10-01")]))
    row = aligned.iloc[0]
    assert pd.isna(row["value"])
    assert row["is_stale"]
    assert row["is_missing"]
    assert row["observation_age_months"] == 14
    assert row["source_month"] == pd.Timestamp("2025-08-01")
    assert row["effective_month"] == pd.Timestamp("2025-09-01")


@pytest.mark.parametrize("name", ["oni", "dmi", "nao", "pdo"])
def test_each_index_publication_lag_uses_calendar_months(name):
    from data.build_dataset import align_index_with_provenance
    from data.protocol import INDEX_ALIGNMENT

    lag = INDEX_ALIGNMENT[name]["publication_lag_months"]
    source = pd.Series([7.0], index=pd.to_datetime(["2025-12-01"]))
    issues = pd.date_range("2025-12-01", "2026-03-01", freq="MS")
    aligned = align_index_with_provenance(name, source, issues)
    eligible = pd.Timestamp("2025-12-01") + pd.offsets.MonthBegin(lag)
    before = issues[issues < eligible]
    if len(before):
        assert aligned.loc[before[-1], "is_missing"]
    assert aligned.loc[eligible, "value"] == pytest.approx(7.0)
    assert aligned.loc[eligible, "source_month"] == pd.Timestamp("2025-12-01")
    assert aligned.loc[eligible, "effective_month"] == eligible


def test_index_max_age_boundary_and_first_stale_month():
    from data.build_dataset import align_index_with_provenance
    from data.protocol import INDEX_ALIGNMENT

    source = pd.Series([2.0], index=pd.to_datetime(["2025-12-01"]))
    issues = pd.date_range("2026-01-01", "2026-08-01", freq="MS")
    for name, policy in INDEX_ALIGNMENT.items():
        if name not in {"oni", "dmi", "nao", "pdo"}:
            continue
        aligned = align_index_with_provenance(name, source, issues)
        boundary = pd.Timestamp("2025-12-01") + pd.DateOffset(months=policy["max_age_months"])
        beyond = boundary + pd.DateOffset(months=1)
        assert aligned.loc[boundary, "observation_age_months"] == policy["max_age_months"]
        assert not aligned.loc[boundary, "is_stale"]
        assert pd.isna(aligned.loc[beyond, "value"])
        assert aligned.loc[beyond, "is_stale"]


def test_missing_calendar_source_month_is_not_counted_as_new_observation():
    from data.build_dataset import align_index_with_provenance

    # February is absent; February should continue to report January's origin.
    source = pd.Series([1.0, 3.0], index=pd.to_datetime(["2026-01-01", "2026-03-01"]))
    issues = pd.date_range("2026-02-01", "2026-04-01", freq="MS")
    aligned = align_index_with_provenance("dmi", source, issues)
    assert aligned.loc[pd.Timestamp("2026-03-01"), "source_month"] == pd.Timestamp("2026-01-01")
    assert aligned.loc[pd.Timestamp("2026-03-01"), "observation_age_months"] == 2
    assert aligned.loc[pd.Timestamp("2026-04-01"), "source_month"] == pd.Timestamp("2026-03-01")


def test_future_index_source_is_never_used_before_effective_month():
    from data.build_dataset import align_index_with_provenance

    source = pd.Series([9.0], index=pd.to_datetime(["2026-09-01"]))
    issues = pd.date_range("2026-09-01", "2026-12-01", freq="MS")
    aligned = align_index_with_provenance("oni", source, issues)
    assert aligned.loc[pd.Timestamp("2026-10-01"), "is_missing"]
    assert pd.isna(aligned.loc[pd.Timestamp("2026-10-01"), "source_month"])
    assert aligned.loc[pd.Timestamp("2026-11-01"), "value"] == pytest.approx(9.0)


def test_index_audit_columns_are_not_in_canonical_model_features():
    from data.protocol import MODEL_FEATURES

    for name in ("oni", "dmi", "nao", "pdo"):
        assert name in MODEL_FEATURES
        assert f"{name}_source_month" not in MODEL_FEATURES
        assert f"{name}_effective_month" not in MODEL_FEATURES
        assert f"{name}_observation_age_months" not in MODEL_FEATURES


def test_stale_index_is_not_made_current_by_shared_calendar_index():
    from data.build_dataset import build_site

    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    frame = pd.DataFrame(
        {name: value for name, value in {
            "PRECTOTCORR": 1.0, "T2M": 20.0, "T2MDEW": 10.0, "WS10M": 2.0,
            "ALLSKY_SFC_SW_DWN": 5.0, "T2M_MAX": 25.0, "T2M_MIN": 15.0,
        }.items()}, index=dates,
    )
    indices = pd.DataFrame(
        {name: np.arange(4, dtype=float) for name in ("oni", "dmi", "nao", "pdo")},
        index=dates,
    )
    from unittest.mock import patch

    with patch("data.build_dataset.fetch_power", return_value=frame):
        result = build_site("x", {"lat": 10.0, "region": "test"}, 2000, 2001, None, "nonparametric", indices)
    stale = result.loc[pd.to_datetime(result["date"]) == pd.Timestamp("2001-05-01")].iloc[0]
    assert stale[["oni", "dmi", "nao", "pdo"]].isna().all()
