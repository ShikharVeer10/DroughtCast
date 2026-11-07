import numpy as np
import pandas as pd

from data.build_dataset import _standardize_accumulation
from experiments.audit_spei_target_integrity import (
    _ranked_nonparametric_value,
    build_extreme_event_audit,
    reconstruct_beed,
)
from experiments.build_spei_alternative_targets import (
    _fisk_transform,
    transform_accumulation,
)
from experiments.validate_spei_reference import align_monthly


def test_trailing_accumulation_uses_only_current_and_prior_months():
    values = pd.Series([1.0, 2.0, 4.0, 8.0], index=pd.date_range("2000-01-01", periods=4, freq="MS"))
    accumulated = values.rolling(3, min_periods=3).sum()
    assert accumulated.iloc[:2].isna().all()
    assert accumulated.iloc[2] == 7.0
    assert accumulated.iloc[3] == 14.0


def test_calendar_month_fit_excludes_post_calibration_observations():
    dates = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    values = pd.Series(np.sin(np.arange(len(dates)) / 9) + np.arange(len(dates)) / 100, index=dates)
    start, end = pd.Timestamp("1990-01-01"), pd.Timestamp("2010-12-01")
    first, first_diagnostics = transform_accumulation(values, "nonparametric", start, end)
    changed = values.copy()
    changed.loc[changed.index > end] += 1e6
    second, second_diagnostics = transform_accumulation(changed, "nonparametric", start, end)
    assert first_diagnostics == second_diagnostics
    np.testing.assert_allclose(first.loc[:end], second.loc[:end], equal_nan=True)
    assert not np.allclose(first.loc["2011-01-01":], second.loc["2011-01-01":], equal_nan=True)


def test_fisk_calibration_parameters_are_independent_of_post_calibration_values():
    dates = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    values = pd.Series(100 + np.sin(np.arange(len(dates)) / 9) * 30 + np.arange(len(dates)) / 100, index=dates)
    start, end = pd.Timestamp("1990-01-01"), pd.Timestamp("2010-12-01")
    _, first_diagnostics = transform_accumulation(values, "fisk_log_logistic", start, end)
    changed = values.copy()
    changed.loc[changed.index > end] = -1e8
    _, second_diagnostics = transform_accumulation(changed, "fisk_log_logistic", start, end)
    assert first_diagnostics == second_diagnostics


def test_nonparametric_score_clip_boundary_is_explicit():
    calibration = np.arange(-10.0, 11.0)
    low, _ = _ranked_nonparametric_value(-1e6, calibration)
    high, _ = _ranked_nonparametric_value(1e6, calibration)
    assert low < -3.5
    assert high > 3.5
    assert np.clip(low, -3.5, 3.5) == -3.5
    assert np.clip(high, -3.5, 3.5) == 3.5
    # Builder's public result is bounded, including the calibration tails.
    index = pd.date_range("1990-01-01", periods=360, freq="MS")
    balance = pd.Series(np.tile(np.arange(30.0), 12), index=index)
    out = _standardize_accumulation(balance, 3, "nonparametric")
    assert out.dropna().abs().max() <= 3.5


def test_fisk_probability_support_and_score_clipping_boundaries():
    params = {"shape_beta": 2.0, "scale_alpha": 1.0, "location_gamma": 0.0}
    values = _fisk_transform(np.array([-1.0, 0.0, 1.0, 1e12]), params)
    assert np.isfinite(values).all()
    assert values[0] == values[1]  # at and below support use the same documented floor
    assert values[0] == -3.5  # lower probability floor maps below the score cap
    assert values[2] == 0.0  # CDF 0.5 maps to the standard-normal median
    assert values[3] == 3.5  # upper tail is capped after probability clipping


def test_failed_fisk_fit_is_recorded_and_not_silently_dropped():
    dates = pd.date_range("1990-01-01", "2012-12-01", freq="MS")
    values = pd.Series(5.0, index=dates)
    result, diagnostics = transform_accumulation(
        values, "fisk_log_logistic", pd.Timestamp("1990-01-01"), pd.Timestamp("2010-12-01")
    )
    assert all(row["fit_status"] == "failed" for row in diagnostics)
    assert result.isna().all()
    assert all("fit_error" in row for row in diagnostics)


def test_reference_alignment_keeps_missing_month_and_rejects_duplicates():
    local = pd.DataFrame({"site": ["beed"] * 3, "date": pd.date_range("2019-07-01", periods=3, freq="MS"), "x": [1.0, 2.0, 3.0]})
    reference = pd.DataFrame({"site": ["beed"] * 2, "date": ["2019-07-10", "2019-09-10"], "y": [1.0, 3.0]})
    aligned = align_monthly(local, reference, "x", "y")
    assert len(aligned) == 3
    assert pd.isna(aligned.loc[aligned.date.eq(pd.Timestamp("2019-08-01")), "y"]).all()
    duplicate = pd.concat([reference, reference.iloc[[0]]], ignore_index=True)
    try:
        align_monthly(local, duplicate, "x", "y")
    except ValueError as exc:
        assert "Duplicate site/month" in str(exc)
    else:
        raise AssertionError("duplicate site-month reference was accepted")


def test_beed_september_2019_reconstruction_matches_canonical():
    canonical = pd.read_csv("data/maharashtra/monthly_sites.csv", parse_dates=["date"])
    reference = pd.read_csv("results/maharashtra/speibase_validation/matched_monthly_points.csv", parse_dates=["date"])
    rows, summary = reconstruct_beed(canonical, reference)
    inputs = rows.loc[rows.row_type.eq("input_month")]
    calibration = rows.loc[rows.row_type.eq("september_calibration_observation")]
    assert inputs.date.tolist() == ["2019-04", "2019-05", "2019-06", "2019-07", "2019-08", "2019-09"]
    assert len(calibration) == summary["calibration_n"]
    assert summary["final_clipped_spei"] == -3.5
    assert summary["speibase_v211_spei_6"] > 0
    assert np.isclose(inputs.six_month_accumulation_mm.iloc[-1], summary["accumulated_water_balance_mm"])


def test_dated_nonparametric_1990_alternative_reproduces_all_canonical_targets():
    canonical = pd.read_csv("data/maharashtra/monthly_sites.csv", parse_dates=["date"])
    alternative = pd.read_csv("results/maharashtra/phase1/spei_alternative_monthly.csv", parse_dates=["date"])
    sites = sorted(canonical.site.astype(str).str.lower().unique())
    assert len(sites) == 12
    for site in sites:
        for scale in (3, 6, 12):
            left = canonical.loc[canonical.site.astype(str).str.lower().eq(site), ["date", f"spei_{scale}"]]
            right = alternative.loc[
                alternative.site.astype(str).str.lower().eq(site)
                & alternative.scale.eq(scale)
                & alternative.transformation.eq("nonparametric")
                & alternative.baseline.eq("1990-2010"), ["date", "spei"]
            ]
            paired = left.merge(right, on="date", validate="one_to_one").dropna()
            assert len(paired) > 0
            np.testing.assert_allclose(paired[f"spei_{scale}"], paired.spei, rtol=0, atol=1e-12)


def test_extreme_disagreement_ranking_is_deterministic():
    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    local = pd.DataFrame({
        "site": ["beed"] * 4, "date": dates, "PRECTOTCORR": [1.0] * 4, "pet_mm": [20.0] * 4,
        **{f"spei_{scale}": [0.0, -1.0, -2.0, -3.0] for scale in (3, 6, 12)},
    })
    reference = pd.DataFrame({
        "site": ["beed"] * 4, "date": dates,
        **{f"spei_{scale}_reference": [0.0, 0.0, 0.0, -2.0] for scale in (3, 6, 12)},
    })
    metadata = {"site_reference_cells": [{"site": "beed", "reference_lat": 18.75, "reference_lon": 75.75}]}
    config = {"sites": {"beed": {"lat": 18.2, "lon": 75.7}}}
    first, detail = build_extreme_event_audit(local, reference, metadata, config)
    second, _ = build_extreme_event_audit(local, reference, metadata, config)
    assert first["absolute_difference"].tolist() == second["absolute_difference"].tolist()
    assert first.absolute_difference.iloc[0] == 2.0
    assert detail["paired_counts_by_scale"][0]["paired_site_months"] == 4

