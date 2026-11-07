import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.analyze_spei_phase14 import (
    _top30_reconstruction,
    _uncapped_nonparametric,
    category,
    comparison_metrics,
    run,
)


def _synthetic_local(start="1990-01-01", end="2015-12-01"):
    dates = pd.date_range(start, end, freq="MS")
    phase = np.arange(len(dates), dtype=float)
    return pd.DataFrame({
        "site": "beed", "date": dates,
        "PRECTOTCORR": 1.2 + 0.4 * np.sin(phase / 7),
        "pet_mm": 50 + 8 * np.cos(phase / 12),
    })


def test_seven_drought_category_boundaries_match_specification():
    values = np.array([-2.1, -2.0, -1.99, -1.5, -1.49, -1.0, -0.99, 0.99, 1.0, 1.49, 1.5, 1.99, 2.0, np.nan])
    np.testing.assert_array_equal(category(values), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, -1])


def test_accumulation_propagates_missing_input_through_complete_trailing_windows():
    local = _synthetic_local()
    local.loc[local.date.eq(pd.Timestamp("2005-05-01")), "PRECTOTCORR"] = np.nan
    score = _uncapped_nonparametric(local, 3)
    for month in ("2005-05-01", "2005-06-01", "2005-07-01"):
        idx = local.index[local.date.eq(pd.Timestamp(month))][0]
        assert pd.isna(score.loc[idx])
    august_index = local.index[local.date.eq(pd.Timestamp("2005-08-01"))][0]
    assert np.isfinite(score.loc[august_index])


def test_calibration_isolation_and_no_future_values_in_historical_scores():
    local = _synthetic_local(end="2020-12-01")
    original = _uncapped_nonparametric(local, 6)
    changed = local.copy()
    changed.loc[changed.date.gt("2010-12-01"), "PRECTOTCORR"] += 100.0
    revised = _uncapped_nonparametric(changed, 6)
    historical = local.date.le("2010-12-01")
    np.testing.assert_allclose(original[historical], revised[historical], equal_nan=True, rtol=0, atol=1e-12)
    future = local.date.gt("2010-12-01")
    assert not np.allclose(original[future], revised[future], equal_nan=True)


def test_comparison_metrics_use_only_common_valid_observations():
    candidate = pd.Series([0.2, -1.2, -2.1, np.nan])
    comparator = pd.Series([0.0, -1.6, -2.0, -3.0])
    result = comparison_metrics(candidate, comparator)
    assert result["paired_n"] == 3
    assert result["candidate_missing_n"] == 1
    assert result["comparator_only_drought_count"] == 0
    assert result["extreme_drought_union_n"] == 1
    assert result["extreme_drought_event_agreement"] == 1.0


def test_beed_september_2019_top_discrepancy_reconstruction_is_reproducible():
    canonical = pd.read_csv("data/maharashtra/monthly_sites.csv", parse_dates=["date"])
    extremes = pd.read_csv("results/maharashtra/phase1/spei_extreme_event_audit.csv")
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    rows = _top30_reconstruction(canonical, extremes, config)
    event = rows.loc[(rows.site == "beed") & (rows.date == "2019-09") & (rows.scale == 6)].iloc[0]
    assert event.unclipped_local_spei < -3.5
    assert event.clipped_local_spei == -3.5
    assert event.reference_spei_v211 > 0
    assert event.outside_calibration_range
    assert json.loads(event.contributing_months) == ["2019-04", "2019-05", "2019-06", "2019-07", "2019-08", "2019-09"]


def test_phase14_run_preserves_canonical_and_config(tmp_path):
    canonical_path = Path("data/maharashtra/monthly_sites.csv")
    config_path = Path("configs/config.yaml")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    before = (digest(canonical_path), digest(config_path))
    phase_dir = Path("results/maharashtra/phase1")
    for name in ("spei_extreme_event_audit.csv", "spei_alternative_monthly.csv", "spei_fit_diagnostics.json"):
        shutil.copy2(phase_dir / name, tmp_path / name)
    report = run(tmp_path)
    after = (digest(canonical_path), digest(config_path))
    assert before == after
    assert report["canonical_modified"] is False
    assert report["config_modified"] is False
    assert report["fit_failures"]["failed_count"] == 53
    assert report["event_distribution"]["top30_event_count"] == 30
    required = {
        "spei_methodology_comparison.json", "spei_clipping_sensitivity.csv",
        "spei_top30_discrepancy_reconstruction.csv", "spei_input_data_comparison.csv",
        "spei_method_sensitivity.csv", "spei_category_stability.csv", "spei_phase1_4_report.md",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})

