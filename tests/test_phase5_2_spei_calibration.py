import numpy as np
import pandas as pd

from experiments.phase5_2_spei_calibration import origin_aware_standardize


def test_origin_aware_calibration_ignores_future_observations():
    dates = pd.date_range("1990-01-01", "2015-12-01", freq="MS")
    values = pd.Series(np.sin(np.arange(len(dates)) / 5), index=dates)
    origin = pd.Timestamp("2008-12-01")
    first = origin_aware_standardize(values, 3, "nonparametric", origin, dates[0])
    changed = values.copy()
    changed.loc[changed.index > origin] += 1e6
    second = origin_aware_standardize(changed, 3, "nonparametric", origin, dates[0])
    np.testing.assert_allclose(first.loc[:origin], second.loc[:origin], equal_nan=True)


def test_origin_aware_calibration_rejects_future_start():
    values = pd.Series([1.0] * 24, index=pd.date_range("2000-01-01", periods=24, freq="MS"))
    try:
        origin_aware_standardize(values, 3, "nonparametric", "2000-01-01", "2001-01-01")
    except ValueError as exc:
        assert "after origin" in str(exc)
    else:
        raise AssertionError("future calibration start was accepted")
