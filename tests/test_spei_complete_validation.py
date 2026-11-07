import json

import numpy as np
import pandas as pd
import pytest

from experiments.build_spei_alternative_targets import (
    _fisk_pwm_fit,
    _fisk_transform,
    transform_accumulation,
)
from experiments.prepare_spei_v212_audit import write_blocked_outputs


def test_fit_parameters_use_calibration_period_only():
    dates = pd.date_range("1981-01-01", "2015-12-01", freq="MS")
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.arange(len(dates)) * 0.001
    base = pd.Series(values, index=dates)
    changed = base.copy()
    changed.loc["2011-01-01":] += 50

    base_output, base_diagnostics = transform_accumulation(base, "nonparametric", pd.Timestamp("1981-01-01"))
    changed_output, changed_diagnostics = transform_accumulation(changed, "nonparametric", pd.Timestamp("1981-01-01"))

    assert base_output.index.equals(base.index)
    np.testing.assert_allclose(base_output.loc[:"2010-12-01"], changed_output.loc[:"2010-12-01"], equal_nan=True)
    assert [d["calibration_n"] for d in base_diagnostics] == [d["calibration_n"] for d in changed_diagnostics]
    assert all(d["fit_status"] == "empirical_reference" for d in base_diagnostics)


def test_fisk_fit_rejects_insufficient_samples_and_preserves_nan_outputs():
    with pytest.raises(ValueError, match="insufficient calibration data"):
        _fisk_pwm_fit(np.arange(5, dtype=float))

    params = {"shape_beta": 2.0, "scale_alpha": 1.0, "location_gamma": -1.0}
    output = _fisk_transform(np.array([np.nan, -2.0, 0.0]), params)
    assert np.isnan(output[0])
    assert np.isfinite(output[1:]).all()


def test_v212_unavailable_metadata_does_not_claim_reference_values(tmp_path):
    probe = {
        "checked_utc": "2026-11-07T00:00:00+00:00",
        "files": {"spei03.nc": {"url": "https://example.invalid/spei03.nc", "http_status": 404, "available": False}},
    }

    metadata = write_blocked_outputs(tmp_path, probe)

    points = pd.read_csv(tmp_path / "speibase_v212_site_month.csv")
    version_comparison = pd.read_csv(tmp_path / "speibase_version_comparison.csv")
    assert points.empty
    assert metadata["v212_downloaded"] is False
    assert metadata["status"] == "blocked_reference_files_unavailable"
    assert metadata["verified_documentation"]["version"] == "SPEIbase v2.12.0"
    assert metadata["verified_documentation"]["expected_scales"] == [3, 6, 12]
    assert version_comparison.loc[0, "status"] == "blocked_reference_unavailable"
    assert json.loads((tmp_path / "speibase_v212_metadata.json").read_text())[
        "source_probe"
    ] == probe
