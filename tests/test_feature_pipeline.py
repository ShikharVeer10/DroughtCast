import numpy as np
import pandas as pd
import pytest

from data.build_dataset import derive_biophysical_features
from data.quality import validate_site_table


def test_temperature_range_calculation():

    frame = pd.DataFrame({
        "T2M_MAX": [35.0],
        "T2M_MIN": [20.0],
        "T2MDEW": [15.0],
    })

    result = derive_biophysical_features(frame)

    assert result.loc[0, "T2M_RANGE"] == 15.0


def test_vpd_is_non_negative():

    frame = pd.DataFrame({
        "T2M_MAX": [35.0],
        "T2M_MIN": [20.0],
        "T2MDEW": [15.0],
    })

    result = derive_biophysical_features(frame)

    assert result.loc[0, "VPD"] >= 0.0


def test_missing_temperature_produces_missing_features():

    frame = pd.DataFrame({"T2M_MAX": [np.nan],"T2M_MIN": [20.0],"T2MDEW": [15.0],})

    result = derive_biophysical_features(frame)

    assert pd.isna(result.loc[0, "T2M_RANGE"])
    assert pd.isna(result.loc[0, "VPD"])


def test_missing_required_feature_fails():

    frame = pd.DataFrame({
        "date": [pd.Timestamp("2010-01-01")],
        "site": ["aurangabad"],
        "PRECTOTCORR": [2.0],
        "spei_3": [-1.0],
        "spei_6": [-0.8],
        "spei_12": [-0.5],
    })

    with pytest.raises(ValueError, match="missing columns"):
        validate_site_table(
            frame,
            ["PRECTOTCORR", "GWETROOT"],
            strict=True,
        )