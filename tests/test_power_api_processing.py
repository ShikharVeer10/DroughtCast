import numpy as np
import pandas as pd

from data.api import PARAMETERS, normalize_uploaded, pet_daily_aggregate


def test_normalize_uploaded_power_daily_csv():
    source = {name: [1.0, 2.0] for name in PARAMETERS}
    source["YYYYMMDD"] = ["20200101", "20200102"]
    frame = normalize_uploaded(pd.DataFrame(source))
    assert frame.index[0] == pd.Timestamp("2020-01-01")
    assert list(frame.columns) == PARAMETERS
    assert frame.shape == (2, len(PARAMETERS))


def test_daily_aggregate_sums_rain_and_pet_and_means_climate():
    dates = pd.date_range("2020-02-01", "2020-02-29", freq="D")
    daily = pd.DataFrame({
        "PRECTOTCORR": np.ones(len(dates)), "T2M": np.full(len(dates), 20.0),
        "T2MDEW": np.full(len(dates), 10.0), "WS10M": np.full(len(dates), 3.0),
        "ALLSKY_SFC_SW_DWN": np.full(len(dates), 4.0),
        "T2M_MAX": np.full(len(dates), 25.0), "T2M_MIN": np.full(len(dates), 15.0),
    }, index=dates)
    monthly = pet_daily_aggregate(daily, latitude=15.0)
    assert monthly.loc[pd.Timestamp("2020-02-01"), "PRECTOTCORR"] == 29.0
    assert monthly.loc[pd.Timestamp("2020-02-01"), "T2M"] == 20.0
    assert monthly.loc[pd.Timestamp("2020-02-01"), "pet_mm"] > 0
