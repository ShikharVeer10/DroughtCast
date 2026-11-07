
DATASET_PROTOCOL_VERSION = 9
SPLITS = {
    "train_end": "2010-12-01",
    "val_start": "2012-01-01",
    "val_end": "2015-12-01",
    "test_start": "2017-01-01",
    "test_end": "2025-12-01",
}

SPLIT_BOUNDARIES = dict(SPLITS)
SPEI_CALIBRATION_END = SPLITS["train_end"]
INDEX_ALIGNMENT = {
    "forecast_issue": "month-end of input month; target starts next month",
    "max_staleness_months": 12,
    "oni": {
        "source": "NOAA CPC centered 3-month season",
        "lag_months": 2,
    },
    "dmi": {
        "source": "NOAA PSL monthly",
        "lag_months": 1,
    },
    "nao": {
        "source": "NOAA PSL monthly",
        "lag_months": 1,
    },
    "pdo": {
        "source": "NOAA PSL monthly",
        "lag_months": 1,
    },
}
MODEL_FEATURES = [
    "PRECTOTCORR",
    "T2M",
    "T2MDEW",
    "WS10M",
    "GWETTOP",
    "GWETROOT",
    "VPD",
    "T2M_RANGE",
    "oni",
    "dmi",
    "nao",
    "pdo",
    "spei_3_obs",
    "spei_6_obs",
    "spei_12_obs",
]
REQUIRED_SITE_COLUMNS = [
    "date",
    "site",
    *MODEL_FEATURES,
    "spei_3",
    "spei_6",
    "spei_12",
]