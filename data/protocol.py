
DATASET_PROTOCOL_VERSION = 10
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
    # Maximum age of the source month at the month-end forecast issue date.
    # ONI is a centered 3-month index and is delayed two calendar months;
    # monthly PSL values are delayed one calendar month. Short limits keep
    # delayed updates from being mistaken for current conditions.
    "oni": {
        "source": "NOAA CPC centered 3-month season",
        "source_url": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        "units": "degC anomaly",
        "publication_lag_months": 2,
        "lag_months": 2,
        "max_age_months": 3,
    },
    "dmi": {
        "source": "NOAA PSL monthly",
        "source_url": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
        "units": "degC anomaly",
        "publication_lag_months": 1,
        "lag_months": 1,
        "max_age_months": 3,
    },
    "nao": {
        "source": "NOAA PSL monthly",
        "source_url": "https://psl.noaa.gov/data/correlation/nao.data",
        "units": "standardized index",
        "publication_lag_months": 1,
        "lag_months": 1,
        "max_age_months": 3,
    },
    "pdo": {
        "source": "NOAA PSL monthly",
        "source_url": "https://psl.noaa.gov/data/correlation/pdo.data",
        "units": "degC anomaly",
        "publication_lag_months": 1,
        "lag_months": 1,
        "max_age_months": 6,
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
