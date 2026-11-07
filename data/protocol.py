"""Versioned rules for information available to monthly forecasts."""

DATASET_PROTOCOL_VERSION = 8

# Single chronological protocol shared by site, spatial and paper pipelines.
SPLITS = {
    "train_end": "2010-12-01",
    "val_start": "2012-01-01",
    "val_end": "2015-12-01",
    "test_start": "2017-01-01",
    "test_end": "2025-12-01",   # 2026 stays an untouched final holdout
}
SPLIT_BOUNDARIES = {**SPLITS, "test_end": "2025-12-01"}
SPEI_CALIBRATION_END = SPLITS["train_end"]

# Forecast inputs end at month m; target months begin at m+1.
# ONI is a centered three-month season; its month label is the center month,
# so two months are removed. The monthly PSL indices use a one-month lag as a
# conservative publication-delay proxy. Exact historical publication timestamps
# are not available in these source tables; carries are bounded to 12 months.
INDEX_ALIGNMENT = {
    "forecast_issue": "month-end of input month; target starts next month",
    "max_staleness_months": 12,
    "oni": {"source": "NOAA CPC centered 3-month season", "lag_months": 2},
    "dmi": {"source": "NOAA PSL monthly", "lag_months": 1},
    "nao": {"source": "NOAA PSL monthly", "lag_months": 1},
    "pdo": {"source": "NOAA PSL monthly", "lag_months": 1},
}

REQUIRED_SITE_COLUMNS = [
    "PRECTOTCORR", "T2M", "T2MDEW", "WS10M",
    "GWETTOP", "GWETROOT", "VPD", "T2M_RANGE",
    "oni", "dmi", "nao", "pdo", "spei_3_obs", "spei_6_obs", "spei_12_obs",
]
