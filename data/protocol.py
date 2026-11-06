"""Versioned rules for information available to monthly forecasts."""

DATASET_PROTOCOL_VERSION = 5

# Forecast inputs end at month m; target months begin at m+1.
# ONI is a centered three-month season; its month label is the center month,
# so two months are removed. The monthly PSL indices use a one-month lag as a
# conservative publication-delay proxy. Missing late releases are carried
# forward from the latest already-published value (never backward-filled).
INDEX_ALIGNMENT = {
    "forecast_issue": "month-end of input month; target starts next month; lagged indices forward-filled from latest prior release",
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
