"""Probe cached or upstream NASA/NOAA coverage without changing model data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.build_dataset import NOAA, POWER_VARS, _parse_noaa_text, fetch_power
from data.protocol import DATASET_PROTOCOL_VERSION


def main() -> None:
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    cache = Path("data/raw_phase1_history")
    output = Path("results/maharashtra/phase1/source_availability_1981_2026.json")
    report = {"period_requested": ["1981-01", "2026-10"], "power": {}, "indices": {}, "limitations": []}
    for site, spec in cfg["sites"].items():
        frame = fetch_power(site, spec, 1981, 2026, cache, end_month=10)
        sidecar = cache / f"{site}_power_1981_2026_v{DATASET_PROTOCOL_VERSION}.metadata.json"
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        site_result = {
            "source_url": "https://power.larc.nasa.gov/api/temporal/monthly/point",
            "source_file": str(cache / f"{site}_power_1981_2026_v{DATASET_PROTOCOL_VERSION}.csv"),
            "retrieved_utc": metadata.get("retrieved_utc"),
            "coverage_months": len(frame),
            "variables": {},
        }
        for variable in POWER_VARS:
            values = pd.to_numeric(frame[variable], errors="coerce").replace([-999, -99, -9999, -32768], np.nan)
            valid = values.notna()
            site_result["variables"][variable] = {
                "earliest": values.index[valid][0].strftime("%Y-%m") if valid.any() else None,
                "latest": values.index[valid][-1].strftime("%Y-%m") if valid.any() else None,
                "missing_months": int((~valid).sum()),
                "coverage_percent": float(valid.mean() * 100),
                "units": metadata.get("units", {}).get(variable),
                "known_limitations": "POWER monthly endpoint may omit recent incomplete months; source-specific historical start can vary.",
            }
        report["power"][site] = site_result
    index_cache = Path("data/raw_protocol7_refresh")
    index_dates = pd.date_range("1981-01-01", "2026-10-01", freq="MS")
    for name, url in NOAA.items():
        path = index_cache / f"noaa_{name}.txt"
        series = _parse_noaa_text(name, path.read_text(encoding="utf-8"))
        series = series.reindex(index_dates)
        valid = series.notna()
        report["indices"][name] = {
            "source_url": url,
            "source_file": str(path),
            "earliest": series.index[valid][0].strftime("%Y-%m") if valid.any() else None,
            "latest": series.index[valid][-1].strftime("%Y-%m") if valid.any() else None,
            "missing_months": int((~valid).sum()),
            "coverage_percent": float(valid.mean() * 100),
            "retrieved_utc": json.loads((index_cache / f"noaa_{name}.metadata.json").read_text(encoding="utf-8")).get("retrieved_utc"),
            "units": {"oni": "degC anomaly", "dmi": "degC anomaly", "nao": "standardized index", "pdo": "degC anomaly"}[name],
            "parsing_method": "NOAA CPC ONI season labels mapped to monthly labels; PSL year rows parsed Jan-Dec with documented sentinels removed.",
            "note": "PSL/CPC source timestamps are observation labels; protocol publication lags are applied in the built dataset.",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}; protocol schema {DATASET_PROTOCOL_VERSION}")


if __name__ == "__main__":
    main()
