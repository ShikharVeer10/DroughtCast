"""Record v2.12 source availability without substituting an older product."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

SCALES = (3, 6, 12)
BASE_URL = "https://spei.csic.es/spei_database_2_12/nc"


def probe_v212(timeout: int = 25) -> dict:
    checked = datetime.now(UTC).isoformat()
    versions = {"v2.11.0": "https://spei.csic.es/spei_database_2_11/nc", "v2.12.0": BASE_URL}
    results = {}
    for version, base_url in versions.items():
        files = {}
        for scale in SCALES:
            filename = f"spei{scale:02d}.nc"
            url = f"{base_url}/{filename}"
            try:
                response = requests.head(url, timeout=timeout, allow_redirects=True)
                files[filename] = {
                    "url": url,
                    "http_status": response.status_code,
                    "content_length": response.headers.get("Content-Length"),
                    "content_type": response.headers.get("Content-Type"),
                    "available": response.status_code == 200,
                }
            except requests.RequestException as exc:
                files[filename] = {"url": url, "available": False, "error": str(exc)}
        results[version] = files
    return {"checked_utc": checked, "versions": results}


def write_blocked_outputs(output_dir: Path, probe: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "speibase_v212_site_month.csv"
    pd.DataFrame(columns=[
        "site", "date", "scale", "spei", "reference_version", "site_lat", "site_lon",
        "grid_lat", "grid_lon", "distance_km", "extraction_method",
    ]).to_csv(points_path, index=False)

    comparison_path = output_dir / "speibase_version_comparison.csv"
    pd.DataFrame([{
        "status": "blocked_reference_unavailable",
        "version_old": "SPEIbase v2.11.0",
        "version_new": "SPEIbase v2.12.0",
        "comparison_period": "1990-01 through 2024-12",
        "reason": "Official documented path responded HTTP 404 for all required NetCDF files; no v2.12 values were downloaded or inferred.",
    }]).to_csv(comparison_path, index=False)

    metadata = {
        "status": "blocked_reference_files_unavailable",
        "verified_documentation": {
            "version": "SPEIbase v2.12.0",
            "input_product": "CRU TS 4.10",
            "documented_coverage": ["1901-01", "2025-12"],
            "documented_resolution": "0.5 degree; monthly",
            "expected_scales": [3, 6, 12],
            "expected_netcdf_names_per_official_generation_code": ["spei03.nc", "spei06.nc", "spei12.nc"],
            "source_input_files": ["cru_ts4.10.1901.2025.pre.dat.nc", "cru_ts4.10.1901.2025.pet.dat.nc"],
            "source_input_units_assumed_by_generation_code": {"pre": "monthly precipitation amount", "pet": "daily PET multiplied by days per month"},
            "expected_variable": "spei (specified by the official generation code; v2.12 file itself unavailable for inspection)",
            "units": "1 (dimensionless, specified by the official generation code; v2.12 file itself unavailable for inspection)",
            "missing_value_encoding": "1e30 in official NetCDF generation code; v2.12 file itself unavailable for inspection",
            "calculation": "Official generation code uses CRU precipitation and PET, then SPEI::spei at each scale. Package default is log-logistic fitted by unbiased PWM; default reference period is the entire supplied series.",
            "documentation_url": "https://github.com/sbegueria/SPEIbase/blob/master/README.md",
            "generation_code_url": "https://github.com/sbegueria/SPEIbase/blob/master/R/computeSPEI.R",
            "function_code_url": "https://github.com/sbegueria/SPEIbase/blob/master/R/functions.R",
        },
        "source_probe": probe,
        "existing_v211_artifact": "results/maharashtra/speibase_validation/matched_monthly_points.csv",
        "v211_source_url": "https://spei.csic.es/spei_database_2_11/nc/",
        "v212_downloaded": False,
        "v212_extraction_performed": False,
        "reason": "The v2.12 release is described in the official repository, but the expected official NetCDF endpoints return HTTP 404. No values were substituted from v2.11.",
        "outputs": {"v212_site_month": str(points_path), "version_comparison": str(comparison_path)},
    }
    (output_dir / "speibase_v212_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/maharashtra/phase1"))
    args = parser.parse_args()
    write_blocked_outputs(args.output_dir, probe_v212())
    print(f"Wrote v2.12 availability audit to {args.output_dir}")


if __name__ == "__main__":
    main()
