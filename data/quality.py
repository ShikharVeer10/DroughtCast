"""Dataset quality diagnostics used before scientific evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALES = ("spei_3", "spei_6", "spei_12")


def target_quality_report(
    frame: pd.DataFrame, calibration_end: str = "2010-12-01"
) -> dict:
    """Return per-scale/site distribution and missingness diagnostics."""
    required = {"site", "date", *SCALES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing quality columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["date"])
    report = {
        "rows": len(frame),
        "sites": sorted(frame["site"].astype(str).unique().tolist()),
        "calibration_end": calibration_end,
        "scales": {},
        "warnings": [],
    }
    for scale in SCALES:
        values = pd.to_numeric(frame[scale], errors="coerce")
        calibration = values[dates <= pd.Timestamp(calibration_end)]
        finite = values[np.isfinite(values)]
        stats = {
            "n": len(values),
            "missing": int(values.isna().sum()),
            "finite": len(finite),
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
            "quantiles": {
                str(q): float(finite.quantile(q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
            }
            if len(finite)
            else {},
            "min": float(finite.min()) if len(finite) else float("nan"),
            "max": float(finite.max()) if len(finite) else float("nan"),
            "calibration_n": int(calibration.notna().sum()),
        }
        by_site = {}
        for site, group in frame.assign(_value=values).groupby("site"):
            site_values = group["_value"].dropna()
            by_site[str(site)] = {
                "n": len(site_values),
                "missing": int(group["_value"].isna().sum()),
                "mean": float(site_values.mean()) if len(site_values) else float("nan"),
                "std": float(site_values.std(ddof=1))
                if len(site_values) > 1
                else float("nan"),
                "min": float(site_values.min()) if len(site_values) else float("nan"),
                "max": float(site_values.max()) if len(site_values) else float("nan"),
            }
        stats["by_site"] = by_site
        report["scales"][scale] = stats
        if stats["max"] > 8 or stats["min"] < -8 or stats["std"] > 3:
            report["warnings"].append(
                f"{scale} has an extreme distribution (min={stats['min']}, "
                f"max={stats['max']}, std={stats['std']})"
            )
    return report


def validate_target_quality(report: dict, fail_on_warning: bool = False) -> None:
    warnings = report.get("warnings", [])
    if fail_on_warning and warnings:
        raise ValueError("Target quality validation failed: " + "; ".join(warnings))


def write_target_quality_report(
    csv_path: str | Path, output_path: str | Path, fail_on_warning: bool = False
) -> dict:
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    report = target_quality_report(frame)
    validate_target_quality(report, fail_on_warning)
    Path(output_path).write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="results/target_quality.json")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    result = write_target_quality_report(
        args.csv_path, args.output, args.fail_on_warning
    )
    print(json.dumps(result, indent=2, allow_nan=True))
