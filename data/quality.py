
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALES = ("spei_3", "spei_6", "spei_12")


def validate_site_table(frame: pd.DataFrame,features: list[str],expected_sites: list[str] | None = None,*,strict: bool = False,) -> dict:

    required = {
        "site",
        "date",
        *features,
        *SCALES,
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Site table missing columns: {sorted(missing)}")

    dates = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError("Invalid dates found in site table")

    if (dates.dt.day != 1).any():
        raise ValueError("All observations must use first-of-month dates")

    if frame.duplicated(["site", "date"]).any():
        raise ValueError("Duplicate site-month observations found")

    sites = sorted(frame["site"].astype(str).unique())

    if strict and expected_sites is not None:
        if set(sites) != set(expected_sites):
            raise ValueError(
                f"Site mismatch: expected={sorted(expected_sites)}, "
                f"actual={sites}"
            )

    report = {}

    calibration_end = pd.Timestamp("2010-12-01")

    for site, group in frame.assign(_date=dates).groupby("site"):

        group = group.sort_values("_date")

        group_dates = pd.to_datetime(group["_date"])

        expected_dates = pd.date_range(
            group_dates.min(),
            group_dates.max(),
            freq="MS",
        )

        actual_dates = pd.DatetimeIndex(group_dates)

        if strict and not actual_dates.equals(expected_dates):
            raise ValueError(f"Monthly chronology has gaps for {site}")

        inputs = group[features].apply(
            pd.to_numeric,
            errors="coerce",
        )

        train_mask = group_dates <= calibration_end

        feature_report = {}

        for column in features:

            values = inputs[column].to_numpy(dtype=float)

            finite = np.isfinite(values)

            training_values = inputs.loc[
                train_mask,
                column,
            ].to_numpy(dtype=float)

            finite_training = np.isfinite(
                training_values
            )

            feature_report[column] = {
                "missing": int((~finite).sum()),
                "finite": int(finite.sum()),
                "training_finite": int(
                    finite_training.sum()
                ),
            }

            if strict and not finite_training.any():
                raise ValueError(
                    f"{column} has no finite training observations "
                    f"for site {site}"
                )

        for column in ("GWETTOP", "GWETROOT"):

            values = inputs[column].dropna()

            if (strict and ((values < 0) | (values > 1)).any()):
                raise ValueError(f"{column} outside [0, 1] for site {site}")

        for column in ("VPD", "T2M_RANGE", "PRECTOTCORR"):
            values = inputs[column].dropna()

            if strict and (values < 0).any():
                raise ValueError(f"Negative {column} values for site {site}")

        target_report = {}

        for target in SCALES:

            values = pd.to_numeric(group[target],errors="coerce",).to_numpy(dtype=float)
            valid_count = int(np.isfinite(values).sum())

            target_report[target] = valid_count

            if strict and valid_count == 0:
                raise ValueError(f"No valid {target} observations for {site}")

        report[str(site)] = {
            "observations": len(group),
            "features": feature_report,
            "targets": target_report,
        }

    return report


def target_quality_report(frame: pd.DataFrame, calibration_end: str = "2010-12-01") -> dict:
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




def write_target_quality_report(csv_path: str | Path, output_path: str | Path, fail_on_warning: bool = False) -> dict:
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    report = target_quality_report(frame)
    validate_target_quality(report, fail_on_warning)
    Path(output_path).write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="results/target_quality.json")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    result = write_target_quality_report(args.csv_path, args.output, args.fail_on_warning)
    print(json.dumps(result, indent=2, allow_nan=True))
