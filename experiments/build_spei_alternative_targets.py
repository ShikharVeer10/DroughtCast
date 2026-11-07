"""Create dated SPEI transformation alternatives without editing canonical targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import norm

from data.build_dataset import _nonparametric_z_scores, estimate_pet_hargreaves
from data.protocol import SPEI_CALIBRATION_END

CALIBRATION_END = pd.Timestamp(SPEI_CALIBRATION_END)
SCALES = (3, 6, 12)
MAX_ABS_Z = 3.5


def _fisk_pwm_fit(calibration: np.ndarray) -> dict:
    """Fit the 3-parameter log-logistic using the project's PWM equations."""
    cal = np.sort(np.asarray(calibration, dtype=float))
    cal = cal[np.isfinite(cal)]
    if len(cal) < 10:
        raise ValueError(f"insufficient calibration data: {len(cal)} < 10")
    f = (np.arange(1, len(cal) + 1, dtype=float) - 0.35) / len(cal)
    w0, w1, w2 = (float(np.mean((1.0 - f) ** power * cal)) for power in (0, 1, 2))
    denominator = 6.0 * w1 - w0 - 6.0 * w2
    beta = (2.0 * w1 - w0) / denominator if denominator != 0 else np.nan
    if not np.isfinite(beta) or beta <= 1.0:
        raise ValueError(f"invalid Fisk shape beta={beta}")
    gamma_product = float(np.exp(np.log(np.pi / beta) - np.log(np.sin(np.pi / beta))))
    alpha = (w0 - 2.0 * w1) * beta / gamma_product
    location = w0 - alpha * gamma_product
    params = {"shape_beta": float(beta), "scale_alpha": float(alpha), "location_gamma": float(location)}
    if not np.isfinite(list(params.values())).all() or alpha <= 0:
        raise ValueError(f"invalid Fisk parameters: {params}")
    return params


def _fisk_transform(values: np.ndarray, params: dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    probability = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    above_location = finite & (values > params["location_gamma"])
    probability[finite & ~above_location] = 1e-6
    logit = params["shape_beta"] * (
        np.log(values[above_location] - params["location_gamma"])
        - np.log(params["scale_alpha"])
    )
    probability[above_location] = expit(logit)
    return np.clip(norm.ppf(np.clip(probability, 1e-6, 1 - 1e-6)), -MAX_ABS_Z, MAX_ABS_Z)


def transform_accumulation(
    accumulation: pd.Series,
    transformation: str,
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp = CALIBRATION_END,
) -> tuple[pd.Series, list[dict]]:
    """Fit each calendar-month distribution on calibration months only."""
    result = pd.Series(np.nan, index=accumulation.index, dtype=float)
    diagnostics = []
    for month in range(1, 13):
        month_values = accumulation.loc[accumulation.index.month == month]
        mask = (
            (month_values.index >= calibration_start)
            & (month_values.index <= calibration_end)
        )
        calibration = month_values.loc[mask].dropna().to_numpy(dtype=float)
        entry = {
            "calendar_month": month,
            "calibration_start": calibration_start.strftime("%Y-%m"),
            "calibration_end": calibration_end.strftime("%Y-%m"),
            "calibration_n": len(calibration),
            "transformation": transformation,
        }
        try:
            if transformation == "nonparametric":
                if len(calibration) < 20:
                    raise ValueError(f"insufficient calibration data: {len(calibration)} < 20")
                transformed = _nonparametric_z_scores(month_values.to_numpy(dtype=float), calibration, MAX_ABS_Z)
                entry.update({"fit_status": "empirical_reference", "parameters": None})
            elif transformation == "fisk_log_logistic":
                params = _fisk_pwm_fit(calibration)
                transformed = _fisk_transform(month_values.to_numpy(dtype=float), params)
                entry.update({
                    "fit_status": "fit_ok",
                    "parameters": params,
                    "extreme_parameter_warning": bool(
                        params["shape_beta"] > 100 or params["scale_alpha"] > 100000
                    ),
                })
            else:
                raise ValueError(f"unsupported transformation: {transformation}")
            finite_count = int(np.isfinite(transformed).sum())
            expected_count = int(np.isfinite(month_values.to_numpy(dtype=float)).sum())
            entry["transformed_finite_n"] = finite_count
            entry["nonfinite_output_n"] = expected_count - finite_count
            if finite_count:
                result.loc[month_values.index] = transformed
        except (ValueError, FloatingPointError, OverflowError) as exc:
            entry.update({"fit_status": "failed", "fit_error": str(exc), "transformed_finite_n": 0})
            # Failed fits remain visible in diagnostics and produce missing values.
        diagnostics.append(entry)
    return result, diagnostics


def _load_site_balance(site: str, spec: dict, canonical: pd.DataFrame, history_dir: Path) -> tuple[pd.Series, dict]:
    canonical = canonical.sort_values("date").set_index("date")
    canonical_balance = canonical["PRECTOTCORR"] * canonical.index.days_in_month - canonical["pet_mm"]
    history_path = history_dir / f"{site}_power_1981_2026_v10.csv"
    history = pd.read_csv(history_path, parse_dates=["date"]).set_index("date").sort_index()
    pre_dates = pd.date_range("1981-01-01", "1989-12-01", freq="MS")
    prehistory = history.reindex(pre_dates)
    prehistory["pet_mm"] = estimate_pet_hargreaves(prehistory, spec["lat"])
    pre_balance = prehistory["PRECTOTCORR"] * prehistory.index.days_in_month - prehistory["pet_mm"]
    combined = pd.concat([pre_balance, canonical_balance]).sort_index()
    if combined.index.duplicated().any():
        raise ValueError(f"Duplicate water-balance month for {site}")
    audit = {
        "pre_1990_source": str(history_path),
        "pre_1990_expected_months": len(pre_dates),
        "pre_1990_missing_precipitation_months": int(prehistory["PRECTOTCORR"].isna().sum()),
        "pre_1990_missing_pet_input_months": int(prehistory[["T2M", "T2M_MAX", "T2M_MIN"]].isna().any(axis=1).sum()),
        "canonical_period": [canonical.index.min().strftime("%Y-%m"), canonical.index.max().strftime("%Y-%m")],
    }
    return combined.rename("water_balance_mm"), audit


def _previous_generalized_logistic_summary(path: Path) -> dict:
    previous = pd.read_csv(path)
    shape = pd.to_numeric(previous["shape"], errors="coerce")
    return {
        "source": str(path),
        "fit_rows": len(previous),
        "converged_rows": int(previous["converged"].fillna(False).astype(bool).sum()),
        "shape_median": float(shape.median()),
        "shape_75th_percentile": float(shape.quantile(0.75)),
        "shape_max": float(shape.max()),
        "shape_over_100_rows": int((shape > 100).sum()),
        "decision": "excluded_unstable_parameters",
        "reason": "Prior 1981-2010 generalized-logistic MLE diagnostics show highly variable/extreme shape estimates, including many values over 100 (near-step CDF behavior); values are not presented as a stable target transformation.",
    }


def generate(config_path: Path, canonical_path: Path, history_dir: Path, output_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    canonical = pd.read_csv(canonical_path, parse_dates=["date"])
    expected_sites = set(cfg["data"]["site_filter"])
    if set(canonical.site.unique()) != expected_sites:
        raise ValueError("Canonical site list does not match configured Maharashtra sites")
    rows, fit_report, source_report = [], [], {}
    specs = [
        ("nonparametric", "1990-2010", 1990),
        ("nonparametric", "1981-2010", 1981),
        ("fisk_log_logistic", "1990-2010", 1990),
        ("fisk_log_logistic", "1981-2010", 1981),
    ]
    genlog_summary = _previous_generalized_logistic_summary(
        output_dir / "spei_distribution_comparison" / "generalized_logistic_fits.csv"
    )
    for site in sorted(expected_sites):
        site_data = canonical.loc[canonical.site == site].copy()
        water_balance_extended, source_audit = _load_site_balance(site, cfg["sites"][site], site_data, history_dir)
        source_report[site] = source_audit
        indexed_site_data = site_data.set_index("date").sort_index()
        canonical_balance = (
            indexed_site_data["PRECTOTCORR"] * indexed_site_data.index.days_in_month
            - indexed_site_data["pet_mm"]
        )
        for scale in SCALES:
            for transformation, baseline, baseline_start in specs:
                baseline_balance = canonical_balance if baseline_start == 1990 else water_balance_extended
                accumulation = baseline_balance.rolling(scale, min_periods=scale).sum()
                calibrated, diag = transform_accumulation(
                    accumulation, transformation, pd.Timestamp(f"{baseline_start}-01-01")
                )
                for item in diag:
                    fit_report.append({"site": site, "scale": scale, "baseline": baseline, **item})
                values = calibrated.reindex(site_data.date)
                status_by_month = {item["calendar_month"]: item["fit_status"] for item in diag}
                accumulation_by_date = accumulation.reindex(site_data.date).to_numpy(dtype=float)
                for date, value, accumulated in zip(site_data.date, values.to_numpy(dtype=float), accumulation_by_date):
                    month_status = status_by_month[pd.Timestamp(date).month]
                    if month_status == "failed":
                        row_status = "fit_failed"
                    elif not np.isfinite(accumulated):
                        row_status = "missing_accumulation"
                    else:
                        row_status = "computed"
                    rows.append({"site": site, "date": date, "scale": scale, "transformation": transformation, "baseline": baseline, "spei": value, "fit_status": row_status})
            # Explicit NaN rows make the omitted generalized-logistic combination auditable.
            for date in site_data.date:
                rows.append({"site": site, "date": date, "scale": scale, "transformation": "generalized_logistic_mle", "baseline": "1981-2010", "spei": np.nan, "fit_status": "excluded_unstable_parameters"})
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = output_dir / "spei_alternative_monthly.csv"
    pd.DataFrame(rows).to_csv(targets_path, index=False, date_format="%Y-%m-%d")
    fisk_fits = [
        item for item in fit_report
        if item["transformation"] == "fisk_log_logistic" and item["fit_status"] == "fit_ok"
    ]
    fisk_parameter_summary = {}
    for baseline in ("1990-2010", "1981-2010"):
        baseline_fits = [item for item in fisk_fits if item["baseline"] == baseline]
        for parameter in ("shape_beta", "scale_alpha", "location_gamma"):
            values = np.asarray([item["parameters"][parameter] for item in baseline_fits], dtype=float)
            fisk_parameter_summary.setdefault(baseline, {})[parameter] = {
                "min": float(np.min(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "max": float(np.max(values)) if len(values) else None,
            }
        fisk_parameter_summary[baseline]["extreme_parameter_warning_count"] = sum(
            item.get("extreme_parameter_warning", False) for item in baseline_fits
        )
    diagnostics = {
        "calibration_end": CALIBRATION_END.strftime("%Y-%m"),
        "water_balance": "monthly PRECTOTCORR (mm/day) * calendar days - Hargreaves-Samani pet_mm; pre-1990 balance from cached NASA POWER monthly inputs; 1990 onward exactly from canonical data columns.",
        "accumulation": "Trailing rolling sums with full scale months required; missing months are not imputed.",
        "transformations": {
            "nonparametric": "Empirical plotting positions and normal scores, calibrated separately by calendar month; same code/helper and +/-3.5 output cap as local builder.",
            "fisk_log_logistic": "Three-parameter log-logistic using the existing project's probability-weighted-moment equations; 3-parameter support location may be negative; values at/below support map to lower probability floor 1e-6; normal-score output is capped at +/-3.5.",
            "generalized_logistic_mle": genlog_summary,
        },
        "calibration_isolation": "Every month-specific fit uses only dates between baseline start and 2010-12 inclusive. Values after calibration end are transformed only after fitting and do not enter fitted parameters.",
        "source_quality": source_report,
        "fisk_parameter_summary": fisk_parameter_summary,
        "extreme_parameter_screen": "A visible warning is set when beta > 100 or alpha > 100000. This is a screening flag for highly concentrated/extreme estimates, not an automatic invalidation rule; all such fits and outputs remain in the artifact and require review.",
        "fit_diagnostics": fit_report,
        "outputs": {"monthly_targets": str(targets_path)},
        "target_file_rows": len(rows),
        "nonfinite_target_count": int(pd.DataFrame(rows)["spei"].isna().sum()),
    }
    (output_dir / "spei_fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--canonical", type=Path, default=Path("data/maharashtra/monthly_sites.csv"))
    parser.add_argument("--history-dir", type=Path, default=Path("data/raw_phase1_history"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/maharashtra/phase1"))
    args = parser.parse_args()
    result = generate(args.config, args.canonical, args.history_dir, args.output_dir)
    print(f"Wrote {result['target_file_rows']} dated alternative target rows")


if __name__ == "__main__":
    main()
