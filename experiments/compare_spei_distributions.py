"""Compare SPEI transforms using the same stored precipitation-PET balance."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import genlogistic, kstest

from data.build_dataset import (
    _standardize_accumulation,
    estimate_pet_hargreaves,
    fetch_power,
)
from data.protocol import SPEI_CALIBRATION_END


def _generalized_logistic(
    accum: pd.Series,
    scale: int,
    cutoff: pd.Timestamp,
    calibration_start: pd.Timestamp,
):
    rolling = accum.rolling(scale, min_periods=scale).sum()
    result = pd.Series(np.nan, index=accum.index, dtype=float)
    fit_rows = []
    for month in range(1, 13):
        subset = rolling.loc[
            (rolling.index.month == month)
            & (rolling.index >= calibration_start)
            & (rolling.index <= cutoff)
        ].dropna()
        values = rolling.loc[rolling.index.month == month]
        if len(subset) < 20:
            continue
        try:
            params = genlogistic.fit(subset.to_numpy(dtype=float))
            c, loc, scale_param = params
            if not np.isfinite(params).all() or scale_param <= 0 or c <= 0:
                raise ValueError("invalid fitted parameters")
            cdf = genlogistic.cdf(values.to_numpy(dtype=float), *params)
            from statistics import NormalDist

            result.loc[values.index] = np.asarray(
                [NormalDist().inv_cdf(float(np.clip(p, 1e-6, 1 - 1e-6))) for p in cdf]
            )
            ks = kstest(subset.to_numpy(dtype=float), "genlogistic", args=params)
            fit_rows.append({"calendar_month": month, "n_calibration": len(subset), "converged": True, "shape": c, "location": loc, "scale": scale_param, "ks_d": ks.statistic, "ks_p_naive": ks.pvalue})
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            fit_rows.append({"calendar_month": month, "n_calibration": len(subset), "converged": False, "fit_error": str(exc)})
    return result, fit_rows


def _metrics(values: pd.Series, calibration_mask: pd.Series) -> dict:
    finite = values[np.isfinite(values)]
    cal = values[calibration_mask & np.isfinite(values)]
    return {
        "n_finite": len(finite),
        "n_missing": int(values.isna().sum()),
        "mean": float(finite.mean()) if len(finite) else None,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
        "calibration_n": len(cal),
        "nonfinite_count": int((~np.isfinite(values.to_numpy(dtype=float))).sum()),
        "moderate_drought_fraction_le_-1": float((finite <= -1).mean()) if len(finite) else None,
        "severe_drought_fraction_le_-1_5": float((finite <= -1.5).mean()) if len(finite) else None,
        "extreme_drought_fraction_le_-2": float((finite <= -2).mean()) if len(finite) else None,
    }


def main() -> None:
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    data = pd.read_csv(cfg["data"]["path"], parse_dates=["date"])
    cutoff = pd.Timestamp(SPEI_CALIBRATION_END)
    long_start = pd.Timestamp("1981-01-01")
    output = Path("results/maharashtra/phase1/spei_distribution_comparison")
    output.mkdir(parents=True, exist_ok=True)
    rows, fits = [], []
    for site, group in data.groupby("site", sort=True):
        group = group.set_index("date").sort_index()
        water_balance = group["PRECTOTCORR"] * group.index.days_in_month - group["pet_mm"]
        cal_mask = pd.Series(group.index <= cutoff, index=group.index)
        historical = fetch_power(site, cfg["sites"][site], 1981, 2026, Path("data/raw_phase1_history"), end_month=10)
        historical["pet_mm"] = estimate_pet_hargreaves(historical, cfg["sites"][site]["lat"])
        long_balance = historical["PRECTOTCORR"] * historical.index.days_in_month - historical["pet_mm"]
        comparison_idx = group.index
        long_balance = long_balance.reindex(long_balance.index.union(comparison_idx)).sort_index()
        for scale in (3, 6, 12):
            existing = group[f"spei_{scale}"]
            fisk = _standardize_accumulation(water_balance, scale, "log-logistic", str(cutoff))
            nonparam_long = _standardize_accumulation(long_balance, scale, "nonparametric", str(cutoff), str(long_start)).reindex(comparison_idx)
            fisk_long = _standardize_accumulation(long_balance, scale, "log-logistic", str(cutoff), str(long_start)).reindex(comparison_idx)
            genlog, fit_rows = _generalized_logistic(long_balance, scale, cutoff, long_start)
            genlog = genlog.reindex(comparison_idx)
            for fit in fit_rows:
                fits.append({"site": site, "scale": scale, **fit})
            for method, values in (
                ("nonparametric_current_1990_baseline", existing),
                ("fisk_log_logistic_1990_baseline", fisk),
                ("nonparametric_1981_baseline", nonparam_long),
                ("fisk_log_logistic_1981_baseline", fisk_long),
                ("generalized_logistic_mle_1981_baseline", genlog),
            ):
                rows.append({"site": site, "scale": scale, "method": method, **_metrics(values, cal_mask)})
    pd.DataFrame(rows).to_csv(output / "site_scale_metrics.csv", index=False)
    pd.DataFrame(fits).to_csv(output / "generalized_logistic_fits.csv", index=False)
    print(f"Wrote {output}; comparison uses calibration data through {cutoff:%Y-%m} only for fitting")


if __name__ == "__main__":
    main()
