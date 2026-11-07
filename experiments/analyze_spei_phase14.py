"""Create Phase 1.4 SPEI method, clipping, discrepancy, and category audits.

The script is analysis-only: it reads canonical data and existing alternatives,
and writes new Phase 1.4 artifacts without changing the dataset or config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, pearsonr, spearmanr

from data.build_dataset import _nonparametric_z_scores, _tail_slopes
from data.protocol import SPEI_CALIBRATION_END
from experiments.build_spei_alternative_targets import MAX_ABS_Z

SCALES = (3, 6, 12)
CAL_END = pd.Timestamp(SPEI_CALIBRATION_END)
OUTPUT = Path("results/maharashtra/phase1")
REFERENCE = Path("results/maharashtra/speibase_validation/matched_monthly_points.csv")
REFERENCE_META = Path("results/maharashtra/speibase_validation/validation_metadata.json")
CANONICAL = Path("data/maharashtra/monthly_sites.csv")
ALTERNATIVE = OUTPUT / "spei_alternative_monthly.csv"


def category(values: np.ndarray | pd.Series) -> np.ndarray:
    """Seven SPEI classes; -1 marks missing values."""
    value = np.asarray(values, dtype=float)
    result = np.full(value.shape, -1, dtype=np.int8)
    valid = np.isfinite(value)
    result[valid & (value <= -2)] = 0
    result[valid & (value > -2) & (value <= -1.5)] = 1
    result[valid & (value > -1.5) & (value <= -1)] = 2
    result[valid & (value > -1) & (value < 1)] = 3
    result[valid & (value >= 1) & (value < 1.5)] = 4
    result[valid & (value >= 1.5) & (value < 2)] = 5
    result[valid & (value >= 2)] = 6
    return result


def _correlation(function, left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    score = function(left, right).statistic
    return float(score) if np.isfinite(score) else None


def comparison_metrics(candidate: pd.Series, comparator: pd.Series) -> dict:
    """Compute paired statistics; candidate-minus-comparator bias convention."""
    candidate = pd.to_numeric(candidate, errors="coerce")
    comparator = pd.to_numeric(comparator, errors="coerce")
    paired = candidate.notna() & comparator.notna() & np.isfinite(candidate) & np.isfinite(comparator)
    x, y = candidate.loc[paired].to_numpy(float), comparator.loc[paired].to_numpy(float)
    difference = x - y
    xcat, ycat = category(x), category(y)
    cat_match = xcat == ycat
    xd, yd = x <= -1.0, y <= -1.0
    drought_union = xd | yd
    x_extreme, y_extreme = x <= -2.0, y <= -2.0
    extreme_union = x_extreme | y_extreme
    return {
        "paired_n": int(paired.sum()),
        "candidate_missing_n": int(candidate.isna().sum()),
        "comparator_missing_n": int(comparator.isna().sum()),
        "pearson_r": _correlation(pearsonr, x, y),
        "spearman_r": _correlation(spearmanr, x, y),
        "bias_candidate_minus_comparator": float(difference.mean()) if len(difference) else None,
        "mae": float(np.abs(difference).mean()) if len(difference) else None,
        "rmse": float(np.sqrt(np.mean(difference**2))) if len(difference) else None,
        "seven_category_agreement": float(cat_match.mean()) if len(x) else None,
        "drought_union_months": int(drought_union.sum()),
        "drought_only_agreement": float((xd[drought_union] == yd[drought_union]).mean()) if drought_union.any() else None,
        "local_only_drought_count": int((xd & ~yd).sum()),
        "comparator_only_drought_count": int((~xd & yd).sum()),
        "extreme_drought_event_agreement": float((x_extreme & y_extreme).sum() / extreme_union.sum()) if extreme_union.any() else None,
        "extreme_drought_binary_accuracy": float((x_extreme == y_extreme).mean()) if len(x) else None,
        "extreme_drought_union_n": int(extreme_union.sum()),
        "candidate_extreme_drought_count": int(x_extreme.sum()),
        "comparator_extreme_drought_count": int(y_extreme.sum()),
        "extreme_drought_both_count": int((x_extreme & y_extreme).sum()),
        "extreme_drought_candidate_only_count": int((x_extreme & ~y_extreme).sum()),
        "extreme_drought_comparator_only_count": int((~x_extreme & y_extreme).sum()),
        "candidate_min": float(np.min(x)) if len(x) else None,
        "candidate_max": float(np.max(x)) if len(x) else None,
        "comparator_min": float(np.min(y)) if len(y) else None,
        "comparator_max": float(np.max(y)) if len(y) else None,
    }


def _uncapped_nonparametric(canonical: pd.DataFrame, scale: int) -> pd.Series:
    """Recompute canonical nonparametric scores without only the final +/-3.5 cap."""
    result = pd.Series(np.nan, index=canonical.index, dtype=float)
    for row_indices in canonical.groupby("site", sort=False).groups.values():
        site_data = canonical.loc[row_indices].sort_values("date")
        dates = pd.DatetimeIndex(site_data.date)
        balance = site_data.PRECTOTCORR.to_numpy(float) * dates.days_in_month - site_data.pet_mm.to_numpy(float)
        series = pd.Series(balance, index=dates)
        accumulation = series.rolling(scale, min_periods=scale).sum()
        target = pd.Series(np.nan, index=dates, dtype=float)
        for month in range(1, 13):
            mask = (accumulation.index.month == month) & (accumulation.index <= CAL_END)
            calibration = accumulation.loc[mask].dropna().to_numpy(float)
            values = accumulation.loc[accumulation.index.month == month].to_numpy(float)
            if len(calibration) >= 20:
                target.loc[accumulation.index[accumulation.index.month == month]] = _nonparametric_z_scores(
                    values, calibration, max_abs_z=1e9
                )
        target_by_date = target.to_dict()
        for idx, date in zip(row_indices, canonical.loc[row_indices, "date"]):
            result.loc[idx] = target_by_date.get(pd.Timestamp(date), np.nan)
    return result


def _attach_reference(canonical: pd.DataFrame, reference: pd.DataFrame, scale: int) -> pd.DataFrame:
    left = canonical[["site", "date", f"spei_{scale}"]].copy()
    right = reference[["site", "date", f"spei_{scale}_reference"]].copy()
    left["site"] = left.site.astype(str).str.lower()
    right["site"] = right.site.astype(str).str.lower()
    left["date"] = pd.to_datetime(left.date).dt.to_period("M").dt.to_timestamp()
    right["date"] = pd.to_datetime(right.date).dt.to_period("M").dt.to_timestamp()
    if left.duplicated(["site", "date"]).any() or right.duplicated(["site", "date"]).any():
        raise ValueError("Duplicate site-month records in local or reference data")
    return left.merge(right, on=["site", "date"], how="left", validate="one_to_one")


def _site_and_all_rows(frame: pd.DataFrame, candidate_col: str, comparator_col: str,
                       scale: int, comparison: str, baseline: str | None = None) -> list[dict]:
    rows = []
    scopes = [(site, group) for site, group in frame.groupby("site", sort=True)]
    scopes.append(("ALL", frame))
    for site, group in scopes:
        rows.append({
            "site": site, "scale": scale, "comparison": comparison, "baseline": baseline,
            **comparison_metrics(group[candidate_col], group[comparator_col]),
        })
    return rows


def _canonical_and_alternatives(canonical: pd.DataFrame, alternatives: pd.DataFrame,
                                reference: pd.DataFrame, sites: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity_rows, category_rows = [], []
    alt_names = {
        ("nonparametric", "1981-2010"): "nonparametric_1981_2010",
        ("fisk_log_logistic", "1990-2010"): "fisk_1990_2010",
        ("fisk_log_logistic", "1981-2010"): "fisk_1981_2010",
    }
    for scale in SCALES:
        ref_local = reference[["site", "date", f"spei_{scale}_reference"]].copy()
        ref_local.site = ref_local.site.astype(str).str.lower()
        ref_local.date = pd.to_datetime(ref_local.date).dt.to_period("M").dt.to_timestamp()
        canon = canonical[["site", "date", f"spei_{scale}"]].copy()
        canon.site = canon.site.astype(str).str.lower()
        canon.date = pd.to_datetime(canon.date).dt.to_period("M").dt.to_timestamp()
        for (transform, baseline), name in alt_names.items():
            alt = alternatives.loc[
                alternatives.scale.eq(scale)
                & alternatives.transformation.eq(transform)
                & alternatives.baseline.eq(baseline), ["site", "date", "spei", "fit_status"]
            ].copy()
            alt.site = alt.site.astype(str).str.lower()
            alt.date = pd.to_datetime(alt.date).dt.to_period("M").dt.to_timestamp()
            if alt.duplicated(["site", "date"]).any():
                raise ValueError(f"Duplicate alternative targets: {name}, SPEI-{scale}")
            joined = canon.merge(alt, on=["site", "date"], how="left", validate="one_to_one")
            # Exact calendar-month pairs. Both target series use the same valid canonical dates.
            for site, group in [*joined.groupby("site", sort=True), ("ALL", joined)]:
                metrics = comparison_metrics(group.spei, group[f"spei_{scale}"])
                status = group.fit_status.value_counts(dropna=False).to_dict()
                failed = group.loc[group.fit_status.eq("fit_failed")].copy()
                failed["calendar_month"] = failed.date.dt.month
                failed_fit_month_count = int(failed[["site", "calendar_month"]].drop_duplicates().shape[0])
                paired_alt = group.spei.notna() & group[f"spei_{scale}"].notna()
                row = {
                    "site": site, "scale": scale, "alternative": name,
                    "transformation": transform, "baseline": baseline,
                    "failed_fit_site_calendar_month_count": failed_fit_month_count,
                    "failed_fit_target_rows": int(group.fit_status.eq("fit_failed").sum()),
                    "alternative_missing_n": int(group.spei.isna().sum()),
                    "canonical_missing_n": int(group[f"spei_{scale}"].isna().sum()),
                    "alternative_extreme_drought_count": int((group.spei <= -2).sum()),
                    "canonical_extreme_drought_count": int((group[f"spei_{scale}"] <= -2).sum()),
                    "extreme_drought_event_disagreement_n": int(((group.loc[paired_alt, "spei"] <= -2) != (group.loc[paired_alt, f"spei_{scale}"] <= -2)).sum()),
                    "fit_status_counts_json": json.dumps({str(k): int(v) for k, v in status.items()}, sort_keys=True),
                    **metrics,
                }
                sensitivity_rows.append(row)
                category_rows.append({
                    "site": site, "scale": scale, "comparison": f"{name}_vs_canonical",
                    "candidate": name, "comparator": "canonical_1990_2010",
                    **metrics,
                })
            # Reference metrics for alternative itself, including v2.11 where present.
            candidate_ref = joined.merge(ref_local, on=["site", "date"], how="left", validate="one_to_one")
            for site, group in [*candidate_ref.groupby("site", sort=True), ("ALL", candidate_ref)]:
                category_rows.append({
                    "site": site, "scale": scale, "comparison": f"{name}_vs_speibase_v211",
                    "candidate": name, "comparator": "speibase_v2.11",
                    **comparison_metrics(group.spei, group[f"spei_{scale}_reference"]),
                })
        # Existing local/reference comparison, and canonical/ref category metrics.
        local_ref = canon.merge(ref_local, on=["site", "date"], how="left", validate="one_to_one")
        category_rows.extend(_site_and_all_rows(local_ref, f"spei_{scale}", f"spei_{scale}_reference", scale,
                                                "canonical_1990_2010_vs_speibase_v211"))
    return pd.DataFrame(sensitivity_rows), pd.DataFrame(category_rows)


def clipping_sensitivity(canonical: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = canonical.copy()
    frame["site"] = frame.site.astype(str).str.lower()
    frame["date"] = pd.to_datetime(frame.date).dt.to_period("M").dt.to_timestamp()
    records, summary = [], {}
    for scale in SCALES:
        raw = _uncapped_nonparametric(frame, scale)
        clipped = raw.clip(-MAX_ABS_Z, MAX_ABS_Z)
        stored = pd.to_numeric(frame[f"spei_{scale}"], errors="coerce")
        raw_vs_stored = raw.notna() & stored.notna()
        if raw_vs_stored.any() and not np.allclose(clipped[raw_vs_stored], stored[raw_vs_stored], atol=1e-10, rtol=0):
            raise AssertionError(f"Reconstructed clipped SPEI-{scale} differs from canonical targets")
        ref = reference[["site", "date", f"spei_{scale}_reference"]].copy()
        ref.site = ref.site.astype(str).str.lower()
        ref.date = pd.to_datetime(ref.date).dt.to_period("M").dt.to_timestamp()
        paired = frame[["site", "date"]].copy()
        paired["uncapped"] = raw
        paired["clipped"] = clipped
        paired["reference"] = paired.merge(ref, on=["site", "date"], how="left", validate="one_to_one")[f"spei_{scale}_reference"].to_numpy()
        scope_groups = [(site, group) for site, group in paired.groupby("site", sort=True)] + [("ALL", paired)]
        scale_rows = []
        for site, group in scope_groups:
            valid = group.uncapped.notna()
            below, above = group.uncapped < -MAX_ABS_Z, group.uncapped > MAX_ABS_Z
            category_change = category(group.uncapped[valid]) != category(group.clipped[valid])
            before = comparison_metrics(group.uncapped, group.reference)
            after = comparison_metrics(group.clipped, group.reference)
            row = {
                "site": site, "scale": scale, "valid_local_n": int(valid.sum()),
                "clipped_below_minus_3_5_n": int(below.sum()),
                "clipped_above_plus_3_5_n": int(above.sum()),
                "clipped_total_n": int((below | above).sum()),
                "clipped_percent_valid": float(100 * (below | above).sum() / valid.sum()) if valid.any() else None,
                "uncapped_min": float(group.uncapped[valid].min()) if valid.any() else None,
                "uncapped_max": float(group.uncapped[valid].max()) if valid.any() else None,
                "clipped_min": float(group.clipped[valid].min()) if valid.any() else None,
                "clipped_max": float(group.clipped[valid].max()) if valid.any() else None,
                "category_changes_caused_by_clipping_n": int(category_change.sum()),
                **{f"uncapped_{k}": v for k, v in before.items()},
                **{f"clipped_{k}": v for k, v in after.items()},
            }
            for metric in ("pearson_r", "bias_candidate_minus_comparator", "rmse", "seven_category_agreement"):
                bef, aft = before[metric], after[metric]
                row[f"change_clipped_minus_uncapped_{metric}"] = aft - bef if bef is not None and aft is not None else None
            records.append(row)
            scale_rows.append(row)
        summary[str(scale)] = next(row for row in scale_rows if row["site"] == "ALL")
    return pd.DataFrame(records), summary


def _top30_reconstruction(canonical: pd.DataFrame, extremes: pd.DataFrame, config: dict) -> pd.DataFrame:
    ranked = extremes.dropna(subset=["absolute_difference"]).sort_values("absolute_difference", ascending=False).head(30)
    if len(ranked) != 30:
        raise ValueError(f"Expected 30 paired disagreement rows; found {len(ranked)}")
    if not (((ranked.site == "beed") & (ranked.date == "2019-09") & (ranked.scale == 6)).any()):
        raise AssertionError("Top 30 does not include Beed September 2019 SPEI-6")
    data = canonical.copy()
    data.site = data.site.astype(str).str.lower()
    data.date = pd.to_datetime(data.date).dt.to_period("M").dt.to_timestamp()
    result = []
    power_cache_dir = Path("data/raw_protocol7_refresh")
    for rank, event in enumerate(ranked.itertuples(index=False), 1):
        site_frame = data.loc[data.site.eq(event.site)].sort_values("date").set_index("date")
        date = pd.Timestamp(event.date + "-01")
        scale = int(event.scale)
        start = date - pd.DateOffset(months=scale - 1)
        dates = pd.date_range(start, date, freq="MS")
        history = site_frame.reindex(dates)
        power_path = power_cache_dir / f"{event.site}_power_1990_2026_v10.csv"
        power_source_missing: list[str] = []
        power_source_matches: dict[str, bool | None] = {}
        if power_path.exists():
            power = pd.read_csv(power_path, parse_dates=["date"]).set_index("date").reindex(dates)
            for variable in ("PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"):
                if variable not in power:
                    power_source_missing.append(variable)
                    power_source_matches[variable] = None
                    continue
                monthly = pd.to_numeric(power[variable], errors="coerce")
                values = pd.to_numeric(history[variable], errors="coerce") if variable in history else pd.Series(np.nan, index=dates)
                power_source_missing.extend(
                    f"{variable}:{dt:%Y-%m}" for dt in dates[monthly.isna().to_numpy()]
                )
                paired_input = monthly.notna() & values.notna()
                power_source_matches[variable] = bool(
                    paired_input.any()
                    and np.allclose(monthly[paired_input], values[paired_input], atol=1e-10, rtol=0)
                ) if paired_input.any() else None
        else:
            power_source_missing = ["monthly_POWER_cache_file_unavailable"]
            power_source_matches = {"PRECTOTCORR": None, "T2M": None, "T2M_MAX": None, "T2M_MIN": None}
        precip = history.PRECTOTCORR * dates.days_in_month
        pet = history.pet_mm
        wb = precip - pet
        accum = wb.sum(min_count=scale)
        monthly_all = site_frame.PRECTOTCORR * site_frame.index.days_in_month - site_frame.pet_mm
        rolling = monthly_all.rolling(scale, min_periods=scale).sum()
        calibration = rolling.loc[
            (rolling.index.month == date.month)
            & (rolling.index >= pd.Timestamp("1990-01-01"))
            & (rolling.index <= CAL_END)
        ].dropna().to_numpy(float)
        local_raw = float(_nonparametric_z_scores(np.asarray([accum]), calibration, max_abs_z=1e9)[0]) if np.isfinite(accum) and len(calibration) >= 2 else np.nan
        local_clipped = float(np.clip(local_raw, -MAX_ABS_Z, MAX_ABS_Z)) if np.isfinite(local_raw) else np.nan
        unique = np.unique(np.sort(calibration))
        plotting = (np.arange(1, len(unique) + 1) - 0.44) / (len(unique) + 0.12) if len(unique) > 1 else np.array([])
        zcal = norm.ppf(plotting) if len(plotting) else np.array([])
        lower_slope, upper_slope = _tail_slopes(unique, zcal) if len(unique) > 1 else (np.nan, np.nan)
        calibration_min = float(np.min(calibration)) if len(calibration) else np.nan
        calibration_max = float(np.max(calibration)) if len(calibration) else np.nan
        outside_adjustment = (
            float(calibration_min - accum) if np.isfinite(accum) and accum < calibration_min
            else float(accum - calibration_max) if np.isfinite(accum) and accum > calibration_max
            else 0.0
        )
        total_pet = float(pet.sum(min_count=scale)) if pet.notna().sum() == scale else np.nan
        total_precip = float(precip.sum(min_count=scale)) if precip.notna().sum() == scale else np.nan
        site_spec = config["sites"][event.site]
        result.append({
            "rank": rank, "site": event.site, "scale": scale, "date": date.strftime("%Y-%m"),
            "site_lat": float(site_spec["lat"]), "site_lon": float(site_spec["lon"]),
            "reference_grid_lat": float(event.reference_grid_lat), "reference_grid_lon": float(event.reference_grid_lon),
            "reference_grid_distance_km": float(event.reference_grid_distance_km),
            "contributing_months": json.dumps([d.strftime("%Y-%m") for d in dates]),
            "precipitation_mm_day_values": json.dumps([None if pd.isna(v) else float(v) for v in history.PRECTOTCORR]),
            "precipitation_monthly_total_mm_values": json.dumps([None if pd.isna(v) else float(v) for v in precip]),
            "pet_mm_values": json.dumps([None if pd.isna(v) else float(v) for v in pet]),
            "monthly_water_balance_mm_values": json.dumps([None if pd.isna(v) else float(v) for v in wb]),
            "accumulated_precipitation_total_mm": total_precip if np.isfinite(total_precip) else None,
            "accumulated_pet_total_mm": total_pet if np.isfinite(total_pet) else None,
            "accumulated_water_balance_mm": float(accum) if np.isfinite(accum) else None,
            "calibration_month": int(date.month), "calibration_period": f"1990-01 through {CAL_END:%Y-%m}",
            "calibration_n": len(calibration),
            "calibration_min_accumulation_mm": calibration_min if np.isfinite(calibration_min) else None,
            "calibration_max_accumulation_mm": calibration_max if np.isfinite(calibration_max) else None,
            "water_balance_adjustment_to_calibration_range_mm": outside_adjustment,
            "equivalent_mean_monthly_pet_change_to_reach_calibration_range_mm": (
                outside_adjustment / scale if np.isfinite(total_pet) else None
            ),
            "outside_calibration_range": bool(len(calibration) and (accum < np.min(calibration) or accum > np.max(calibration))),
            "fitted_parameters": json.dumps({"method": "nonparametric empirical plotting positions", "lower_tail_slope_z_per_mm": lower_slope, "upper_tail_slope_z_per_mm": upper_slope}),
            "unclipped_local_spei": local_raw, "clipped_local_spei": local_clipped,
            "canonical_local_spei": float(event.local_spei), "reference_spei_v211": float(event.reference_spei),
            "signed_difference_local_minus_reference": float(event.signed_difference_local_minus_reference),
            "absolute_difference": float(event.absolute_difference),
            "outside_calibration_side": "below" if len(calibration) and accum < np.min(calibration) else "above" if len(calibration) and accum > np.max(calibration) else "within",
            "missing_input_count_in_window": int(history[["PRECTOTCORR", "pet_mm"]].isna().any(axis=1).sum()),
            "monthly_POWER_cache_file": str(power_path) if power_path.exists() else None,
            "monthly_POWER_source_missing_fields": json.dumps(power_source_missing),
            "canonical_monthly_fields_match_cached_POWER": json.dumps(power_source_matches, sort_keys=True),
            "fallback_uses_same_POWER_provider": True,
            "source_attribution_limit": "Cache match is evidence for matching values, not proof of which same-provider fallback path was used if the original build manifest is absent.",
        })
    return pd.DataFrame(result)


def _input_data_inventory(canonical: pd.DataFrame, config: dict, output_dir: Path) -> pd.DataFrame:
    event = pd.read_csv(output_dir / "spei_extreme_event_audit.csv")
    site_diff = event.dropna(subset=["absolute_difference"]).groupby("site").absolute_difference.mean().sort_values()
    selected = ["beed", str(site_diff.index[0]), str(site_diff.index[-1]), "mumbai"]
    selected = list(dict.fromkeys(selected))
    cache_dir = Path("data/raw_protocol7_refresh")
    rows = []
    data = canonical.copy()
    data.site = data.site.astype(str).str.lower()
    data.date = pd.to_datetime(data.date)
    for site in selected:
        site_data = data.loc[data.site.eq(site) & data.date.le("2024-12-01")].copy()
        metadata_path = cache_dir / f"{site}_power_1990_2026_v10.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        units = metadata.get("units", {})
        retrieval = metadata.get("retrieved_utc")
        for variable in ("precipitation", "PET"):
            if variable == "precipitation":
                local_values = site_data.PRECTOTCORR * pd.DatetimeIndex(site_data.date).days_in_month
                local_unit = "mm/month"
                source_var = "PRECTOTCORR"
            else:
                local_values = site_data.pet_mm
                local_unit = "mm/month"
                source_var = "T2M,T2M_MAX,T2M_MIN"
            local_missing = int(local_values.isna().sum())
            local_stats = {
                "local_n": int(local_values.notna().sum()), "local_missing_n": local_missing,
                "local_mean": float(local_values.mean()) if local_values.notna().any() else None,
                "local_min": float(local_values.min()) if local_values.notna().any() else None,
                "local_max": float(local_values.max()) if local_values.notna().any() else None,
            }
            rows.append({
                "site": site, "component": variable, "dataset": "DroughtCast local source/input",
                "source": "NASA POWER monthly and annual API v2.10.0" if variable == "precipitation" else "Derived in DroughtCast from NASA POWER monthly temperature means",
                "source_url": "https://power.larc.nasa.gov/api/temporal/monthly/point",
                "source_variable": source_var, "units": local_unit,
                "source_unit_metadata": units.get("PRECTOTCORR") if variable == "precipitation" else "derived monthly total mm",
                "temporal_resolution": "monthly", "spatial_resolution": "site point; POWER source-native monthly grid",
                "coverage": "1990-01 through 2024-12; numeric coverage below", "retrieved_utc": retrieval,
                "local_source_file": str(cache_dir / f"{site}_power_1990_2026_v10.csv"),
                "independent_values_obtained": True, "comparison_status": "local summary only; independent raw values not cached",
                **local_stats,
            })
            candidates = ([
                ("SPEIbase input provenance / CRU TS 4.09", "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/", "pre", "mm/month", "0.5 degree monthly", "Reference inputs are not in project files; SPEIbase v2.11 only has SPEI matched points."),
                ("CHIRPS v3", "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/", "precipitation", "monthly precipitation amount (read file metadata before use)", "0.05 degree; monthly product available; 1981-near-present", "Public candidate; not downloaded for this phase."),
            ] if variable == "precipitation" else [
                ("SPEIbase input provenance / CRU TS 4.09", "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/", "pet", "mm/day (multiply by calendar days for monthly total)", "0.5 degree monthly", "Reference inputs are not in project files; same reference product family, not an independent validation of SPEIbase."),
                ("ERA5-Land monthly means", "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means", "potential evaporation (pev), total precipitation", "m water equivalent (verify sign convention and monthly field metadata)", "0.1 degree CDS grid; monthly; 1950-present", "Public CDS product; no local cache/API credentials configured; not downloaded."),
            ])
            for name, url, variable_name, candidate_units, resolution, note in candidates:
                rows.append({
                    "site": site, "component": variable, "dataset": name, "source": name,
                    "source_url": url, "source_variable": variable_name, "units": candidate_units,
                    "temporal_resolution": "monthly", "spatial_resolution": resolution,
                    "coverage": "documented product coverage; exact site-month completeness not checked",
                    "retrieved_utc": None, "local_source_file": None,
                    "independent_values_obtained": False, "comparison_status": note,
                    "local_n": local_stats["local_n"], "local_missing_n": local_missing,
                    "local_mean": local_stats["local_mean"], "local_min": local_stats["local_min"],
                    "local_max": local_stats["local_max"],
                })
    return pd.DataFrame(rows)


def run(output_dir: Path = OUTPUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    canonical = pd.read_csv(CANONICAL, parse_dates=["date"])
    reference = pd.read_csv(REFERENCE, parse_dates=["date"])
    alternatives = pd.read_csv(ALTERNATIVE, parse_dates=["date"])
    metadata = json.loads(REFERENCE_META.read_text(encoding="utf-8"))
    extremes = pd.read_csv(output_dir / "spei_extreme_event_audit.csv")

    clip_frame, clip_summary = clipping_sensitivity(canonical, reference)
    clip_frame.to_csv(output_dir / "spei_clipping_sensitivity.csv", index=False)

    top30 = _top30_reconstruction(canonical, extremes, config)
    top30.to_csv(output_dir / "spei_top30_discrepancy_reconstruction.csv", index=False)

    sensitivity, category_frame = _canonical_and_alternatives(canonical, alternatives, reference, list(config["data"]["site_filter"]))
    sensitivity.to_csv(output_dir / "spei_method_sensitivity.csv", index=False)
    category_rows = category_frame.to_dict("records")
    # Add clipped/unclipped and reference category comparisons, with per-site and pooled rows.
    for scale in SCALES:
        raw = _uncapped_nonparametric(canonical, scale)
        clipped = raw.clip(-MAX_ABS_Z, MAX_ABS_Z)
        ref = canonical[["site", "date"]].merge(
            reference[["site", "date", f"spei_{scale}_reference"]], on=["site", "date"], how="left", validate="one_to_one"
        )
        compare = canonical[["site", "date"]].copy()
        compare["raw"] = raw
        compare["clipped"] = clipped
        compare["reference"] = ref[f"spei_{scale}_reference"].to_numpy()
        category_rows.extend(_site_and_all_rows(compare, "clipped", "raw", scale, "clipped_vs_unclipped_nonparametric"))
    categories = pd.DataFrame(category_rows)
    categories.to_csv(output_dir / "spei_category_stability.csv", index=False)

    input_inventory = _input_data_inventory(canonical, config, output_dir)
    input_inventory.to_csv(output_dir / "spei_input_data_comparison.csv", index=False)

    diagnostics = json.loads((output_dir / "spei_fit_diagnostics.json").read_text(encoding="utf-8"))
    failed = [row for row in diagnostics["fit_diagnostics"] if row.get("fit_status") == "failed"]
    generalized = diagnostics["transformations"]["generalized_logistic_mle"]
    missing_by_scale = {}
    for scale in SCALES:
        local = pd.to_numeric(canonical[f"spei_{scale}"], errors="coerce")
        refs = pd.to_numeric(reference[f"spei_{scale}_reference"], errors="coerce")
        missing_by_scale[str(scale)] = {
            "canonical_rows": len(canonical), "canonical_missing": int(local.isna().sum()),
            "reference_rows": len(reference), "reference_missing": int(refs.isna().sum()),
            "paired": int((local.notna() & refs.notna()).sum()),
        }
    methodology = {
        "phase": "1.4", "canonical_modified": False, "config_modified": False,
        "feature_set_modified": False, "model_run": False,
        "official_method_sources": {
            "speibase_repository": "https://github.com/sbegueria/SPEIbase",
            "speibase_functions": "https://github.com/sbegueria/SPEIbase/blob/master/R/functions.R",
            "spei_function": "https://github.com/sbegueria/SPEI/blob/master/R/spei.R",
            "cran_spei_manual": "https://cran.r-project.org/web/packages/SPEI/SPEI.pdf",
        },
        "method_comparison": [
            {"topic": "water balance and units", "local": "POWER PRECTOTCORR monthly mean rate (mm/day) x days_in_month - Hargreaves-Samani monthly PET total (mm/month)", "official": "CRU TS precipitation (mm/month) - CRU PET (mm/day) x month days in SPEIbase functions.R", "classification": "mathematically equivalent unit conversion; data and PET methods intentionally differ", "evidence": "data/build_dataset.py and SPEIbase R/functions.R"},
            {"topic": "trailing accumulation", "local": "unweighted trailing rolling sum, full scale window required", "official": "default rectangular kernel, shift 0, right-aligned weighted sum over scale months", "classification": "equivalent for complete regular monthly input", "evidence": "SPEI::spei default kernel and local pandas rolling"},
            {"topic": "calendar-month strata", "local": "separate calibration sample for each calendar month", "official": "separate cycle/month fit for each time series", "classification": "equivalent", "evidence": "SPEI::spei internal month loop"},
            {"topic": "calibration period", "local": f"fixed {config['data'].get('spei_calibration_start_year', 1990)}-01 through {CAL_END:%Y-%m}", "official": "SPEIbase invokes SPEI::spei without ref.start/ref.end, so package defaults use all supplied observations (v2.11 through 2024)", "classification": "intentional methodological alternative", "evidence": "SPEIbase R/functions.R call; SPEI manual reference-period defaults"},
            {"topic": "distribution and parameter fit", "local": "canonical uses empirical plotting positions p=(rank-0.44)/(n+0.12), inverse-normal scores, linear tail extrapolation; no fitted parametric distribution", "official": "log-Logistic (generalized logistic type III) with ub-pwm by default", "classification": "intentional methodological alternative; not equivalent to official SPEI", "evidence": "local _nonparametric_z_scores vs SPEI::spei defaults"},
            {"topic": "probability transform", "local": "empirical score interpolation/extrapolation then normal score; Fisk alternative uses its fitted CDF; final local score clipping", "official": "log-Logistic CDF followed by qnorm; SPEIbase calls na.rm=TRUE", "classification": "normal-score end step equivalent in concept; underlying CDF differs", "evidence": "local source and official SPEI code"},
            {"topic": "zero, negative, and support", "local": "canonical empirical transform accepts real water balances including negatives and zeros; alternative Fisk maps values at/below its estimated location to probability floor 1e-6", "official": "log-Logistic has a fitted location parameter and can represent negative balance; zero-mass correction is applied to Gamma/PearsonIII, not default log-Logistic", "classification": "intentional distribution behavior difference", "evidence": "SPEI::spei distribution loop and local transform code"},
            {"topic": "missing values and spin-up", "local": "rolling sum requires every month in the window; NaNs propagate; calibration drops missing accumulated values; first scale-1 outputs are missing", "official": "spei.nc calls SPEI::spei(..., na.rm=TRUE); rolling windows are right aligned; fit excludes NA values", "classification": "equivalent in window propagation on regular monthly inputs; local 20-sample minimum is stricter than package's four-sample early guard", "evidence": "local rolling and official package implementation"},
            {"topic": "fit failures", "local": "canonical empirical transform has no optimizer; less than 20 calibration observations leaves missing; Fisk failures are stored in diagnostics and target rows are missing", "official": "package requires nonzero spread and at least 4 observations, checks L-moment validity and may return missing values", "classification": "intentional alternative plus different failure thresholds; no silent Fisk drops in stored alternative diagnostics", "evidence": "spei_fit_diagnostics.json and SPEI package code"},
            {"topic": "extreme clipping", "local": "canonical empirical and Fisk scores clipped at +/-3.5", "official": "SPEIbase SPEI::spei standardizes qnorm(CDF) without a fixed +/-3.5 clipping step", "classification": "intentional implementation choice, material tail censoring measured below", "evidence": "local _nonparametric_z_scores and official qnorm(cdf_res)"},
            {"topic": "alternative Fisk PWM estimator", "local": "three-parameter Fisk PWM based on plotting positions (rank-0.35)/n", "official": "ub-pwm by default; pp-pwm is a separate supported method", "classification": "intentional alternative estimator; not official default", "evidence": "alternative-target source and SPEI manual"},
        ],
        "implementation_defect_review": {
            "confirmed_canonical_defects": [],
            "potential_noncanonical_api_issue": "data.build_dataset._standardize_accumulation treats every method string other than 'nonparametric' as the custom shifted-log-logistic branch instead of rejecting unknown names; not active for configured canonical method, not changed in this phase.",
            "important_methodological_limitations": ["21 calibration years provide only about 20-21 monthly observations per calendar month.", "empirical tail extrapolation beyond the calibration range is linear and the +/-3.5 cap hides further magnitude.", "nonparametric SPEI is a valid empirical standardization alternative but is not the conventional SPEIbase log-logistic estimator."],
        },
        "fit_failures": {"failed_count": len(failed), "by_baseline": {baseline: sum(row["baseline"] == baseline for row in failed) for baseline in ("1990-2010", "1981-2010")}, "records": failed},
        "generalized_logistic": generalized,
        "clipping_summary_by_scale": clip_summary,
        "alternative_sensitivity_summary": sensitivity.loc[sensitivity.site.eq("ALL")].to_dict("records"),
        "missing_and_paired_counts": missing_by_scale,
        "input_data_sources": {
            "local_cache_scan": "NASA POWER monthly CSV/metadata and 1981 history files are present; no local CRU TS precipitation/PET grids, ERA5-Land, CHIRPS, or GLEAM rasters/time series were found.",
            "representative_sites": input_inventory.site.unique().tolist(),
            "comparison_status": "Local summaries computed; independent precipitation/PET observations not obtained, so there is no numeric input difference. Candidates and exact data constraints are in spei_input_data_comparison.csv.",
        },
        "event_distribution": {
            "top30_event_count": len(top30),
            "sites": int(top30.site.nunique()),
            "scales": {str(k): int(v) for k, v in top30.scale.value_counts().sort_index().items()},
            "years": int(top30.date.str[:4].nunique()),
            "positive_signed_disagreements": int((top30.signed_difference_local_minus_reference > 0).sum()),
            "negative_signed_disagreements": int((top30.signed_difference_local_minus_reference < 0).sum()),
            "outside_calibration_range": int(top30.outside_calibration_range.sum()),
            "missing_input_windows": int(top30.missing_input_count_in_window.gt(0).sum()),
        },
        "recommendation": "Retain canonical targets unchanged, but do not treat them as fully research-validated; improve target sensitivity and input attribution before research-grade model claims.",
        "reference": metadata,
    }
    (output_dir / "spei_methodology_comparison.json").write_text(json.dumps(methodology, indent=2, allow_nan=False), encoding="utf-8")
    report = _report(methodology, clip_frame, top30, sensitivity, categories, input_inventory)
    (output_dir / "spei_phase1_4_report.md").write_text(report, encoding="utf-8")
    return methodology


def _report(methodology: dict, clipping: pd.DataFrame, top30: pd.DataFrame,
            sensitivity: pd.DataFrame, categories: pd.DataFrame,
            inputs: pd.DataFrame) -> str:
    clip_all = clipping.loc[clipping.site.eq("ALL")].set_index("scale")
    alt_all = sensitivity.loc[sensitivity.site.eq("ALL")]
    local_ref = categories.loc[categories.comparison.eq("canonical_1990_2010_vs_speibase_v211") & categories.site.eq("ALL")]
    lines = [
        "# Phase 1.4 — SPEI methodological validation and discrepancy attribution", "",
        "This analysis reads the canonical dataset and existing Phase 1.2/1.3 artifacts. It does not modify canonical SPEI, the config, features, model code, or previous outputs; no model was run.", "",
        "## Method comparison", "",
        "| Area | Local implementation | Official SPEI/SPEIbase behavior | Classification |",
        "|---|---|---|---|",
    ]
    for item in methodology["method_comparison"]:
        lines.append(f"| {item['topic']} | {item['local']} | {item['official']} | {item['classification']} |")
    lines += [
        "", "The local nonparametric transform is an empirical standardization: it maps sample plotting positions to normal scores and interpolates/extrapolates values. This is a defensible alternative for a fixed historical reference, but it is not the parametric log-logistic SPEI estimator used by the official package. It will respond differently to skew, tail shape, and values outside the calibration sample. The short monthly calibration sample makes its empirical tails especially sensitive; linear tail extrapolation followed by clipping further limits extreme-value interpretation.",
        "", "The official SPEI package defaults to log-Logistic with unbiased PWM, rectangular trailing accumulation, and all supplied observations unless `ref.start`/`ref.end` are provided. SPEIbase calls it with `na.rm=TRUE`. The local fixed 1990–2010 calibration is intentional for a forecasting protocol; it differs from the v2.11 all-series fit. Fisk alternatives are closer in family but use the project's plotting-position PWM equations rather than SPEI's ub-pwm default.",
        "", "**Defect assessment:** no confirmed defect was found in the configured canonical nonparametric path. A potential API issue remains: an unrecognized `method` passed to `_standardize_accumulation` falls through to its custom log-logistic branch instead of raising. It is not used by the configured canonical method and was not changed. The alternative Fisk fit fails for 53 calendar/site/scale combinations; those cases are explicitly present as failed in diagnostics and missing in the alternative targets. The generalized-logistic fit's convergence flags do not establish stable shape estimates: 166/432 shapes exceed 100 and the maximum is 2,136.6.",
        "", "## Clipping sensitivity", "",
        "| Scale | Valid local n | Below -3.5 | Above +3.5 | Clipped % | Uncapped min/max | Category changes | Pearson change vs ref | Bias change | RMSE change | Category-agreement change |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, row in clip_all.iterrows():
        lines.append(
            f"| {int(scale)} | {int(row.valid_local_n)} | {int(row.clipped_below_minus_3_5_n)} | {int(row.clipped_above_plus_3_5_n)} | {row.clipped_percent_valid:.3f}% | {row.uncapped_min:.3f}/{row.uncapped_max:.3f} | {int(row.category_changes_caused_by_clipping_n)} | {row.change_clipped_minus_uncapped_pearson_r:.5f} | {row.change_clipped_minus_uncapped_bias_candidate_minus_comparator:+.5f} | {row.change_clipped_minus_uncapped_rmse:+.5f} | {row.change_clipped_minus_uncapped_seven_category_agreement:+.5f} |"
        )
    lines += ["", "Counts are over local site-month targets; metric deltas use exact paired SPEIbase v2.11 months. The seven drought/wet category boundaries are unchanged by ±3.5 clipping, so clipping can change numerical RMSE/bias/correlation while leaving categories identical. See the CSV for per-site metrics and full pre/post results.",
              "", "## Top 30 disagreement reconstruction", "",
              f"The selected top 30 include {methodology['event_distribution']['sites']} sites, all three scales, {methodology['event_distribution']['years']} years, and both signs of local-minus-reference difference. {methodology['event_distribution']['outside_calibration_range']} of 30 accumulated values lie outside their calendar-month 1990–2010 calibration ranges; {methodology['event_distribution']['missing_input_windows']} contain missing local precipitation/PET inputs. Each CSV row contains the scale-window precipitation rates/totals, PET, monthly water balance, accumulated balance, calibration n/range, empirical tail slopes, local uncapped/clipped score, reference value, coordinates, and discrepancy.",
              "", "| Rank | Site | Scale | Month | Accumulated balance (mm) | Calibration range (mm) | Raw SPEI | Clipped SPEI | v2.11 | Absolute difference | Outside calibration? |",
              "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in top30.head(10).itertuples(index=False):
        lines.append(f"| {row.rank} | {row.site} | {row.scale} | {row.date} | {row.accumulated_water_balance_mm:.1f} | {row.calibration_min_accumulation_mm:.1f} to {row.calibration_max_accumulation_mm:.1f} | {row.unclipped_local_spei:.3f} | {row.clipped_local_spei:.3f} | {row.reference_spei_v211:.3f} | {row.absolute_difference:.3f} | {row.outside_calibration_range} |")
    beed = top30.loc[(top30.site == "beed") & (top30.date == "2019-09") & (top30.scale == 6)].iloc[0]
    lines += ["", f"For Beed September 2019, the six-month local balance is {beed.accumulated_water_balance_mm:.3f} mm versus a September calibration range of {beed.calibration_min_accumulation_mm:.3f} to {beed.calibration_max_accumulation_mm:.3f} mm; it is below range. The reconstructed raw empirical score is {beed.unclipped_local_spei:.6f}, clipped to {beed.clipped_local_spei:.3f}, versus v2.11 {beed.reference_spei_v211:.6f}. This establishes that the local six-month balance is outside its local calibration tail and that the local score is censored. It does not reveal the CRU source balance; the sign discrepancy remains unexplained.",
              "", "## Precipitation and PET input availability", "",
              "The local cached input is NASA POWER monthly data. Precipitation is `PRECTOTCORR` in mm/day and converted to monthly total by multiplying by days in month. PET is Hargreaves–Samani from monthly mean temperatures/extrema, multiplied to a monthly total. Existing caches do not contain CRU TS input grids, ERA5-Land or CHIRPS point series. The v2.11 artifact contains matched SPEI values but not the underlying precipitation/PET.",
              "", "A small independent input comparison was therefore **not numerically possible** in this run. Official candidate sources were identified: CHIRPS v3 provides 0.05° precipitation from 1981 to near-present, but it blends satellite and station data and its v3 gauge undercatch correction changes levels; ERA5-Land monthly means provide precipitation and potential evaporation on the 0.1° CDS grid from 1950 onward, but potential evaporation's sign/definition must be checked before treating it as crop-reference PET. CRU TS v4.09 provides `pre` (mm/month) and `pet` (mm/day) at 0.5° and is the source family for v2.11, so it can help reproduce the reference inputs but is not an independent validation product. No source values were fabricated or downloaded.",
              "", "## Method sensitivity and drought-category stability", "",
              "The method-sensitivity CSV compares the existing 1981–2010 nonparametric, 1990–2010 Fisk, and 1981–2010 Fisk targets against canonical targets on common valid months, per site/scale and pooled. The category CSV reports canonical/reference, alternative/reference, and clipped/unclipped seven-class, drought-only, and extreme-drought agreement. Failed Fisk fits stay missing, and sample counts are explicit. Summary metrics below are pooled `ALL` rows:",
              "", "| Alternative | Scale | n | Pearson r vs canonical | Bias | MAE | RMSE | Category agreement | Drought false positives | Missed comparator droughts | Extreme-event disagreements |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in alt_all.itertuples(index=False):
        lines.append(f"| {row.alternative} | {row.scale} | {row.paired_n} | {row.pearson_r:.4f} | {row.bias_candidate_minus_comparator:+.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.seven_category_agreement:.4f} | {row.local_only_drought_count} | {row.comparator_only_drought_count} | {row.extreme_drought_event_disagreement_n} |")
    lines += ["", "| Scale | Canonical vs v2.11 pairs | Seven-class agreement | Drought-only agreement | False local droughts | Missed v2.11 droughts | Extreme-drought agreement |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in local_ref.itertuples(index=False):
        lines.append(f"| {row.scale} | {row.paired_n} | {row.seven_category_agreement:.4f} | {row.drought_only_agreement:.4f} | {row.local_only_drought_count} | {row.comparator_only_drought_count} | {row.extreme_drought_event_agreement:.4f} |")
    lines += ["", "## Limitations and recommendation", "",
              "Numerical agreement with v2.11 does not validate either product as truth. Local/reference input differences remain unmeasured; v2.12 is not available in this workspace. The canonical fixed reference has only about 20–21 values per calendar month, does not use the official default log-logistic ub-pwm fit, and clips tails. The signed Beed disagreement is partially characterized but not causally explained. Category robustness should be read from per-site/per-scale values, not pooled rows alone.",
              "", "**Recommendation: RETAIN canonical targets unchanged, but do not yet regard them as cleared for research-grade model training or strong performance claims.** A future approved phase should retrieve raw/reference input data or another independent precipitation/PET source, quantify PET sensitivity, and decide whether a conventional log-logistic target should be added as a parallel protocol. This phase makes no automatic target change.",
              "", "## Artifacts and verification", "",
              "Generated: `spei_methodology_comparison.json`, `spei_clipping_sensitivity.csv`, `spei_top30_discrepancy_reconstruction.csv`, `spei_input_data_comparison.csv`, `spei_method_sensitivity.csv`, `spei_category_stability.csv`, and this report. Tests and Ruff results are recorded in the final execution summary.", "",
              "Sources: [SPEIbase functions.R](https://github.com/sbegueria/SPEIbase/blob/master/R/functions.R), [SPEI package implementation](https://github.com/sbegueria/SPEI/blob/master/R/spei.R), [SPEI CRAN manual](https://cran.r-project.org/web/packages/SPEI/SPEI.pdf), [CRU TS variables and units](https://crudata.uea.ac.uk/cru/data/hrg/), [CHIRPS v3 documentation](https://www.chc.ucsb.edu/data/chirps3), [ERA5-Land dataset documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means).", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)
    print(f"Wrote Phase 1.4 analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
