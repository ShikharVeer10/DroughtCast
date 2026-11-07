"""Compare Maharashtra local SPEI targets with matched SPEIbase v2.11 values.

The script uses exact site/month joins and never fills missing reference months.
It does not modify the canonical dataset or run a forecasting model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

SCALES = (3, 6, 12)
THRESHOLDS = (-1.0, -1.5, -2.0)
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between a site and a reference-cell center."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_grid_cell(lat: float, lon: float, cells: list[tuple[float, float]]) -> tuple[float, float]:
    """Return nearest (latitude, longitude) cell center using great-circle distance."""
    if not cells:
        raise ValueError("At least one reference grid cell is required")
    return min(cells, key=lambda cell: haversine_km(lat, lon, *cell))


def align_monthly(local: pd.DataFrame, reference: pd.DataFrame, local_column: str, reference_column: str) -> pd.DataFrame:
    """Exact site/month alignment; missing values remain missing and are counted."""
    keys = ["site", "date"]
    left = local[keys + [local_column]].copy()
    right = reference[keys + [reference_column]].copy()
    for frame in (left, right):
        frame["site"] = frame["site"].astype(str).str.lower()
        frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate site/month rows prevent unambiguous alignment")
    return left.merge(right, on=keys, how="outer", validate="one_to_one").sort_values(keys).reset_index(drop=True)


def drought_category(values: np.ndarray) -> np.ndarray:
    """Map SPEI to no, moderate, severe, extreme drought; NaNs remain -1."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    out[valid] = 0
    out[valid & (values <= -1.0) & (values > -1.5)] = 1
    out[valid & (values <= -1.5) & (values > -2.0)] = 2
    out[valid & (values <= -2.0)] = 3
    return out


def _episodes(frame: pd.DataFrame, column: str, threshold: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous threshold events; gaps in paired months terminate an episode."""
    ordered = frame.sort_values("date")
    events: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = end = None
    previous = None
    for row in ordered.itertuples(index=False):
        date, value = row.date, getattr(row, column)
        if previous is not None and date != previous + pd.offsets.MonthBegin(1):
            if start is not None:
                events.append((start, end))
            start = end = None
        if pd.notna(value) and value <= threshold:
            if start is None:
                start = date
            end = date
        elif start is not None:
            events.append((start, end))
            start = end = None
        previous = date
    if start is not None:
        events.append((start, end))
    return events


def _event_agreement(paired: pd.DataFrame, local_col: str, ref_col: str, threshold: float, tolerance_months: int = 1) -> dict:
    local_events = _episodes(paired[["date", local_col]], local_col, threshold)
    reference_events = _episodes(paired[["date", ref_col]], ref_col, threshold)
    unused = set(range(len(reference_events)))
    onset_errors, termination_errors = [], []
    for onset, termination in local_events:
        candidates = sorted(unused, key=lambda i: abs((reference_events[i][0].year - onset.year) * 12 + reference_events[i][0].month - onset.month))
        if not candidates:
            continue
        idx = candidates[0]
        ref_onset, ref_termination = reference_events[idx]
        onset_error = (onset.year - ref_onset.year) * 12 + onset.month - ref_onset.month
        if abs(onset_error) <= tolerance_months:
            unused.remove(idx)
            onset_errors.append(onset_error)
            termination_errors.append((termination.year - ref_termination.year) * 12 + termination.month - ref_termination.month)
    return {
        "local_event_count": len(local_events), "reference_event_count": len(reference_events),
        "matched_events_within_tolerance": len(onset_errors), "onset_tolerance_months": tolerance_months,
        "mean_absolute_onset_error_months": float(np.mean(np.abs(onset_errors))) if onset_errors else None,
        "mean_absolute_termination_error_months": float(np.mean(np.abs(termination_errors))) if termination_errors else None,
    }


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = fn(x, y).statistic
    return float(value) if np.isfinite(value) else None


def compare_pair(site: str, scale: int, target_name: str, aligned: pd.DataFrame, site_info: dict, cell: dict, period_start: str, period_end: str) -> tuple[dict, dict]:
    local_col, ref_col = f"spei_{scale}_local", f"spei_{scale}_reference"
    frame = aligned[(aligned.site == site)].copy()
    frame = frame[(frame.date >= pd.Timestamp(period_start)) & (frame.date <= pd.Timestamp(period_end))]
    valid = frame[local_col].notna() & frame[ref_col].notna()
    paired = frame.loc[valid].copy()
    x = paired[local_col].to_numpy(dtype=float)
    y = paired[ref_col].to_numpy(dtype=float)
    diff = x - y
    local_categories, ref_categories = drought_category(x), drought_category(y)
    valid_categories = (local_categories >= 0) & (ref_categories >= 0)
    category_agreement = float(np.mean(local_categories[valid_categories] == ref_categories[valid_categories])) if valid_categories.any() else None
    confusion = np.zeros((4, 4), dtype=int)
    if valid_categories.any():
        np.add.at(confusion, (ref_categories[valid_categories], local_categories[valid_categories]), 1)
    metrics = {
        "site": site, "scale": scale, "target": target_name, "n_pairs": len(paired),
        "period_start": period_start, "period_end": period_end,
        "expected_months": len(pd.date_range(period_start, period_end, freq="MS")),
        "local_missing_months": int(frame[local_col].isna().sum()), "reference_missing_months": int(frame[ref_col].isna().sum()),
        "overlap_start": paired.date.min().strftime("%Y-%m") if len(paired) else None,
        "overlap_end": paired.date.max().strftime("%Y-%m") if len(paired) else None,
        "pearson_r": _safe_corr(pearsonr, x, y), "spearman_r": _safe_corr(spearmanr, x, y),
        "mean_bias_local_minus_reference": float(np.mean(diff)) if len(diff) else None,
        "mae": float(np.mean(np.abs(diff))) if len(diff) else None,
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if len(diff) else None,
        "std_ratio_local_over_reference": float(np.std(x, ddof=1) / np.std(y, ddof=1)) if len(x) > 1 and np.std(y, ddof=1) else None,
        "category_agreement_4class": category_agreement,
        "category_thresholds": "none > -1; moderate <= -1; severe <= -1.5; extreme <= -2",
        "site_lat": site_info["lat"], "site_lon": site_info["lon"],
        "reference_lat": cell["reference_lat"], "reference_lon": cell["reference_lon"],
        "grid_distance_km": haversine_km(site_info["lat"], site_info["lon"], cell["reference_lat"], cell["reference_lon"]),
    }
    category_detail = {
        "four_class_confusion_matrix_reference_rows_local_columns": confusion.tolist(),
        "four_class_labels": ["no_drought", "moderate", "severe", "extreme"],
    }
    for threshold in THRESHOLDS:
        local_event, ref_event = x <= threshold, y <= threshold
        tp = int(np.sum(local_event & ref_event)); fp = int(np.sum(local_event & ~ref_event)); fn = int(np.sum(~local_event & ref_event))
        category_detail[str(threshold)] = {
            "local_event_months": int(local_event.sum()), "reference_event_months": int(ref_event.sum()),
            "both_event_months": tp, "agreement": float(np.mean(local_event == ref_event)) if len(x) else None,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "event_agreement": _event_agreement(paired, local_col, ref_col, threshold),
        }
    metrics["drought_categories_json"] = json.dumps(category_detail, sort_keys=True)
    return metrics, {"site": site, "scale": scale, "target": target_name, "category_metrics": category_detail}


def run(config_path: Path, canonical_path: Path, alternative_path: Path, reference_path: Path, metadata_path: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reference_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    canonical = pd.read_csv(canonical_path, parse_dates=["date"])
    alternative = pd.read_csv(alternative_path, parse_dates=["date"])
    reference = pd.read_csv(reference_path, parse_dates=["date"])
    site_cells = {item["site"]: item for item in reference_metadata["site_reference_cells"]}
    start, end = reference_metadata["comparison_period"]
    all_metrics, details = [], []
    targets = [("canonical_nonparametric_1990_baseline", canonical), ("alternative_nonparametric_1981_2010_baseline", alternative)]
    for site, site_info in cfg["sites"].items():
        for target_name, local in targets:
            local_site = local[local.site.astype(str).str.lower() == site].copy()
            ref_site = reference[reference.site.astype(str).str.lower() == site].copy()
            for scale in SCALES:
                aligned = align_monthly(local_site.rename(columns={f"spei_{scale}": "spei_local"}), ref_site.rename(columns={f"spei_{scale}_reference": "spei_reference"}), "spei_local", "spei_reference")
                aligned = aligned.rename(columns={"spei_local": f"spei_{scale}_local", "spei_reference": f"spei_{scale}_reference"})
                metric, detail = compare_pair(site, scale, target_name, aligned, site_info, site_cells[site], start, end)
                all_metrics.append(metric)
                detail.update({key: value for key, value in metric.items() if key != "drought_categories_json"})
                details.append(detail)
    result = pd.DataFrame(all_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "spei_reference_comparison.csv", index=False)
    report = {
        "reference": {
            **reference_metadata,
            "spatial_resolution": "0.5 degree",
            "monthly_coverage": "1901-01 through 2024-12 for the compared SPEIbase v2.11 product",
            "reference_calibration": "SPEIbase generation code calls SPEI::spei(d, scale) without ref.start/ref.end; SPEI package defaults fit to all supplied series values (1901-2024). Default distribution is log-logistic with unbiased probability-weighted moments.",
            "independent_product_notes": "SPEIbase v2.11 uses CRU TS 4.09 and SPEI package 1.8.1. The generation code computes balance using monthly precipitation minus PET and runs SPEI::spei for each grid time series. Local targets use NASA POWER and Hargreaves-Samani PET with a nonparametric transform, so differences reflect inputs and method, not only standardization.",
            "latest_public_release": {"reference": "SPEIbase v2.12.0", "coverage": "1901-01 through 2025-12", "input": "CRU TS 4.10", "url": "https://github.com/sbegueria/SPEIbase/blob/master/README.md"},
            "reference_source_limitations": ["Only the existing matched-point v2.11 artifact is available locally; raw NetCDF was not present for re-extraction. The newer v2.12.0 product is documented upstream but was not extracted for this run.", "The v2.11 comparison ends at 2024-12; local observations after this date have no reference counterpart.", "Nearest-cell comparison uses a 0.5-degree grid and does not represent within-cell/site-scale variability."],
        },
        "comparison": {"targets": [item[0] for item in targets], "scales": list(SCALES), "thresholds": list(THRESHOLDS), "event_matching_tolerance_months": 1, "missing_month_policy": "exact joins only; no interpolation or filling", "n_metric_rows": len(result), "dated_other_transformations": False, "excluded_transformations": "Fisk/log-logistic and generalized-logistic variants are present only as summary rows in results/maharashtra/phase1/spei_distribution_comparison/site_scale_metrics.csv; timestamped target series are not available, so they cannot be aligned against SPEIbase."},
        "site_scale_results": details,
        "major_drought_periods": _major_period_summary(result, canonical, reference, start, end),
        "canonical_severe_drought_disagreement_by_year": _severe_disagreement_by_year(canonical, reference, start, end),
        "interpretation": "Correlation is descriptive and does not establish target validity. Assess bias, error, variability, drought-category agreement, event timing, input differences, and reference-grid representativeness together.",
    }
    (output_dir / "spei_reference_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "spei_reference_validation.md").write_text(_markdown_report(result, report), encoding="utf-8")
    return result, report


def _major_period_summary(metrics: pd.DataFrame, canonical: pd.DataFrame, reference: pd.DataFrame, start: str, end: str) -> list[dict]:
    summaries = []
    merged = align_monthly(canonical, reference, "spei_3", "spei_3_reference")
    merged = merged[(merged.date >= pd.Timestamp(start)) & (merged.date <= pd.Timestamp(end))]
    for scale in SCALES:
        if scale != 3:
            merged = align_monthly(canonical, reference, f"spei_{scale}", f"spei_{scale}_reference")
            merged = merged[(merged.date >= pd.Timestamp(start)) & (merged.date <= pd.Timestamp(end))]
        lc, rc = f"spei_{scale}", f"spei_{scale}_reference"
        merged["year"] = merged.date.dt.year
        paired = merged[merged[lc].notna() & merged[rc].notna()].copy()
        paired["difference"] = paired[lc] - paired[rc]
        for decade, group in paired.groupby((paired.year // 5) * 5):
            summaries.append({"scale": scale, "period": f"{int(decade)}-{int(decade)+4}", "n": len(group), "mean_bias": float(group.difference.mean()), "rmse": float(np.sqrt(np.mean(group.difference ** 2))), "local_severe_months": int((group[lc] <= -1.5).sum()), "reference_severe_months": int((group[rc] <= -1.5).sum()), "severe_disagreement_months": int(((group[lc] <= -1.5) != (group[rc] <= -1.5)).sum())})
    return summaries


def _severe_disagreement_by_year(canonical: pd.DataFrame, reference: pd.DataFrame, start: str, end: str) -> list[dict]:
    """Summarize severe drought disagreement by year, preserving site-month pairs."""
    summaries = []
    for scale in SCALES:
        local_col, ref_col = f"spei_{scale}", f"spei_{scale}_reference"
        merged = align_monthly(canonical, reference, local_col, ref_col)
        merged = merged[(merged.date >= pd.Timestamp(start)) & (merged.date <= pd.Timestamp(end))]
        merged = merged[merged[local_col].notna() & merged[ref_col].notna()].copy()
        merged["year"] = merged.date.dt.year
        local_event, ref_event = merged[local_col] <= -1.5, merged[ref_col] <= -1.5
        merged["local_only"] = local_event & ~ref_event
        merged["reference_only"] = ref_event & ~local_event
        merged["both"] = local_event & ref_event
        for year, group in merged.groupby("year"):
            summaries.append({"scale": scale, "year": int(year), "paired_site_months": len(group), "local_severe_only": int(group.local_only.sum()), "reference_severe_only": int(group.reference_only.sum()), "both_severe": int(group.both.sum())})
    return summaries


def _markdown_report(metrics: pd.DataFrame, report: dict) -> str:
    period_rows = []
    period_frame = pd.DataFrame(report["major_drought_periods"])
    for scale, group in period_frame.groupby("scale"):
        worst = group.loc[group.rmse.idxmax()]
        period_rows.append(f"SPEI-{scale}: {worst.period} has the highest 5-year pooled site-month RMSE ({worst.rmse:.3f}), bias {worst.mean_bias:+.3f}, and {int(worst.severe_disagreement_months)} severe-category disagreement months.")
    lines = ["# Independent SPEI reference validation", "", "Reference: SPEIbase v2.11 (0.5° monthly; CRU TS 4.09; log-logistic distribution with unbiased PWM; parameters fitted over the supplied 1901–2024 series), compared through December 2024 at nearest grid cells. Local SPEI uses NASA POWER water balance and Hargreaves–Samani PET; the independent product has distinct source inputs and methodology. SPEIbase v2.12.0 (CRU TS 4.10 through 2025) is documented upstream but was not locally extracted for this comparison.", "", "## Site-average comparison", "", "| Target | Scale | Mean Pearson r | Mean Spearman r | Mean bias | Mean MAE | Mean RMSE | Mean std ratio | Mean 4-class agreement | Paired months (range) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (target, scale), group in metrics.groupby(["target", "scale"]):
        lines.append(f"| {target} | SPEI-{scale} | {group.pearson_r.mean():.3f} | {group.spearman_r.mean():.3f} | {group.mean_bias_local_minus_reference.mean():.3f} | {group.mae.mean():.3f} | {group.rmse.mean():.3f} | {group.std_ratio_local_over_reference.mean():.3f} | {group.category_agreement_4class.mean():.3f} | {group.n_pairs.min()}–{group.n_pairs.max()} |")
    lines += ["", "## Drought-period concentration", "", *period_rows, "", "The JSON also reports annual site-month disagreement counts for SPEI ≤ -1.5, and onset/termination agreement for each site, scale, and threshold. These are descriptive because neighboring months, sites, and grid cells are dependent.", "", "## Interpretation and limitations", "", "These scores measure agreement with a gridded reference, not truth. Site-to-cell distances and per-site results are included in the CSV. Drought categories use thresholds -1.0, -1.5, and -2.0; event matching allows onset within one month and is descriptive where event counts are low. Missing months were not interpolated. SPEIbase v2.11 ends in 2024, so 2025–2026 cannot be independently checked from the matched artifact. Dated outputs for Fisk/generalized-logistic variants are unavailable, so they were not compared to the reference. The 1981–2010 alternative changes correlation/error only modestly and does not uniformly improve agreement; exact values are tabulated above.", "", "**Recommendation: RECONSIDER before making a validity claim.** Keep the canonical dataset unchanged for reproducibility, but the v2.11 comparison does not establish local target validity: canonical per-site mean Pearson correlation ranges from 0.686 to 0.786 by scale, mean RMSE from 0.736 to 0.841, and mean moderate-event recall is about 0.50. The product differences (NASA POWER/Hargreaves vs CRU/SPEIbase) and 0.5° grid mismatch prevent treating SPEIbase as ground truth. Repeat the comparison with v2.12 and investigate discrepant events before changing target generation.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--canonical", type=Path, default=Path("data/maharashtra/monthly_sites.csv"))
    parser.add_argument("--alternative", type=Path, default=Path("results/maharashtra/phase1/spei_1981_baseline/monthly_sites.csv"))
    parser.add_argument("--reference", type=Path, default=Path("results/maharashtra/speibase_validation/matched_monthly_points.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("results/maharashtra/speibase_validation/validation_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/maharashtra/phase1"))
    args = parser.parse_args()
    run(args.config, args.canonical, args.alternative, args.reference, args.metadata, args.output_dir)
    print(f"Wrote independent SPEI comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
