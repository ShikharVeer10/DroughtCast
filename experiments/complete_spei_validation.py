"""Run the available independent SPEI comparisons and diagnose 2015-2019."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.validate_spei_reference import (
    SCALES,
    align_monthly,
    compare_pair,
    drought_category,
)


def _reference_from_v211(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.rename(columns={f"spei_{s}_reference": f"spei_{s}_reference" for s in SCALES})


def _largest_disagreements(canonical: pd.DataFrame, reference: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict]:
    joined = canonical.merge(reference, on=["site", "date"], how="left", validate="one_to_one")
    joined = joined.sort_values(["site", "date"]).copy()
    joined["water_balance_mm"] = joined.PRECTOTCORR * joined.date.dt.days_in_month - joined.pet_mm
    for scale in SCALES:
        joined[f"accumulated_balance_spei_{scale}"] = joined.groupby("site").water_balance_mm.transform(
            lambda values, window=scale: values.rolling(window, min_periods=window).sum()
        )
    window = joined[(joined.date >= "2015-01-01") & (joined.date <= "2019-12-01")].copy()
    target_rows, site_scale = [], []
    for site, group in window.groupby("site", sort=True):
        group = group.sort_values("date").copy()
        for scale in SCALES:
            local_col, ref_col = f"spei_{scale}", f"spei_{scale}_reference"
            valid = group[local_col].notna() & group[ref_col].notna()
            paired = group.loc[valid].copy()
            paired["difference"] = paired[local_col] - paired[ref_col]
            paired["abs_difference"] = paired.difference.abs()
            episode_detail = {}
            for threshold in (-1.0, -1.5, -2.0):
                disagreement = (group[local_col] <= threshold) != (group[ref_col] <= threshold)
                disagreement &= group[local_col].notna() & group[ref_col].notna()
                runs = []
                start = last = None
                for date, differs in zip(group.date, disagreement):
                    if differs:
                        if start is None or date != last + pd.offsets.MonthBegin(1):
                            if start is not None:
                                runs.append({"start": start.strftime("%Y-%m"), "end": last.strftime("%Y-%m"), "months": (last.year - start.year) * 12 + last.month - start.month + 1})
                            start = date
                        last = date
                    elif start is not None:
                        runs.append({"start": start.strftime("%Y-%m"), "end": last.strftime("%Y-%m"), "months": (last.year - start.year) * 12 + last.month - start.month + 1})
                        start = last = None
                if start is not None:
                    runs.append({"start": start.strftime("%Y-%m"), "end": last.strftime("%Y-%m"), "months": (last.year - start.year) * 12 + last.month - start.month + 1})
                episode_detail[str(threshold)] = runs
            for row in paired.nlargest(5, "abs_difference").itertuples(index=False):
                date = pd.Timestamp(row.date)
                site_history = canonical[canonical.site == site]
                wb_window = site_history.loc[
                    (site_history.date <= date) & (site_history.date > date - pd.DateOffset(months=scale)),
                    ["date", "PRECTOTCORR", "pet_mm"],
                ].copy()
                wb_window["precipitation_total_mm"] = wb_window.PRECTOTCORR * wb_window.date.dt.days_in_month
                wb_window["water_balance_mm"] = wb_window.precipitation_total_mm - wb_window.pet_mm
                target_rows.append({
                    "site": site, "scale": scale, "date": date,
                    "local_spei": float(getattr(row, local_col)),
                    "reference_spei": float(getattr(row, ref_col)),
                    "local_minus_reference": float(row.difference),
                    "absolute_difference": float(row.abs_difference),
                    "local_category": int(drought_category(np.array([getattr(row, local_col)]))[0]),
                    "reference_category": int(drought_category(np.array([getattr(row, ref_col)]))[0]),
                    "local_precip_mm_day": float(row.PRECTOTCORR), "local_pet_mm_month": float(row.pet_mm),
                    "local_water_balance_mm_month": float(row.PRECTOTCORR * date.days_in_month - row.pet_mm),
                    "local_trailing_accumulated_water_balance_mm": float(group.loc[group.date == date, f"accumulated_balance_spei_{scale}"].iloc[0]),
                    "input_window_months": json.dumps(wb_window.to_dict("records"), default=str),
                })
            site_scale.append({
                "site": site, "scale": scale, "paired_months": int(valid.sum()),
                "mean_bias_2015_2019": float(paired.difference.mean()) if len(paired) else None,
                "mae_2015_2019": float(paired.difference.abs().mean()) if len(paired) else None,
                "rmse_2015_2019": float(np.sqrt(np.mean(paired.difference ** 2))) if len(paired) else None,
                "largest_absolute_disagreements": [
                    {"date": pd.Timestamp(row.date).strftime("%Y-%m"), "local": float(getattr(row, local_col)), "reference": float(getattr(row, ref_col)), "difference": float(row.difference)}
                    for row in paired.nlargest(5, "abs_difference").itertuples(index=False)
                ],
                "drought_category_disagreement_episodes": episode_detail,
            })
    result = pd.DataFrame(target_rows)
    result.to_csv(output, index=False, date_format="%Y-%m-%d")
    return result, {"period": ["2015-01", "2019-12"], "largest_month_rows": len(result), "site_scale": site_scale}


def _period_comparison(canonical: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    joined = canonical.merge(reference, on=["site", "date"], how="left", validate="one_to_one")
    summaries = []
    joined["period_start_year"] = (joined.date.dt.year // 5) * 5
    joined = joined[(joined.date.dt.year >= 1990) & (joined.date.dt.year <= 2024)]
    for (site, period), site_period in joined.groupby(["site", "period_start_year"]):
        for scale in SCALES:
            local_col, ref_col = f"spei_{scale}", f"spei_{scale}_reference"
            valid = site_period[local_col].notna() & site_period[ref_col].notna()
            x = site_period.loc[valid, local_col].to_numpy(float)
            y = site_period.loc[valid, ref_col].to_numpy(float)
            if not len(x):
                continue
            summaries.append({
                "site": site, "scale": scale, "period": f"{period}-{period + 4}", "site_month_pairs": len(x),
                "mean_bias": float(np.mean(x - y)), "mae": float(np.mean(np.abs(x - y))),
                "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                "category_agreement": float(np.mean(drought_category(x) == drought_category(y))),
                "local_severe_months": int((x <= -1.5).sum()), "reference_severe_months": int((y <= -1.5).sum()),
            })
    return summaries


def _period_comparison_summary(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    results = []
    for scale, group in frame.groupby("scale"):
        counts_as_worst = 0
        site_deltas = []
        for _, site_group in group.groupby("site"):
            current = site_group[site_group.period == "2015-2019"]
            other = site_group[site_group.period != "2015-2019"]
            if current.empty or other.empty:
                continue
            current_rmse = float(current.rmse.iloc[0])
            counts_as_worst += int(current_rmse >= float(other.rmse.max()))
            site_deltas.append(current_rmse - float(other.rmse.mean()))
        current_rows = group[group.period == "2015-2019"]
        other_rows = group[group.period != "2015-2019"]
        results.append({
            "scale": int(scale), "site_count": int(group.site.nunique()),
            "sites_where_2015_2019_is_worst_rmse_period": counts_as_worst,
            "2015_2019_macro_site_rmse": float(current_rows.rmse.mean()),
            "other_period_macro_site_rmse": float(other_rows.rmse.mean()),
            "mean_site_rmse_difference_vs_other_periods": float(np.mean(site_deltas)) if site_deltas else None,
        })
    return results


def _fit_status_map(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    statuses = {}
    for item in data["fit_diagnostics"]:
        key = (item["site"], int(item["scale"]), item["transformation"], item["baseline"])
        statuses.setdefault(key, []).append(item["fit_status"])
    return {key: ("partial_fit_failure" if "failed" in vals else "fit_ok") for key, vals in statuses.items()}


def _target_sensitivity_2015_2019(config: dict, metadata: dict, reference: pd.DataFrame, alternatives: pd.DataFrame) -> list[dict]:
    results = []
    candidates = sorted(set(zip(alternatives.transformation, alternatives.baseline)))
    candidates = [candidate for candidate in candidates if candidate[0] != "generalized_logistic_mle"]
    cells = {row["site"]: row for row in metadata["site_reference_cells"]}
    for site, site_info in config["sites"].items():
        for scale in SCALES:
            ref_site = reference[(reference.site == site) & reference[f"spei_{scale}_reference"].notna()]
            common_dates = set(ref_site.loc[(ref_site.date >= "2015-01-01") & (ref_site.date <= "2019-12-01"), "date"])
            site_candidates = {}
            for transformation, baseline in candidates:
                values = alternatives[
                    (alternatives.site == site) & (alternatives.scale == scale)
                    & (alternatives.transformation == transformation) & (alternatives.baseline == baseline)
                ][["site", "date", "spei"]].rename(columns={"spei": "spei_local"})
                site_candidates[(transformation, baseline)] = values
                valid_dates = values.loc[
                    values.spei_local.notna() & (values.date >= "2015-01-01") & (values.date <= "2019-12-01"), "date"
                ]
                common_dates.intersection_update(valid_dates)
            canonical_key = ("nonparametric", "1990-2010")
            canonical = site_candidates[canonical_key].set_index("date").spei_local
            for candidate_key, values in site_candidates.items():
                candidate = values.set_index("date").spei_local
                local_diff = candidate.loc[sorted(common_dates)].to_numpy(float) - canonical.loc[sorted(common_dates)].to_numpy(float)
                joined = align_monthly(values, ref_site[["site", "date", f"spei_{scale}_reference"]].rename(columns={f"spei_{scale}_reference": "spei_reference"}), "spei_local", "spei_reference")
                joined = joined[joined.date.isin(common_dates)].rename(columns={"spei_local": f"spei_{scale}_local", "spei_reference": f"spei_{scale}_reference"})
                metrics, _ = compare_pair(site, scale, f"{candidate_key[0]}_{candidate_key[1]}", joined, site_info, cells[site], "2015-01-01", "2019-12-01")
                results.append({
                    "site": site, "scale": scale, "transformation": candidate_key[0], "baseline": candidate_key[1],
                    "shared_months_across_candidates": len(common_dates),
                    "mean_spei_difference_vs_canonical_1990_nonparametric": float(np.mean(local_diff)) if len(local_diff) else None,
                    "rmse_vs_canonical_1990_nonparametric": float(np.sqrt(np.mean(local_diff ** 2))) if len(local_diff) else None,
                    "reference_metrics": metrics,
                })
    return results


def _make_unified_comparison(config: dict, metadata_v211: dict, reference: pd.DataFrame, alternatives: pd.DataFrame, fit_status: dict, output: Path) -> tuple[pd.DataFrame, list[dict]]:
    metrics, details = [], []
    period_start, period_end = metadata_v211["comparison_period"]
    target_specs = sorted(set(zip(alternatives.transformation, alternatives.baseline)))
    available_specs = [spec for spec in target_specs if spec[0] != "generalized_logistic_mle"]
    cells = {row["site"]: row for row in metadata_v211["site_reference_cells"]}
    for site, site_info in config["sites"].items():
        shared_dates_by_scale = {}
        for scale in SCALES:
            ref_dates = reference.loc[
                (reference.site == site) & reference[f"spei_{scale}_reference"].notna(), "date"
            ]
            common_dates = set(ref_dates)
            for candidate_transformation, candidate_baseline in available_specs:
                candidate = alternatives[
                    (alternatives.site == site)
                    & (alternatives.scale == scale)
                    & (alternatives.transformation == candidate_transformation)
                    & (alternatives.baseline == candidate_baseline)
                    & alternatives.spei.notna()
                ]
                common_dates.intersection_update(candidate.date)
            shared_dates_by_scale[scale] = common_dates
        for transformation, baseline in target_specs:
            local_target = alternatives[
                (alternatives.site == site)
                & (alternatives.transformation == transformation)
                & (alternatives.baseline == baseline)
            ][["site", "date", "scale", "spei"]]
            for scale in SCALES:
                local_site = local_target[local_target.scale == scale][["site", "date", "spei"]].rename(columns={"spei": "spei_local"})
                ref_site = reference[reference.site == site][["site", "date", f"spei_{scale}_reference"]].rename(columns={f"spei_{scale}_reference": "spei_reference"})
                aligned = align_monthly(local_site, ref_site, "spei_local", "spei_reference")
                aligned = aligned.rename(columns={"spei_local": f"spei_{scale}_local", "spei_reference": f"spei_{scale}_reference"})
                state = fit_status.get((site, scale, transformation, baseline), "excluded_unstable_parameters" if transformation == "generalized_logistic_mle" else "no_fit_diagnostic")
                pair_period = aligned[(aligned.date >= pd.Timestamp(period_start)) & (aligned.date <= pd.Timestamp(period_end))]
                pairwise_valid = pair_period[f"spei_{scale}_local"].notna() & pair_period[f"spei_{scale}_reference"].notna()
                pairwise_row = {
                    "n_pairs": int(pairwise_valid.sum()),
                    "local_missing_months": int(pair_period[f"spei_{scale}_local"].isna().sum()),
                    "reference_missing_months": int(pair_period[f"spei_{scale}_reference"].isna().sum()),
                }
                common_dates = shared_dates_by_scale[scale]
                common = aligned[aligned.date.isin(common_dates)].copy()
                row, detail = compare_pair(site, scale, f"{transformation}_{baseline}", common, site_info, cells[site], period_start, period_end)
                row.update({
                    "pairwise_n": pairwise_row["n_pairs"],
                    "common_overlap_n": len(common_dates) if transformation != "generalized_logistic_mle" else 0,
                    "common_excluded_months": max(0, 420 - len(common_dates)) if transformation != "generalized_logistic_mle" else 420,
                    "expected_months": 420,
                    "local_missing_months": pairwise_row["local_missing_months"],
                    "reference_missing_months": pairwise_row["reference_missing_months"],
                    "comparison_basis": "shared valid calendar months across all available NP/Fisk baselines and v2.11 reference",
                })
                row.update({"reference_version": "SPEIbase v2.11.0", "comparison_status": "computed" if row["n_pairs"] else "no_valid_target_pairs", "fit_status": state})
                metrics.append(row)
                details.append({**detail, "pairwise_n": pairwise_row["n_pairs"], "common_overlap_n": len(common_dates) if transformation != "generalized_logistic_mle" else 0, "reference_version": "SPEIbase v2.11.0", "fit_status": state})
                # Keep the v2.12 non-availability explicit for every requested combination.
                blocked = {
                    "site": site, "scale": scale, "target": f"{transformation}_{baseline}",
                    "reference_version": "SPEIbase v2.12.0", "comparison_status": "blocked_reference_files_unavailable",
                    "fit_status": state, "n_pairs": 0, "pairwise_n": 0,
                    "common_overlap_n": 0, "expected_months": 420,
                    "local_missing_months": int(local_site.spei_local.isna().sum()), "reference_missing_months": None,
                }
                metrics.append(blocked)
    result = pd.DataFrame(metrics)
    result.to_csv(output, index=False)
    return result, details


def _summary_by_target(frame: pd.DataFrame) -> list[dict]:
    summaries = []
    usable = frame[(frame.comparison_status == "computed") & frame.n_pairs.gt(0)]
    for (reference, target, scale), group in usable.groupby(["reference_version", "target", "scale"]):
        summaries.append({
            "reference_version": reference, "target": target, "scale": int(scale),
            "site_count": int(group.site.nunique()), "macro_mean_n": float(group.n_pairs.mean()),
            "pearson_r_site_mean": float(group.pearson_r.mean()) if group.pearson_r.notna().any() else None,
            "spearman_r_site_mean": float(group.spearman_r.mean()) if group.spearman_r.notna().any() else None,
            "bias_site_mean": float(group.mean_bias_local_minus_reference.mean()) if group.mean_bias_local_minus_reference.notna().any() else None,
            "mae_site_mean": float(group.mae.mean()) if group.mae.notna().any() else None,
            "rmse_site_mean": float(group.rmse.mean()) if group.rmse.notna().any() else None,
            "category_agreement_site_mean": float(group.category_agreement_4class.mean()) if group.category_agreement_4class.notna().any() else None,
        })
    return summaries


def run(config_path: Path, canonical_path: Path, reference_path: Path, ref_metadata_path: Path, alt_path: Path, fit_path: Path, v212_metadata_path: Path, output_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metadata_v211 = json.loads(ref_metadata_path.read_text(encoding="utf-8"))
    metadata_v212 = json.loads(v212_metadata_path.read_text(encoding="utf-8"))
    canonical = pd.read_csv(canonical_path, parse_dates=["date"])
    reference = _reference_from_v211(reference_path)
    alternatives = pd.read_csv(alt_path, parse_dates=["date"])
    fit_status = _fit_status_map(fit_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _disagreements_csv, disagreement_data = _largest_disagreements(
        canonical, reference, output_dir / "spei_2015_2019_disagreement.csv"
    )
    period_metrics = _period_comparison(canonical, reference)
    period_summary = _period_comparison_summary(period_metrics)
    comparison, detail = _make_unified_comparison(
        cfg, metadata_v211, reference, alternatives, fit_status,
        output_dir / "spei_reference_comparison_complete.csv",
    )
    disagreement_data["five_year_comparisons"] = period_metrics
    disagreement_data["five_year_comparison_summary"] = period_summary
    disagreement_data["target_sensitivity_2015_2019"] = _target_sensitivity_2015_2019(cfg, metadata_v211, reference, alternatives)
    disagreement_data["causes"] = {
        "confirmed": [
            "Reference is a 0.5-degree nearest grid cell while local values represent configured point coordinates; distances are recorded in the v2.11 comparison metadata/CSV.",
            "The sources and target methods differ: local uses NASA POWER and Hargreaves-Samani PET with a nonparametric transform; SPEIbase uses CRU TS and the SPEI package log-logistic/PWM procedure.",
            "2015-2019 has higher per-scale five-year RMSE than the other assessed five-year bins; computed values and category episodes are listed in this diagnostic.",
        ],
        "not_verifiable_with_available_artifacts": [
            "CRU precipitation and PET time series for the 12 matched cells are not present, so input-level precipitation/PET differences cannot be quantified.",
            "Reference-cell raw water balance and its fitted parameters are unavailable; the local accumulated balance is reported, but attribution to precipitation, PET, accumulation, or distribution cannot be isolated.",
            "SPEIbase v2.12 values are unavailable because the documented v2.12 NetCDF paths return 404; version effects cannot be quantified.",
        ],
        "missingness_2015_2019": {
            "canonical_target_missing": int(canonical.loc[(canonical.date >= "2015-01-01") & (canonical.date <= "2019-12-01"), [f"spei_{s}" for s in SCALES]].isna().sum().sum()),
            "local_precipitation_missing": int(canonical.loc[(canonical.date >= "2015-01-01") & (canonical.date <= "2019-12-01"), "PRECTOTCORR"].isna().sum()),
            "local_pet_missing": int(canonical.loc[(canonical.date >= "2015-01-01") & (canonical.date <= "2019-12-01"), "pet_mm"].isna().sum()),
        },
    }
    (output_dir / "spei_disagreement_diagnostics.json").write_text(json.dumps(disagreement_data, indent=2, allow_nan=False), encoding="utf-8")
    fit_diagnostics = json.loads(fit_path.read_text(encoding="utf-8"))
    execution_path = output_dir / "spei_validation_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else {
        "status": "not_recorded",
        "note": "Run the targeted and full test suites and Ruff before presenting completion status.",
    }
    report = {
        "v211_reference": metadata_v211,
        "v212_reference_availability": metadata_v212,
        "spei_fit_diagnostics_summary": {
            "fit_status_counts": pd.Series([item["fit_status"] for item in fit_diagnostics["fit_diagnostics"]]).value_counts().to_dict(),
            "fisk_failed_fits": [item for item in fit_diagnostics["fit_diagnostics"] if item["fit_status"] == "failed"],
            "generalized_logistic": fit_diagnostics["transformations"]["generalized_logistic_mle"],
        },
        "unified_summary": _summary_by_target(comparison),
        "unified_sample_protocol": "All numeric nonparametric and Fisk transformation/baseline metrics for a site/scale use the same exact intersection of valid target months across the four available target variants and the v2.11 reference. CSV fields also retain each variant's pairwise_n, actual missingness, and common_excluded_months. Excluded generalized-logistic rows are not part of this intersection.",
        "unified_site_scale_records": detail,
        "disagreement_summary": disagreement_data,
        "measured_evidence_vs_interpretation": {
            "measured": ["Exact aligned monthly local-reference metrics, site-cell distances, target fit failures, and 2015-2019 disagreements are included in machine-readable outputs."],
            "interpretation": ["The observed 2015-2019 deterioration is real in this comparison, but its cause is not identified by index-only reference values."],
            "unverified_hypotheses": ["Differences in CRU versus NASA POWER precipitation, PET method, grid versus point representation, and calibration/distribution may contribute; this dataset does not isolate their effects."],
        },
        "limitations": [
            "SPEIbase v2.12.0 is documented but official expected files were unavailable during this run; v2.11 is the only independently matched reference used for numeric metrics.",
            "The reference ends in 2024; canonical local data through 2026 is not reference-validated.",
            "SPEIbase is a gridded comparison reference, not ground truth.",
            "Drought-month samples are temporally correlated and site/grid cells are spatially dependent; no independent-sample confidence claims are made.",
            "Fisk PWM target fit failures remain missing and are enumerated, not silently removed.",
        ],
        "verification": execution,
    }
    (output_dir / "spei_reference_validation_complete.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "spei_reference_validation_complete.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict) -> str:
    lines = [
        "# Complete independent SPEI validation — Phase 1.2", "",
        "## Reference availability", "",
        "SPEIbase v2.12.0 is documented as CRU TS 4.10, 0.5-degree monthly, 1901–2025. However, the official expected `spei03.nc`, `spei06.nc`, and `spei12.nc` endpoints returned HTTP 404 during this run. No v2.12 values were downloaded or substituted. Numeric results therefore use the existing SPEIbase v2.11 matched-site artifact (CRU TS 4.09, 1901–2024). See the v2.12 metadata and version-comparison status row.", "",
        "## Local transformations versus SPEIbase v2.11", "",
        "All four numeric nonparametric/Fisk variants for a given site and scale use the same exact intersection of valid months. The CSV also records each variant's pairwise count and missing values; failed Fisk calendar-month fits are not silently dropped.", "",
        "| Target | Scale | Sites | Mean Pearson r | Mean bias | Mean MAE | Mean RMSE | Mean category agreement |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["unified_summary"]:
        values = [item[key] for key in (
            "pearson_r_site_mean", "bias_site_mean", "mae_site_mean",
            "rmse_site_mean", "category_agreement_site_mean",
        )]
        formatted = [f"{value:.3f}" if value is not None else "NA" for value in values]
        lines.append(f"| {item['target']} | SPEI-{item['scale']} | {item['site_count']} | {' | '.join(formatted)} |")
    lines += ["", "## 2015–2019 period comparison", ""]
    for item in report["disagreement_summary"]["five_year_comparison_summary"]:
        lines.append(
            f"SPEI-{item['scale']}: 2015–2019 is the worst-RMSE five-year period for "
            f"{item['sites_where_2015_2019_is_worst_rmse_period']} of {item['site_count']} sites; "
            f"macro site RMSE {item['2015_2019_macro_site_rmse']:.3f} versus "
            f"{item['other_period_macro_site_rmse']:.3f} in other periods."
        )
    fit = report["spei_fit_diagnostics_summary"]
    verification = report.get("verification", {})
    verification_text = json.dumps(verification, sort_keys=True)
    lines += ["", "## Fit stability and calibration", "", f"Fit status counts: `{json.dumps(fit['fit_status_counts'], sort_keys=True)}`. Fisk PWM failed fits are listed by site, scale, and calendar month in `spei_fit_diagnostics.json`; failed months remain missing. The generalized-logistic MLE candidate was excluded from numeric target output due to highly variable/extreme shape estimates: median {fit['generalized_logistic']['shape_median']:.3g}, 75th percentile {fit['generalized_logistic']['shape_75th_percentile']:.3g}, maximum {fit['generalized_logistic']['shape_max']:.3g}; {fit['generalized_logistic']['shape_over_100_rows']} of {fit['generalized_logistic']['fit_rows']} calibration-month fits exceeded shape 100. This is instability evidence, not proof that every generalized-logistic formulation is invalid.", "", "Each per-month fit used only its declared baseline through the protocol calibration end. The 1981 baseline uses cached pre-1990 POWER data for accumulation/calibration while the 1990+ water balance is taken directly from canonical precipitation and PET columns. Canonical SPEI values were not edited.", "", "## 2015–2019 disagreement", "", "The per-site/scale top-five differences, categories, local water balance, and trailing accumulation windows are in `spei_2015_2019_disagreement.csv`. Five-year metrics for other periods, target sensitivity, and category-disagreement episodes are in `spei_disagreement_diagnostics.json`. Numeric findings establish where the mismatch is concentrated; they do not isolate its cause. CRU precipitation/PET and reference raw water balance were not available locally, so input-level attribution remains unverified.", "", "## Evidence, interpretation, and limitations", "", "Measured evidence includes per-site metrics, sample counts, missingness, grid distance, fit diagnostics, and disagreement periods. The products differ in input dataset, PET method, standardization, spatial support, and reference period. These differences are plausible contributors, but this study cannot attribute a share of error to each factor. SPEIbase is not ground truth; the common-period results cover 1990–2024 only. v2.12 version effects remain blocked by unavailable files.", "", "**Recommendation: RECONSIDER before using canonical SPEI as independently validated research truth; retain the current canonical values unchanged pending source-level investigation.** SPEI-3 agreement is weaker than SPEI-6/12, 2015–2019 has the highest macro five-year RMSE, and the 1981 baseline does not uniformly improve metrics. Do not select a target method solely by reference correlation.", "", "## Verification", "", f"Execution record: `{verification_text}`. No model training or forward pass was executed.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--canonical", type=Path, default=Path("data/maharashtra/monthly_sites.csv"))
    parser.add_argument("--reference", type=Path, default=Path("results/maharashtra/speibase_validation/matched_monthly_points.csv"))
    parser.add_argument("--reference-metadata", type=Path, default=Path("results/maharashtra/speibase_validation/validation_metadata.json"))
    parser.add_argument("--alternatives", type=Path, default=Path("results/maharashtra/phase1/spei_alternative_monthly.csv"))
    parser.add_argument("--fit-diagnostics", type=Path, default=Path("results/maharashtra/phase1/spei_fit_diagnostics.json"))
    parser.add_argument("--v212-metadata", type=Path, default=Path("results/maharashtra/phase1/speibase_v212_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/maharashtra/phase1"))
    args = parser.parse_args()
    run(args.config, args.canonical, args.reference, args.reference_metadata, args.alternatives, args.fit_diagnostics, args.v212_metadata, args.output_dir)
    print(f"Wrote complete SPEI validation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
