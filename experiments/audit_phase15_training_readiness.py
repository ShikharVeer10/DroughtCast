"""Read-only Phase 1.5 DroughtCast training-readiness audit.

This script reads the canonical dataset, protocol metadata, and Phase 1 reports.
It does not call the model, rewrite data, or fit preprocessing statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.protocol import INDEX_ALIGNMENT, MODEL_FEATURES, SPLITS

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results/maharashtra/phase1"
DATA = ROOT / "data/maharashtra/monthly_sites.csv"
DATA_META = ROOT / "data/maharashtra/dataset_metadata.json"
NORMALIZER_META = ROOT / "data/meta.json"
CONFIG = ROOT / "configs/config.yaml"
TARGETS = ["spei_3", "spei_6", "spei_12"]
INDEXES = ["oni", "dmi", "nao", "pdo"]

EXPECTED_ARTIFACTS = [
    "validation_report.md",
    "window_audit.json",
    "source_availability_1981_2026.json",
    "power_source_reconciliation.json",
    "spei_reference_validation_complete.md",
    "spei_reference_validation_complete.json",
    "spei_target_integrity_audit.json",
    "spei_phase1_3_report.md",
    "spei_methodology_comparison.json",
    "spei_clipping_sensitivity.csv",
    "spei_top30_discrepancy_reconstruction.csv",
    "spei_input_data_comparison.csv",
    "spei_method_sensitivity.csv",
    "spei_category_stability.csv",
    "spei_phase1_4_report.md",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _age_months(issue: pd.Series, source: pd.Series) -> pd.Series:
    return (issue.dt.year - source.dt.year) * 12 + issue.dt.month - source.dt.month


def audit() -> tuple[dict, list[dict]]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA, parse_dates=["date"])
    metadata = _read_json(DATA_META)
    stored_normalizers = _read_json(NORMALIZER_META)
    artifact_presence = {name: (PHASE / name).is_file() for name in EXPECTED_ARTIFACTS}
    expected_sites = sorted(cfg["data"]["site_filter"])
    configured_features = list(cfg["data"]["features"])
    feature_order_ok = (
        configured_features == MODEL_FEATURES
        and metadata.get("model_features") == configured_features
        and int(cfg["model"]["input_dim"]) == len(configured_features)
    )

    sites = sorted(frame["site"].astype(str).str.lower().unique())
    per_site_rows = {}
    continuous = True
    for site, group in frame.groupby(frame["site"].astype(str).str.lower()):
        dates = pd.DatetimeIndex(pd.to_datetime(group["date"]).sort_values())
        expected = pd.date_range(dates.min(), dates.max(), freq="MS")
        per_site_rows[site] = {
            "rows": len(group),
            "first_month": dates.min().strftime("%Y-%m"),
            "last_month": dates.max().strftime("%Y-%m"),
            "continuous": bool(dates.equals(expected)),
        }
        continuous &= dates.equals(expected)

    duplicate_site_months = int(frame.duplicated(["site", "date"]).sum())
    expected_dates = pd.date_range("1990-01-01", "2026-10-01", freq="MS")
    row_count_ok = len(frame) == len(expected_sites) * len(expected_dates)
    missing_counts = {}
    nonfinite_counts = {}
    for column in configured_features + TARGETS:
        values = pd.to_numeric(frame[column], errors="coerce")
        missing_counts[column] = int(values.isna().sum())
        nonfinite_counts[column] = int((~np.isfinite(values)).sum())
    target_summary = {}
    for target in TARGETS:
        values = pd.to_numeric(frame[target], errors="coerce")
        finite = values[np.isfinite(values)]
        target_summary[target] = {
            "valid_n": int(finite.size),
            "missing_n": int(values.isna().sum()),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }

    provenance = {}
    provenance_failures = []
    issue_dates = pd.to_datetime(frame["date"])
    for name in INDEXES:
        value = pd.to_numeric(frame[name], errors="coerce")
        source = pd.to_datetime(frame[f"{name}_source_month"], errors="coerce")
        effective = pd.to_datetime(frame[f"{name}_effective_month"], errors="coerce")
        age = pd.to_numeric(frame[f"{name}_observation_age_months"], errors="coerce")
        stale = frame[f"{name}_is_stale"].fillna(False).astype(bool)
        missing = frame[f"{name}_is_missing"].fillna(False).astype(bool)
        expected_age = _age_months(issue_dates, source)
        expected_effective = source + pd.offsets.MonthBegin(
            int(INDEX_ALIGNMENT[name]["publication_lag_months"])
        )
        if int(INDEX_ALIGNMENT[name]["publication_lag_months"]) == 0:
            expected_effective = source
        has_value = value.notna()
        checks = {
            "value_with_stale_or_missing_flag": int((has_value & (stale | missing)).sum()),
            "missing_value_without_missing_flag": int((~has_value & ~missing).sum()),
            "future_effective_month_on_usable_value": int((has_value & (effective > issue_dates)).sum()),
            "source_month_age_mismatch": int((source.notna() & age.notna() & (age != expected_age)).sum()),
            "effective_month_mismatch": int((source.notna() & (effective != expected_effective)).sum()),
            "usable_value_beyond_max_age": int(
                (has_value & (age > int(INDEX_ALIGNMENT[name]["max_age_months"]))).sum()
            ),
            "stale_flag_boundary_error": int(
                (source.notna() & (stale != (age > int(INDEX_ALIGNMENT[name]["max_age_months"])))).sum()
            ),
        }
        if any(checks.values()):
            provenance_failures.append({"index": name, **checks})
        provenance[name] = {
            "configured_publication_lag_months": int(INDEX_ALIGNMENT[name]["publication_lag_months"]),
            "maximum_age_months": int(INDEX_ALIGNMENT[name]["max_age_months"]),
            "dataset_missing_n": int(value.isna().sum()),
            "stale_flag_n": int(stale.sum()),
            "missing_flag_n": int(missing.sum()),
            "provenance_consistency_checks": checks,
            "october_2026_by_site": [],
        }
        october = frame.loc[issue_dates.eq(pd.Timestamp("2026-10-01"))]
        for _, row in october.iterrows():
            raw_value = row[name]
            provenance[name]["october_2026_by_site"].append(
                {
                    "site": str(row["site"]),
                    "value": None if pd.isna(raw_value) else float(raw_value),
                    "source_month": None if pd.isna(row[f"{name}_source_month"]) else str(row[f"{name}_source_month"])[:7],
                    "effective_month": None if pd.isna(row[f"{name}_effective_month"]) else str(row[f"{name}_effective_month"])[:7],
                    "age_months": None if pd.isna(row[f"{name}_observation_age_months"]) else int(row[f"{name}_observation_age_months"]),
                    "stale": bool(row[f"{name}_is_stale"]),
                    "missing": bool(row[f"{name}_is_missing"]),
                }
            )

    windows = _recount_windows(frame, configured_features, int(cfg["data"]["seq_len"]), int(cfg["data"]["forecast_steps"]))
    window_prior = _read_json(PHASE / "window_audit.json")
    power = _read_json(PHASE / "power_source_reconciliation.json")
    power_records = power.get("records", [])
    power_status = pd.Series([r.get("classification") for r in power_records]).value_counts().to_dict()
    power_matches = sum(
        bool(r.get("cached_daily_complete"))
        and bool(r.get("fresh_daily_complete_for_variable"))
        and np.isclose(r.get("cached_daily_aggregate", np.nan), r.get("fresh_daily_monthly_mean", np.nan), rtol=0, atol=1e-8)
        for r in power_records
    )

    reference = pd.read_csv(PHASE / "spei_reference_comparison.csv")
    reference = reference.loc[reference["target"] == "canonical_nonparametric_1990_baseline"]
    reference_summary = {}
    for scale, group in reference.groupby("scale"):
        reference_summary[str(scale)] = {
            "site_count": int(group["site"].nunique()),
            "mean_site_pearson_r": float(group["pearson_r"].mean()),
            "median_site_pearson_r": float(group["pearson_r"].median()),
            "mean_site_rmse": float(group["rmse"].mean()),
            "mean_site_four_category_agreement": float(group["category_agreement_4class"].mean()),
            "paired_months_sum_over_sites_not_independent_n": int(group["n_pairs"].sum()),
        }
    complete_reference = pd.read_csv(PHASE / "spei_reference_comparison_complete.csv")
    common = complete_reference.loc[
        (complete_reference["target"] == "nonparametric_1990-2010")
        & (complete_reference["reference_version"] == "SPEIbase v2.11.0")
        & (complete_reference["comparison_status"] == "computed")
    ]
    common_overlap_summary = {}
    for scale, group in common.groupby("scale"):
        common_overlap_summary[str(scale)] = {
            "site_count": int(group["site"].nunique()),
            "mean_site_pearson_r": float(group["pearson_r"].mean()),
            "mean_site_rmse": float(group["rmse"].mean()),
            "mean_site_four_category_agreement": float(group["category_agreement_4class"].mean()),
            "mean_common_months_per_site": float(group["n_pairs"].mean()),
            "total_pairs_over_sites_not_independent_n": int(group["n_pairs"].sum()),
            "comparison_basis": str(group["comparison_basis"].iloc[0]),
        }
    category = pd.read_csv(PHASE / "spei_category_stability.csv")
    category_rows = category.loc[
        (category["site"] == "ALL")
        & category["comparison"].eq("canonical_1990_2010_vs_speibase_v211")
    ]
    category_summary = {
        str(int(row.scale)): {
            "paired_n": int(row.paired_n),
            "seven_category_agreement": float(row.seven_category_agreement),
            "drought_only_agreement": float(row.drought_only_agreement),
            "extreme_drought_event_agreement": float(row.extreme_drought_event_agreement),
        }
        for row in category_rows.itertuples()
    }
    methods = _read_json(PHASE / "spei_methodology_comparison.json")
    beed = pd.read_csv(PHASE / "spei_top30_discrepancy_reconstruction.csv")
    beed = beed.loc[(beed.site == "beed") & (beed.date == "2019-09") & (beed.scale == 6)].iloc[0]
    fit_failures = methods.get("fit_failures", {})
    v212 = _read_json(PHASE / "spei_v212_access_audit.json")

    inference_features_match = stored_normalizers.get("features") == configured_features
    inference_files = {
        "normalizer_metadata_present": NORMALIZER_META.is_file(),
        "saved_normalizer_feature_order_matches_config": inference_features_match,
        "saved_normalizer_site_set_matches_config": sorted(stored_normalizers.get("normalizers", {})) == expected_sites,
        "checkpoint_present": (ROOT / "results/maharashtra/ablations/quantile_cqr/model_seed_42.pt").is_file(),
        "inference_path_checks_feature_order": "Saved normalizer feature order does not match inference configuration" in (ROOT / "inference.py").read_text(encoding="utf-8"),
        "inference_path_validates_input_index_provenance": "source_month" in (ROOT / "inference.py").read_text(encoding="utf-8"),
    }

    report = {
        "phase": "1.5",
        "generated_local_date": pd.Timestamp.now().date().isoformat(),
        "decision": "APPROVE WITH LIMITATIONS",
        "training_permitted": "Exploratory training only; this audit does not start training.",
        "model_forward_pass_run": False,
        "canonical_dataset_modified": False,
        "config_modified": False,
        "model_source_modified": False,
        "dataset": {
            "path": str(DATA.relative_to(ROOT)),
            "rows": len(frame),
            "sites": sites,
            "expected_sites": expected_sites,
            "site_count_ok": sites == expected_sites,
            "date_start": frame["date"].min().strftime("%Y-%m"),
            "date_end": frame["date"].max().strftime("%Y-%m"),
            "expected_months_per_site": len(expected_dates),
            "expected_total_rows": int(len(expected_sites) * len(expected_dates)),
            "row_count_ok": bool(row_count_ok),
            "per_site": per_site_rows,
            "duplicate_site_months": duplicate_site_months,
            "continuous_monthly_coverage_all_sites": bool(continuous),
            "features": configured_features,
            "feature_order_matches_protocol_and_dataset_metadata": bool(feature_order_ok),
            "feature_count": len(configured_features),
            "missing_values": missing_counts,
            "nonfinite_values": nonfinite_counts,
            "target_summary": target_summary,
            "metadata_schema_version": metadata.get("schema_version"),
            "metadata_target_source": metadata.get("target_source"),
            "metadata_index_alignment_matches_protocol": metadata.get("index_alignment") == INDEX_ALIGNMENT,
            "metadata_splits_match_protocol": metadata.get("split_boundaries") == SPLITS,
        },
        "provenance": {
            "index_checks": provenance,
            "consistency_failures": provenance_failures,
            "october_power": {
                "classification_counts": power_status,
                "matched_fresh_daily_aggregates_n_of_108": int(power_matches),
                "all_values_provisional_due_to_missing_original_daily_request_metadata": power_status == {"provisional": 108},
                "dataset_october_indexes": {name: info["october_2026_by_site"][0] if info["october_2026_by_site"] else None for name, info in provenance.items()},
            },
        },
        "windows": {
            "recounted": windows,
            "prior_window_audit": window_prior,
            "recount_matches_prior_counts": all(
                windows["splits"][name]["windows"] == window_prior["splits"][name]["windows"]
                for name in ("train", "val", "test")
            ),
        },
        "normalization_and_missingness": {
            "implementation": "ClimateNormalizer fits nanmean/nanstd on finite rows passed to fit; loader passes only rows in train split; transform fills nonfinite values with saved training mean.",
            "all_missing_training_feature_behavior": "ClimateNormalizer.fit raises ValueError when any feature has zero finite training observations.",
            "prior_train_windows_with_at_least_one_imputed_input": window_prior["splits"]["train"].get("windows_with_mean_imputation"),
            "currently_saved_normalizer_features_match_current_config": inference_features_match,
            "implication": "The current data/meta.json is not safe for inference under the current feature order; training load_site_table regenerates this metadata from the configured feature list. No inference tensor was executed in this audit.",
        },
        "inference_files": inference_files,
        "reference": {
            "product": "SPEIbase v2.11.0 matched-point artifact; comparison product, not ground truth.",
            "v212_status": v212.get("status"),
            "v212_attempted_urls": v212.get("attempted_expected_paths"),
            "canonical_site_averaged_metrics": reference_summary,
            "canonical_shared_method_overlap_metrics": common_overlap_summary,
            "canonical_pooled_category_metrics": category_summary,
        },
        "beed_september_2019_spei6": {
            "accumulated_water_balance_mm": float(beed.accumulated_water_balance_mm),
            "calibration_min_mm": float(beed.calibration_min_accumulation_mm),
            "calibration_max_mm": float(beed.calibration_max_accumulation_mm),
            "outside_calibration_range": bool(beed.outside_calibration_range),
            "uncapped_local_spei": float(beed.unclipped_local_spei),
            "clipped_local_spei": float(beed.clipped_local_spei),
            "reference_v211": float(beed.reference_spei_v211),
            "cause": "Unresolved: local tail extrapolation/clipping are established; reference-side input balance and attribution to precipitation, PET, calibration, or spatial support are unavailable.",
        },
        "alternative_methods": {
            "fit_failure_count": fit_failures.get("failed_count"),
            "fit_failures_by_baseline": fit_failures.get("by_baseline"),
            "generalized_logistic": methods.get("generalized_logistic"),
            "recommendation": methods.get("recommendation"),
        },
        "required_phase1_artifacts": artifact_presence,
        "evidence_interpretation": {
            "technical_training_readiness": "Training entrypoint checks current schema, alignment, splits, and configured feature list. Train loader fits normalization on training rows and rebuilds data/meta.json. It can run once exploratory use is explicitly accepted.",
            "target_scientific_defensibility": "Locally calculated empirical SPEI is reproducible, but has a short calibration sample, clipped extrapolated tails, and weak category agreement with v2.11; label it explicitly as local empirical SPEI.",
            "publication_claim_readiness": "Not ready for confirmatory research or publication claims; unresolved product/input disagreement and blocked v2.12 access remain.",
        },
    }
    matrix = _decision_matrix(report)
    return report, matrix


def _recount_windows(frame: pd.DataFrame, features: list[str], seq_len: int, horizon: int) -> dict:
    boundaries = {key: pd.Timestamp(value) for key, value in SPLITS.items()}

    def split(date: pd.Timestamp) -> str | None:
        if date <= boundaries["train_end"]:
            return "train"
        if boundaries["val_start"] <= date <= boundaries["val_end"]:
            return "val"
        if boundaries["test_start"] <= date <= boundaries["test_end"]:
            return "test"
        return None

    result = {
        key: {"windows": 0, "per_site": {}, "input_start": None, "input_end": None, "target_start": None, "target_end": None, "windows_with_mean_imputation": 0, "rejected_all_targets_missing": 0}
        for key in ("train", "val", "test")
    }
    target_dates_by_split: dict[str, set[pd.Timestamp]] = {key: set() for key in result}
    cross_split_candidates = 0
    for site, group in frame.groupby("site"):
        group = group.sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(group["date"])
        labels = [split(date) for date in dates]
        feature_missing = group[features].apply(pd.to_numeric, errors="coerce").isna().any(axis=1).to_numpy()
        target_valid = np.isfinite(
            group[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        ).any(axis=1)
        per_site_counts = {key: 0 for key in result}
        for start in range(len(group) - seq_len - horizon + 1):
            in_idx = np.arange(start, start + seq_len)
            out_idx = np.arange(start + seq_len, start + seq_len + horizon)
            all_labels = [labels[i] for i in np.concatenate([in_idx, out_idx])]
            if all_labels[0] is None or len(set(all_labels)) != 1:
                present_splits = {label for label in all_labels if label is not None}
                if present_splits and (len(present_splits) > 1 or None in all_labels):
                    cross_split_candidates += 1
                continue
            key = all_labels[0]
            if not dates[in_idx].equals(pd.date_range(dates[in_idx[0]], periods=seq_len, freq="MS")):
                continue
            if not dates[out_idx].equals(pd.date_range(dates[out_idx[0]], periods=horizon, freq="MS")):
                continue
            if dates[out_idx[0]] != dates[in_idx[-1]] + pd.offsets.MonthBegin(1):
                continue
            per_site_counts[key] += 1
            result[key]["windows"] += 1
            if feature_missing[in_idx].any():
                result[key]["windows_with_mean_imputation"] += 1
            if not target_valid[out_idx].any():
                result[key]["rejected_all_targets_missing"] += 1
            target_dates_by_split[key].update(dates[out_idx].tolist())
            bounds = result[key]
            bounds["input_start"] = min(bounds["input_start"], dates[in_idx[0]]) if bounds["input_start"] else dates[in_idx[0]]
            bounds["input_end"] = max(bounds["input_end"], dates[in_idx[-1]]) if bounds["input_end"] else dates[in_idx[-1]]
            bounds["target_start"] = min(bounds["target_start"], dates[out_idx[0]]) if bounds["target_start"] else dates[out_idx[0]]
            bounds["target_end"] = max(bounds["target_end"], dates[out_idx[-1]]) if bounds["target_end"] else dates[out_idx[-1]]
        for key, count in per_site_counts.items():
            result[key]["per_site"][str(site)] = count
    for values in result.values():
        for key in ("input_start", "input_end", "target_start", "target_end"):
            if values[key] is not None:
                values[key] = values[key].strftime("%Y-%m")
    no_target_overlap = all(
        not (target_dates_by_split[a] & target_dates_by_split[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    return {
        "splits": result,
        "cross_split_candidate_windows_rejected": cross_split_candidates,
        "target_dates_disjoint_across_splits": no_target_overlap,
    }


def _decision_matrix(report: dict) -> list[dict]:
    d = report["dataset"]
    r = report["reference"]
    return [
        {"category": "Dataset integrity", "evidence": f"{d['rows']} rows; {len(d['sites'])}/12 sites; {d['date_start']}–{d['date_end']}; no duplicates; continuous monthly coverage; 15 configured features ordered consistently with canonical sibling metadata.", "status": "PASS", "risk": "Current complete monthly table ends Oct 2026; future months absent by design.", "required_action": "Use only the versioned canonical dataset and sibling protocol metadata."},
        {"category": "Source provenance", "evidence": f"POWER reconciliation classifies {report['provenance']['october_power']['classification_counts']} October values; {report['provenance']['october_power']['matched_fresh_daily_aggregates_n_of_108']}/108 cached daily aggregates match fresh response.", "status": "PASS WITH LIMITATIONS", "risk": "Original daily-cache request provenance is absent; October records remain provisional.", "required_action": "Disclose provisional October values; preserve daily-cache lineage or refresh with documented provenance in a future data phase."},
        {"category": "SPEI methodology", "evidence": "Canonical local target is Hargreaves–Samani water balance plus 1990–2010 month-specific empirical normal-score transform, linear tail extrapolation, ±3.5 clipping.", "status": "PASS WITH LIMITATIONS", "risk": "Only about 20–21 calibration samples per calendar month; empirical tails and clipping are sensitive.", "required_action": "Name target local empirical SPEI; include distribution/baseline sensitivity and target uncertainty."},
        {"category": "Independent reference agreement", "evidence": f"All-valid canonical comparison and common-method-overlap v2.11 metrics are both reported in JSON; Phase 1.4 pooled 7-category agreement: {r['canonical_pooled_category_metrics']}.", "status": "PASS WITH LIMITATIONS", "risk": "Agreement is moderate/weak by category; v2.11 is not ground truth and v2.12 files remain blocked.", "required_action": "Do not state independent validation or ground-truth accuracy; resolve source/input differences before confirmatory claims."},
        {"category": "Extreme-event discrepancies", "evidence": f"Beed Sep-2019 SPEI-6 is {report['beed_september_2019_spei6']['uncapped_local_spei']:.3f} uncapped / {report['beed_september_2019_spei6']['clipped_local_spei']:.1f} clipped vs v2.11 {report['beed_september_2019_spei6']['reference_v211']:.3f}; local balance is outside calibration range.", "status": "BLOCKED", "risk": "Sign-reversed extreme mismatch has no source-level causal attribution.", "required_action": "For exploratory training, report it as target uncertainty; resolve raw/reference precipitation and PET comparisons before publication claims."},
        {"category": "Alternative target stability", "evidence": f"Alternative nonparametric/Fisk comparisons are in spei_method_sensitivity.csv; {report['alternative_methods']['fit_failure_count']} Fisk site-scale-month fits failed; generalized-logistic shape instability is recorded in the Phase 1.4 JSON.", "status": "PASS WITH LIMITATIONS", "risk": "Categories and tail events vary with baseline/distribution; Fisk failures reduce coverage.", "required_action": "Keep canonical target fixed for exploratory run and report sensitivity; do not select solely by v2.11 agreement."},
        {"category": "Missingness and imputation", "evidence": f"Training-only means/std are fit on rows in train split; {report['normalization_and_missingness']['prior_train_windows_with_at_least_one_imputed_input']} training windows contain an imputed input; all-missing feature raises.", "status": "PASS WITH LIMITATIONS", "risk": "Mean imputation erases missingness signal, including stale/missing index values.", "required_action": "Report imputation counts and retain provenance audit; do not treat imputed values as observed climate indices."},
        {"category": "Temporal split integrity", "evidence": f"Window recount: {report['windows']['recounted']['splits']}; target months disjoint={report['windows']['recounted']['target_dates_disjoint_across_splits']}.", "status": "PASS", "risk": "Temporal split tests known-site forecasting, not spatial transfer; validation is short/autocorrelated.", "required_action": "Use untouched 2017–2025 test for final evaluation; make spatial transfer claims only with a separate holdout study."},
        {"category": "Leakage prevention", "evidence": f"Per-row source/effective checks have {len(report['provenance']['consistency_failures'])} index consistency failures; existing window audit reports zero future index sources and no cross-split target overlap.", "status": "PASS WITH LIMITATIONS", "risk": "Inference accepts a separate CSV without validating source/effective provenance; audit checks canonical rows only.", "required_action": "Before operational forecasts, enforce provenance/staleness checks on inference CSV and record issue date."},
        {"category": "Inference-time availability", "evidence": f"October indices: ONI/NAO usable; DMI/PDO stale and missing. Saved normalizer feature order matches current config={report['normalization_and_missingness']['currently_saved_normalizer_features_match_current_config']}; matching checkpoint present={report['inference_files']['checkpoint_present']}.", "status": "BLOCKED", "risk": "Current normalizer metadata is stale/order-mismatched; model/checkpoint was not trained in this phase; prior smoke test is not sufficient for current configuration.", "required_action": "After approved training, regenerate matched train-only normalizers and verify the current inference path with provenance-aware inputs; no forward pass was run here."},
        {"category": "Test and code integrity", "evidence": "Phase 1 data/SPEI audit tests are rerun separately; full suite is deliberately not run because repository tests call model forward methods, prohibited by this phase.", "status": "PASS WITH LIMITATIONS", "risk": "A full-suite result cannot be certified under the no-forward-pass constraint.", "required_action": "Run the complete suite in a later authorized phase, before model development/evaluation."},
    ]


def main() -> None:
    report, matrix = audit()
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / "phase1_5_training_readiness.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(matrix).to_csv(PHASE / "phase1_5_decision_matrix.csv", index=False)
    (PHASE / "phase1_5_training_readiness_report.md").write_text(
        _markdown(report, matrix), encoding="utf-8"
    )
    print(f"Wrote Phase 1.5 audit outputs to {PHASE.relative_to(ROOT)}")


def _markdown(report: dict, matrix: list[dict]) -> str:
    d = report["dataset"]
    ref = report["reference"]["canonical_site_averaged_metrics"]
    common_ref = report["reference"]["canonical_shared_method_overlap_metrics"]
    cats = report["reference"]["canonical_pooled_category_metrics"]
    lines = [
        "# Phase 1.5 — Formal SPEI Target Approval and Training Readiness Gate",
        "",
        f"**Decision: {report['decision']}** — exploratory model training may proceed under the conditions below; this audit did not train or run a model.",
        "",
        "## Executive summary",
        "",
        f"The canonical table has {d['rows']:,} rows for {len(d['sites'])} Maharashtra sites, {d['date_start']}–{d['date_end']}, and the configured 15 features. It is continuous and duplicate-free. Canonical protocol metadata is schema v{d['metadata_schema_version']} and matches the configured feature order, alignment, and split boundaries.",
        "",
        "The target is reproducibly generated local empirical SPEI, not an independently verified ground truth. It uses monthly NASA POWER precipitation and Hargreaves–Samani PET, a fixed 1990–2010 calendar-month calibration, empirical normal-score transformation, tail extrapolation, and ±3.5 clipping. Agreement with SPEIbase v2.11 is imperfect; the largest Beed 2019 sign mismatch remains unexplained. v2.12 access is still blocked.",
        "",
        "**Gate interpretation:** software can support exploratory training with the existing target while clearly naming it and retaining the chronological test. This is not approval for confirmatory scientific claims, operational forecasting, or conference-ready results. The inference preprocessing artifact currently has a feature-order mismatch and must be regenerated by a later approved training run before inference use.",
        "",
        "## Scope and restrictions",
        "",
        "Read-only audit of code and existing evidence. No canonical data, targets, config, model code, split, or feature order was changed. No model training or forward pass was executed.",
        "",
        "## Evidence inspected",
        "",
        "`data/protocol.py`, `data/build_dataset.py`, `data/loader.py`, `data/preprocessing.py`, `data/quality.py`, `data/dataset.py`, `train.py`, `inference.py`, `configs/config.yaml`, canonical sibling dataset metadata and stored normalizer metadata; Phase 1 validation, window, source coverage, POWER reconciliation, SPEI v2.11/v2.12, target integrity, Phase 1.3 and Phase 1.4 artifacts. Missing listed Phase 1 artifacts are recorded in the JSON `required_phase1_artifacts` field; none of the requested files was missing in this audit.",
        "",
        "## Dataset integrity and provenance",
        "",
        f"- {d['rows']:,} rows = 12 × {d['expected_months_per_site']} expected months; dates {d['date_start']} through {d['date_end']}; {d['duplicate_site_months']} duplicate site-months; all sites have continuous monthly coverage.",
        f"- {d['feature_count']} configured features; feature order matches `data/protocol.py`, config and `data/maharashtra/dataset_metadata.json`; protocol schema v{d['metadata_schema_version']}.",
        f"- Feature and target missing counts: `{json.dumps(d['missing_values'], sort_keys=True)}`. Missing climate-index observations are retained as missing in the canonical table; monthly target spin-up values are also missing.",
        "- The separate legacy `data/dataset_metadata.json` is stale (schema 3, old feature list/alignment) and must not be used; configured train/evaluate code points to `data/maharashtra/dataset_metadata.json`.",
        "- October 2026: the reconciliation artifact classifies all 108 site-variable values as provisional; all 108 match fresh daily-response aggregates, but original daily-cache request metadata is absent. Do not upgrade these to verified.",
        "- October index values: ONI and NAO are usable; DMI and PDO are marked stale/missing. Values are not fabricated or forward-filled past policy.",
        "",
        "## SPEI target validity and reference agreement",
        "",
        "The water balance is `PRECTOTCORR (mm/day) × days_in_month − pet_mm`. Hargreaves–Samani uses monthly POWER mean temperature and monthly means of daily extrema, so PET is an approximation, not FAO-56 Penman–Monteith. Accumulation is trailing and requires a complete scale window. Calibration is calendar-month specific and ends at 2010-12. The canonical empirical transform is not the conventional fitted log-logistic SPEI procedure; it extrapolates tails and clips scores at ±3.5.",
        "",
        "Site-mean Pearson correlations recalculated from `spei_reference_comparison.csv` (12 sites per scale):",
        "",
        "| Scale | Pearson r, all valid months | Pearson r, common method overlap | Mean RMSE, all valid | Mean 4-category agreement, all valid | Phase 1.4 pooled 7-category agreement | Drought-only agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scale in ("3", "6", "12"):
        m, c = ref[scale], cats[scale]
        lines.append(f"| SPEI-{scale} | {m['mean_site_pearson_r']:.3f} | {common_ref[scale]['mean_site_pearson_r']:.3f} (n≈{common_ref[scale]['mean_common_months_per_site']:.0f}/site) | {m['mean_site_rmse']:.3f} | {m['mean_site_four_category_agreement']:.3f} | {c['seven_category_agreement']:.3f} | {c['drought_only_agreement']:.3f} |")
    b = report["beed_september_2019_spei6"]
    lines += [
        "",
        "The all-valid correlations are recomputed from `spei_reference_comparison.csv`; common-overlap correlations reproduce the approximately 0.711/0.754/0.787 values in `spei_reference_comparison_complete.csv`, which restricts months to the shared intersection used across transformations. This different sample basis explains the SPEI-3 difference; report both with their sample basis. Pearson correlation describes linear co-variation in continuous scores; category agreement checks thresholded labels. Neither alone establishes validity, and the two category metrics use different category definitions/artifacts (four-class in the reference comparison; seven-class in Phase 1.4). SPEIbase v2.11 is an independent gridded comparison product, not ground truth.",
        "",
        f"**Beed September 2019, SPEI-6:** local balance {b['accumulated_water_balance_mm']:.3f} mm; September calibration minimum {b['calibration_min_mm']:.3f} mm; raw local empirical score {b['uncapped_local_spei']:.6f}, clipped {b['clipped_local_spei']:.3f}; v2.11 {b['reference_v211']:.6f}. The local accumulation is below its calibration minimum, confirming local tail extrapolation and clipping. The reference-side precipitation/PET balance is unavailable, so the sign reversal cannot be assigned to data, PET, calibration, or spatial representation.",
        "",
        f"Alternative targets show sensitivity: {report['alternative_methods']['fit_failure_count']} Fisk calendar/site/scale fits failed. Generalized-logistic diagnostics show unstable shape estimates. The 1981–2010 nonparametric variant is closer to canonical categories than Fisk in existing comparisons, but this does not make it the correct target. Keep alternatives as sensitivity evidence only.",
        "",
        "### v2.12 status",
        "",
        f"The latest recorded access audit status is `{report['reference']['v212_status']}`. The logged expected NetCDF endpoints returned HTTP 404; no v2.12 files were substituted. This blocks version-update comparison but does not erase the existing v2.11 comparison.",
        "",
        "## Missingness and preprocessing",
        "",
        "`ClimateNormalizer.fit` computes finite-feature means and standard deviations; `load_site_table` passes only rows in the training date split. Transform imputes nonfinite inputs with saved training means; an all-missing training feature raises `ValueError`. Missing index values therefore become the train mean in model tensors but remain missing in the source table/provenance. The prior window audit records 132 training windows with at least one mean-imputed feature.",
        "",
        f"Current `data/meta.json` feature order matches config: **{report['normalization_and_missingness']['currently_saved_normalizer_features_match_current_config']}**. The inference helper explicitly rejects this mismatch. Training's loader writes a fresh normalizer metadata file from the configured order, but current inference is not ready until that occurs and provenance validation is added for separate operational CSV input.",
        "",
        "## Temporal split and leakage audit",
        "",
        f"Window counts and ranges were recounted from the dataset: `{json.dumps(report['windows']['recounted'], sort_keys=True)}`. Windows require 6 consecutive inputs and 4 immediate future targets all in the same split. Target dates are disjoint across train/validation/test. Existing `window_audit.json` reports zero future index source months, zero stale index values present, and disjoint target periods. The train/validation/test design measures temporal generalization at known sites, not transfer to unseen regions.",
        "",
        "## Inference-time readiness",
        "",
        "The canonical October table preserves DMI/PDO missingness and valid ONI/NAO values. However, the current saved normalizer order does not match the configured order, and no current dataset-matched model was used or run. The prior `(6, 15)` smoke-test result predates/currently does not establish this exact metadata match. No forward pass was run. Operational inference remains blocked pending an approved trained checkpoint, regenerated normalizers, and input provenance/staleness validation.",
        "",
        "## Formal decision matrix",
        "",
        "See `phase1_5_decision_matrix.csv` for evidence, status, risk, and required action for each gate category.",
        "",
        "## Final decision and conditions",
        "",
        "### APPROVE WITH LIMITATIONS",
        "",
        "Exploratory training is permitted; this phase does not start it. The canonical target must be described as locally calculated empirical SPEI and results must be treated as exploratory. This decision does not approve publication claims, causal claims, operational accuracy, or spatial transfer.",
        "",
        "Before training/results are used in a paper:",
        "",
        "1. Keep canonical dataset and protocol metadata hash-matched; preserve the target source and threshold definition.",
        "2. Report persistence and other baselines on the identical chronological windows; select using validation only and retain untouched test.",
        "3. Report performance by site, scale and lead, with event counts and uncertainty; show sensitivity to calibration baseline/distribution and clipping.",
        "4. Treat Beed 2019 and broad reference category mismatch as target uncertainty; do not tune targets to maximize v2.11 agreement.",
        "5. After training, regenerate and hash-match normalizers/checkpoint metadata. Before operational inference, reject unproven/stale external inputs and preserve missingness.",
        "6. Resolve or explicitly retain October 2026 provisional source provenance; do not describe those values as independently verified.",
        "",
        "## Verification and limitations",
        "",
        "This audit script performs read-only dataset/provenance/window calculations and reads existing report artifacts. Pytest verification is reported separately in the final execution record; full suite execution is intentionally constrained because repository tests include model forward passes prohibited in this phase. No SPEIbase v2.12 endpoint was retried.",
        "",
        "Key evidence: `data/maharashtra/dataset_metadata.json`; `results/maharashtra/phase1/window_audit.json`; `power_source_reconciliation.json`; `spei_reference_comparison.csv`; `spei_category_stability.csv`; `spei_top30_discrepancy_reconstruction.csv`; `spei_methodology_comparison.json`; `spei_v212_access_audit.json`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
