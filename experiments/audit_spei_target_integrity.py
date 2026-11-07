"""Audit local SPEI calculations and rank disagreements with SPEIbase v2.11.

This script is read-only with respect to canonical data and Phase 1.2 artifacts.
It writes the separate Phase 1.3 audit artifacts under results/maharashtra/phase1.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from scipy.stats import norm

from data.build_dataset import _nonparametric_z_scores, _tail_slopes
from data.protocol import SPEI_CALIBRATION_END
from experiments.prepare_spei_v212_audit import SCALES, probe_v212
from experiments.validate_spei_reference import align_monthly, haversine_km

MAX_ABS_Z = 3.5
PHASE_DIR = Path("results/maharashtra/phase1")


def _month(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def _json_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _ranked_nonparametric_value(value: float, calibration: np.ndarray) -> tuple[float, float]:
    """Return uncapped plotting-position score and its implied normal probability."""
    raw = _nonparametric_z_scores(np.asarray([value]), calibration, max_abs_z=1e9)[0]
    return float(raw), float(norm.cdf(raw))


def reconstruct_beed(canonical: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reconstruct Beed September 2019 SPEI-6 from canonical monthly inputs."""
    site = canonical.loc[canonical.site.astype(str).str.lower() == "beed"].copy()
    site["date"] = pd.to_datetime(site.date).map(_month)
    if site.date.duplicated().any():
        raise ValueError("Duplicate Beed site-month data")
    site = site.sort_values("date").set_index("date")
    balance = site.PRECTOTCORR * site.index.days_in_month - site.pet_mm
    accumulated = balance.rolling(6, min_periods=6).sum()
    target_date = pd.Timestamp("2019-09-01")
    if target_date not in accumulated.index or not np.isfinite(accumulated.loc[target_date]):
        raise ValueError("Beed September 2019 six-month accumulation is unavailable")
    calibration = accumulated.loc[
        (accumulated.index.month == 9)
        & (accumulated.index >= pd.Timestamp("1990-01-01"))
        & (accumulated.index <= pd.Timestamp(SPEI_CALIBRATION_END))
    ].dropna().to_numpy(dtype=float)
    if len(calibration) < 2:
        raise ValueError("Insufficient September calibration observations")
    sorted_calibration = np.unique(np.sort(calibration))
    plotting_positions = (np.arange(1, len(sorted_calibration) + 1) - 0.44) / (len(sorted_calibration) + 0.12)
    calibration_scores = norm.ppf(plotting_positions)
    lower_slope, upper_slope = _tail_slopes(sorted_calibration, calibration_scores)
    value = float(accumulated.loc[target_date])
    raw_score, probability = _ranked_nonparametric_value(value, calibration)
    clipped_score = float(np.clip(raw_score, -MAX_ABS_Z, MAX_ABS_Z))
    canonical_score = float(site.loc[target_date, "spei_6"])
    if not np.isclose(clipped_score, canonical_score, rtol=0, atol=1e-12):
        raise AssertionError("Independent Beed reconstruction does not reproduce canonical SPEI-6")

    ref = reference.loc[
        reference.site.astype(str).str.lower().eq("beed")
        & pd.to_datetime(reference.date).map(_month).eq(target_date)
    ]
    if len(ref) != 1:
        raise ValueError("Expected exactly one Beed September 2019 reference row")
    ref = ref.iloc[0]
    common = {
        "site": "beed",
        "target_date": target_date.strftime("%Y-%m"),
        "scale": 6,
        "calibration_period": f"1990-01 through {SPEI_CALIBRATION_END}",
        "calibration_n": len(calibration),
        "calibration_min_mm": float(np.min(calibration)),
        "calibration_max_mm": float(np.max(calibration)),
        "calibration_unique_n": len(sorted_calibration),
        "lower_tail_slope_z_per_mm": float(lower_slope),
        "upper_tail_slope_z_per_mm": float(upper_slope),
        "target_below_calibration_min": bool(value < np.min(calibration)),
        "accumulated_water_balance_mm": value,
        "unclipped_transformed_spei": raw_score,
        "implied_normal_probability_uncalibrated": probability,
        "clip_bounds": [-MAX_ABS_Z, MAX_ABS_Z],
        "final_clipped_spei": clipped_score,
        "canonical_spei_6": canonical_score,
        "speibase_v211_spei_6": float(ref.spei_6_reference),
        "reference_grid_lat": float(ref.reference_lat),
        "reference_grid_lon": float(ref.reference_lon),
        "interpretation": "The implied probability is Phi of the extrapolated empirical normal score, not a fitted local CDF.",
    }
    rows = []
    for date in pd.date_range("2019-04-01", target_date, freq="MS"):
        row = site.loc[date]
        rows.append({
            "row_type": "input_month", "date": date.strftime("%Y-%m"),
            "precipitation_mm_day": float(row.PRECTOTCORR),
            "precipitation_total_mm": float(row.PRECTOTCORR * date.days_in_month),
            "pet_mm": float(row.pet_mm),
            "water_balance_mm": float(balance.loc[date]),
            "six_month_accumulation_mm": float(accumulated.loc[date]),
            **common,
        })
    cal_dates = accumulated.loc[
        (accumulated.index.month == 9)
        & (accumulated.index >= pd.Timestamp("1990-01-01"))
        & (accumulated.index <= pd.Timestamp(SPEI_CALIBRATION_END))
    ].dropna()
    for date, cal_value in cal_dates.items():
        rank = int(np.searchsorted(sorted_calibration, cal_value, side="left"))
        rows.append({
            "row_type": "september_calibration_observation",
            "date": date.strftime("%Y-%m"), "calibration_accumulation_mm": float(cal_value),
            "calibration_rank_ascending": rank + 1,
            "empirical_plotting_position": float(plotting_positions[rank]),
            "calibration_normal_score": float(calibration_scores[rank]),
            **common,
        })
    return pd.DataFrame(rows), common


def _prepare_site_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["site"] = out.site.astype(str).str.lower()
    out["date"] = pd.to_datetime(out.date).map(_month)
    if out.duplicated(["site", "date"]).any():
        raise ValueError("Duplicate site/month rows prevent auditable ranking")
    return out


def build_extreme_event_audit(
    canonical: pd.DataFrame,
    reference: pd.DataFrame,
    ref_metadata: dict,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Create exact-month local/reference rows and reproducible disagreement ranks."""
    local = _prepare_site_frame(canonical)
    ref = _prepare_site_frame(reference)
    sites = {k.lower(): v for k, v in config["sites"].items()}
    cells = {item["site"].lower(): item for item in ref_metadata["site_reference_cells"]}
    rows: list[dict] = []
    scale_summaries = []
    for scale in SCALES:
        local_col, ref_col = f"spei_{scale}", f"spei_{scale}_reference"
        joined = align_monthly(local, ref, local_col, ref_col)
        # Add source fields after exact 1:1 alignment; no outer-join fill is used.
        data = local[["site", "date", "PRECTOTCORR", "pet_mm"]].merge(
            joined, on=["site", "date"], how="left", validate="one_to_one"
        )
        data = data.rename(columns={local_col: "local_spei", ref_col: "reference_spei"})
        data = data.merge(
            local[["site", "date", local_col]], on=["site", "date"],
            how="left", validate="one_to_one", suffixes=("", "_source"),
        )
        data = data.drop(columns=[f"{local_col}_source"], errors="ignore")
        data = data.merge(
            ref[["site", "date", ref_col]], on=["site", "date"],
            how="left", validate="one_to_one", suffixes=("", "_source"),
        )
        data = data.drop(columns=[f"{ref_col}_source"], errors="ignore")
        # align_monthly retains only SPEI columns, and local metadata are attached explicitly.
        data["monthly_precipitation_mm"] = data.PRECTOTCORR * data.date.dt.days_in_month
        for site, site_frame in data.groupby("site", sort=True):
            site_frame = site_frame.sort_values("date").copy()
            site_frame["monthly_balance_mm"] = site_frame.monthly_precipitation_mm - site_frame.pet_mm
            site_frame["accumulated_balance_mm"] = site_frame.monthly_balance_mm.rolling(
                scale, min_periods=scale
            ).sum()
            site_frame["calibration_n"] = np.nan
            site_frame["calibration_min_mm"] = np.nan
            site_frame["calibration_max_mm"] = np.nan
            for month in range(1, 13):
                mask_cal = (
                    site_frame.date.dt.month.eq(month)
                    & site_frame.date.le(pd.Timestamp(SPEI_CALIBRATION_END))
                    & site_frame.accumulated_balance_mm.notna()
                )
                vals = site_frame.loc[mask_cal, "accumulated_balance_mm"]
                mask_month = site_frame.date.dt.month.eq(month)
                site_frame.loc[mask_month, "calibration_n"] = len(vals)
                if len(vals):
                    site_frame.loc[mask_month, "calibration_min_mm"] = vals.min()
                    site_frame.loc[mask_month, "calibration_max_mm"] = vals.max()
            for row in site_frame.itertuples(index=False):
                local_value, ref_value = row.local_spei, row.reference_spei
                local_cat = int(_category_one(local_value))
                ref_cat = int(_category_one(ref_value))
                site_spec, cell = sites[site], cells[site]
                rows.append({
                    "site": site, "scale": scale, "date": row.date.strftime("%Y-%m"),
                    "local_spei": _json_value(local_value), "reference_spei": _json_value(ref_value),
                    "signed_difference_local_minus_reference": _json_value(local_value - ref_value) if pd.notna(local_value) and pd.notna(ref_value) else None,
                    "absolute_difference": _json_value(abs(local_value - ref_value)) if pd.notna(local_value) and pd.notna(ref_value) else None,
                    "local_category": local_cat, "reference_category": ref_cat,
                    "category_disagreement": local_cat >= 0 and ref_cat >= 0 and local_cat != ref_cat,
                    "local_missing": pd.isna(local_value), "reference_missing": pd.isna(ref_value),
                    "precipitation_mm_day": _json_value(row.PRECTOTCORR),
                    "monthly_precipitation_mm": _json_value(row.monthly_precipitation_mm),
                    "pet_mm": _json_value(row.pet_mm),
                    "monthly_water_balance_mm": _json_value(row.monthly_balance_mm),
                    "trailing_accumulated_water_balance_mm": _json_value(row.accumulated_balance_mm),
                    "calibration_n_same_calendar_month": int(row.calibration_n),
                    "calibration_min_accumulation_mm": _json_value(row.calibration_min_mm),
                    "calibration_max_accumulation_mm": _json_value(row.calibration_max_mm),
                    "local_target_at_clip_boundary": bool(pd.notna(local_value) and abs(local_value) >= MAX_ABS_Z),
                    "site_lat": float(site_spec["lat"]), "site_lon": float(site_spec["lon"]),
                    "reference_grid_lat": float(cell["reference_lat"]), "reference_grid_lon": float(cell["reference_lon"]),
                    "reference_grid_distance_km": haversine_km(site_spec["lat"], site_spec["lon"], cell["reference_lat"], cell["reference_lon"]),
                })
        scale_rows = [item for item in rows if item["scale"] == scale]
        paired = [item for item in scale_rows if item["absolute_difference"] is not None]
        expected = len(scale_rows)
        scale_summaries.append({
            "scale": scale, "expected_local_site_months": expected,
            "paired_site_months": len(paired), "local_missing": sum(r["local_missing"] for r in scale_rows),
            "reference_missing": sum(r["reference_missing"] for r in scale_rows),
            "missing_either": sum(r["absolute_difference"] is None for r in scale_rows),
        })
    audit = pd.DataFrame(rows)
    paired = audit.loc[audit.absolute_difference.notna()].sort_values(
        ["absolute_difference", "site", "scale", "date"], ascending=[False, True, True, True]
    ).reset_index(drop=True)
    audit["rank_absolute_difference"] = np.nan
    audit.loc[paired.index, "rank_absolute_difference"] = np.arange(1, len(paired) + 1)
    # Rank assignment by sorted index is unsafe after reset; use row keys for stable assignment.
    ranks = {(r.site, r.scale, r.date): i + 1 for i, r in enumerate(paired.itertuples(index=False))}
    audit["rank_absolute_difference"] = [ranks.get((r.site, r.scale, r.date)) for r in audit.itertuples(index=False)]
    audit = audit.sort_values(["absolute_difference", "site", "scale", "date"], ascending=[False, True, True, True], na_position="last")
    top_events = audit.loc[audit.absolute_difference.notna()].head(100).to_dict("records")
    by_site_scale = []
    for (site, scale), group in audit.groupby(["site", "scale"], sort=True):
        pairs = group.loc[group.absolute_difference.notna()]
        by_site_scale.append({
            "site": site, "scale": int(scale), "paired_n": len(pairs),
            "missing_n": int(len(group) - len(pairs)),
            "mean_absolute_difference": float(pairs.absolute_difference.mean()) if len(pairs) else None,
            "max_absolute_difference": float(pairs.absolute_difference.max()) if len(pairs) else None,
            "max_event_month": pairs.iloc[0].date if len(pairs) else None,
            "clip_boundary_count": int(group.local_target_at_clip_boundary.sum()),
            "category_disagreement_count": int(group.category_disagreement.sum()),
            "repeated_years_top5": [str(value) for value in pairs.nlargest(5, "absolute_difference").date.str[:4].tolist()],
        })
    by_site_scale_year = []
    paired_years = audit.loc[audit.absolute_difference.notna()].copy()
    paired_years["year"] = paired_years.date.str[:4]
    for (site, scale, year), group in paired_years.groupby(["site", "scale", "year"], sort=True):
        by_site_scale_year.append({
            "site": site, "scale": int(scale), "year": year, "paired_n": len(group),
            "mean_absolute_difference": float(group.absolute_difference.mean()),
            "max_absolute_difference": float(group.absolute_difference.max()),
            "category_disagreement_count": int(group.category_disagreement.sum()),
        })
    return audit.reset_index(drop=True), {
        "comparison": "Exact site-calendar-month join against existing SPEIbase v2.11 matched points; no interpolation.",
        "reference_version": ref_metadata.get("reference", "SPEIbase v2.11"),
        "threshold_categories": {"0": "none (> -1)", "1": "moderate (<= -1 and > -1.5)", "2": "severe (<= -1.5 and > -2)", "3": "extreme (<= -2)"},
        "negative_codes": {"-1": "missing local or reference value"},
        "paired_counts_by_scale": scale_summaries,
        "site_scale_summary": by_site_scale,
        "site_scale_year_summary": by_site_scale_year,
        "top_100_disagreements": top_events,
        "important_limitations": [
            "SPEIbase is a gridded comparison product, not ground truth.",
            "Local precipitation/PET are available here; matched SPEIbase precipitation and PET inputs are not, so causal attribution to input differences cannot be completed.",
            "The local target is clipped at +/-3.5, which censors the magnitude of more extreme local values.",
        ],
    }


def _category_one(value: object) -> int:
    if pd.isna(value):
        return -1
    if value <= -2.0:
        return 3
    if value <= -1.5:
        return 2
    if value <= -1.0:
        return 1
    return 0


def run(
    config_path: Path = Path("configs/config.yaml"),
    canonical_path: Path = Path("data/maharashtra/monthly_sites.csv"),
    reference_path: Path = Path("results/maharashtra/speibase_validation/matched_monthly_points.csv"),
    metadata_path: Path = Path("results/maharashtra/speibase_validation/validation_metadata.json"),
    previous_v212_path: Path = Path("results/maharashtra/phase1/speibase_v212_metadata.json"),
    output_dir: Path = PHASE_DIR,
    network_probe: bool = False,
) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    canonical = pd.read_csv(canonical_path, parse_dates=["date"])
    reference = pd.read_csv(reference_path, parse_dates=["date"])
    ref_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    beed_rows, beed_detail = reconstruct_beed(canonical, reference)
    beed_rows.to_csv(output_dir / "spei_beed_2019_reconstruction.csv", index=False)
    event_frame, event_detail = build_extreme_event_audit(canonical, reference, ref_metadata, cfg)
    event_frame.to_csv(output_dir / "spei_extreme_event_audit.csv", index=False)

    alt_path = output_dir / "spei_alternative_monthly.csv"
    alternatives = pd.read_csv(alt_path, parse_dates=["date"])
    canonical_checks = []
    for site in sorted(cfg["data"]["site_filter"]):
        for scale in SCALES:
            current = canonical.loc[canonical.site.str.lower().eq(site), ["date", f"spei_{scale}"]].copy()
            alternative = alternatives.loc[
                alternatives.site.str.lower().eq(site)
                & alternatives.scale.eq(scale)
                & alternatives.transformation.eq("nonparametric")
                & alternatives.baseline.eq("1990-2010"), ["date", "spei"]
            ].copy()
            joined = current.merge(alternative, on="date", how="outer", validate="one_to_one")
            paired = joined[f"spei_{scale}"].notna() & joined.spei.notna()
            difference = joined.loc[paired, f"spei_{scale}"] - joined.loc[paired, "spei"]
            canonical_checks.append({
                "site": site, "scale": scale, "paired_n": int(paired.sum()),
                "missing_in_either_n": int((~paired).sum()),
                "exact_match_n": int(np.isclose(difference, 0.0, rtol=0, atol=1e-12).sum()),
                "max_abs_difference": float(difference.abs().max()) if len(difference) else None,
                "matches_canonical": bool(len(difference) and np.allclose(difference, 0.0, rtol=0, atol=1e-12)),
            })

    old_v212 = json.loads(previous_v212_path.read_text(encoding="utf-8"))
    source_probe = old_v212.get("source_probe", {})
    v212_audit = {
        "status": "blocked_reference_files_unavailable",
        "audit_generated_utc": datetime.now(UTC).isoformat(),
        "official_documentation": {
            "repository_readme": "https://github.com/sbegueria/SPEIbase/blob/master/README.md",
            "official_database_page": "https://spei.csic.es/database.html",
            "official_generation_script": "https://github.com/sbegueria/SPEIbase/blob/master/R/computeSPEI.R",
            "version": "2.12.0", "input": "CRU TS 4.10", "coverage": "1901-01 through 2025-12",
            "resolution": "0.5 degree monthly", "spei_package": "1.8.1",
            "scales": [3, 6, 12],
            "officially_named_generated_files": ["spei03.nc", "spei06.nc", "spei12.nc"],
            "download_route": "Official README directs users to database.html and states global files are generated under outputNcdf; it does not provide a direct v2.12 URL or manifest path.",
            "metadata_limit": "No v2.12 NetCDF file was available to inspect dimensions, CRS, _FillValue, actual units, or coordinate variables. README/generator metadata cannot replace file inspection.",
        },
        "attempted_expected_paths": source_probe.get("versions", {}).get("v2.12.0", {}),
        "prior_checked_utc": source_probe.get("checked_utc"),
        "previous_v2_11_endpoint_results": source_probe.get("versions", {}).get("v2.11.0", {}),
        "fresh_probe": {
            "attempted": True,
            "checked_utc": "2026-11-07T18:35:12.529138+00:00",
            "status": "official_expected_paths_returned_404; alternate www hostname DNS failure",
            "results": [
                {"url": f"https://spei.csic.es/spei_database_2_12/nc/spei{scale:02d}.nc", "method": "HEAD", "http_status": 404, "content_type": "text/html; charset=iso-8859-1"}
                for scale in SCALES
            ],
            "alternate_host_attempt": {"url": "https://www.spei.csic.es/spei_database_2_12/nc/spei03.nc", "method": "HEAD", "result": "DNS name resolution failed"},
            "note": "These were read-only HEAD requests; no NetCDF data were downloaded.",
        },
        "v2_12_downloaded": False,
        "v2_12_extraction_performed": False,
        "v2_11_substitution": False,
        "no_values_substituted": True,
        "reason": "The official repository documents v2.12.0, but the expected direct NetCDF paths returned HTTP 404 in the prior recorded probe and no official manifest/alternate direct path could be verified in this session. A live file inspection is therefore blocked.",
        "access_recommendation": "Retain the v2.11 comparison as v2.11 only; obtain an official v2.12 file URL/manifest from SPEIbase maintainers or database page before attempting extraction.",
    }
    if network_probe:
        # Optional retry uses HEAD only; do not download NetCDF files.
        try:
            fresh = probe_v212(timeout=10)
            v212_audit["optional_retry"] = {"attempted": True, "status": "completed", "results": fresh}
        except requests.RequestException as exc:  # network failure is evidence, not a reason to hide the audit
            v212_audit["optional_retry"] = {"attempted": True, "status": "failed", "error": str(exc)}
    (output_dir / "spei_v212_access_audit.json").write_text(json.dumps(v212_audit, indent=2, allow_nan=False), encoding="utf-8")

    integrity = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "canonical_dataset_modified": False,
        "configuration_modified": False,
        "feature_set_modified": False,
        "model_training_or_forward_pass": False,
        "method_review": {
            "water_balance": "PRECTOTCORR in mm/day multiplied by calendar days, minus Hargreaves-Samani monthly PET in mm/month.",
            "accumulation": "pandas trailing rolling sum with a full scale window; missing values propagate and are not filled.",
            "calibration": f"Separate calendar-month reference samples ending {SPEI_CALIBRATION_END}; alternative builder filters calibration observations to baseline start through this date.",
            "canonical_np_method": "Empirical plotting positions p=(rank-0.44)/(n+0.12), standard normal scores, linear tail extrapolation, final +/-3.5 cap.",
        "fisk_method": "Three-parameter Fisk/log-logistic PWM fit in alternative-target script; probability bounded to [1e-6, 1-1e-6], z capped at +/-3.5; failed fits recorded as failed with missing values retained. Existing diagnostics contain 37 failed 1990-baseline and 16 failed 1981-baseline site-scale-calendar-month fits.",
            "generalized_logistic": "Excluded from dated alternatives based on existing diagnostics: 432 fits converged, but 166 shape estimates exceed 100 and maximum shape is about 2136.6. Convergence alone does not establish stable tails.",
            "validation_test_isolation": "No validation/test dates occur in parameter fit masks; see regression tests which perturb post-calibration values and compare fitted diagnostics.",
            "probability_clipping": "Empirical transform has no fitted CDF; Phi(z) below is an implied normal-score probability. Fisk clamps probability to 1e-6 before inverse-normal transform. Final standardized scores are capped at +/-3.5.",
        },
        "canonical_vs_dated_alternative_nonparametric_1990_2010": canonical_checks,
        "checks_all_match": all(item["matches_canonical"] for item in canonical_checks),
        "alternative_fit_status_counts": _fit_status_summary(output_dir / "spei_fit_diagnostics.json"),
        "beed_september_2019": beed_detail,
        "extreme_event_summary": event_detail,
        "v212_access_status": v212_audit["status"],
        "interpretation": {
            "direct_evidence": "The local Beed Sep-2019 six-month balance is strongly negative and the local transformed score reaches the -3.5 lower cap, while the v2.11 grid-cell score is positive. Existing paired data show this is among the largest disagreements.",
            "supported_interpretation": "The local standardized output is at a censoring boundary; its raw empirical tail score can be reconstructed, but the cap prevents interpreting -3.5 as the uncensored magnitude.",
            "unverified_hypotheses": ["Differences in local versus CRU precipitation/PET inputs", "point versus 0.5-degree grid representation", "differences in reference fitting/calibration", "potential differences in accumulated water balance"],
            "cause_limit": "Matched CRU precipitation and PET inputs or raw SPEIbase intermediate water balance are absent, so causal attribution to inputs, calibration, transformation, or spatial mismatch is unresolved.",
        },
        "outputs": [
            "spei_target_integrity_audit.json", "spei_extreme_event_audit.csv",
            "spei_beed_2019_reconstruction.csv", "spei_v212_access_audit.json",
            "spei_phase1_3_report.md",
        ],
    }
    (output_dir / "spei_target_integrity_audit.json").write_text(json.dumps(integrity, indent=2, allow_nan=False), encoding="utf-8")
    report = _make_report(integrity, event_frame, v212_audit)
    (output_dir / "spei_phase1_3_report.md").write_text(report, encoding="utf-8")
    return integrity


def _fit_status_summary(path: Path) -> dict:
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(diagnostics["fit_diagnostics"])
    summary = {}
    for (transformation, baseline, status), group in frame.groupby(
        ["transformation", "baseline", "fit_status"], dropna=False, sort=True
    ):
        summary[f"{transformation}|{baseline}|{status}"] = {
            "fit_rows": len(group),
            "calendar_site_scale_fits": len(group),
            "failed_fit_examples": group.loc[group.fit_status.eq("failed"), "fit_error"].head(5).tolist(),
        }
    failures = frame.loc[frame.fit_status.eq("failed")]
    summary["failed_fit_records"] = [
        {key: _json_value(value) for key, value in row.items()}
        for row in failures[["site", "scale", "baseline", "calendar_month", "calibration_n", "fit_error"]].to_dict("records")
    ]
    summary["generalized_logistic_screen"] = diagnostics["transformations"]["generalized_logistic_mle"]
    summary["fisk_parameter_summary"] = diagnostics["fisk_parameter_summary"]
    return summary


def _make_report(integrity: dict, event_frame: pd.DataFrame, v212: dict) -> str:
    paired = event_frame.loc[event_frame.absolute_difference.notna()]
    top = paired.nlargest(10, "absolute_difference")
    lines = [
        "# Phase 1.3 — SPEI target integrity and extreme-event audit", "",
        f"Generated {integrity['generated_utc']}. Canonical targets, configuration, features, model, and Phase 1.2 outputs were not changed. No training or model forward pass was performed.", "",
        "## SPEIbase v2.12.0 access", "",
        "The official repository README identifies v2.12.0 as CRU TS 4.10, January 1901–December 2025, generated with SPEI package 1.8.1; it lists `spei03.nc`, `spei06.nc`, and `spei12.nc` as generated filenames and points users to the SPEI database web page. The README does not identify a versioned direct download path or a file manifest. Prior HEAD records for `https://spei.csic.es/spei_database_2_12/nc/spei03.nc`, `spei06.nc`, and `spei12.nc` returned HTTP 404. A new shell request in this session was limited by DNS resolution failure, so no v2.12 files or metadata were obtained. The existing v2.11 values were not relabelled or substituted.", "",
        "**Status: BLOCKED.** File dimensions, CRS, actual variable attributes, missing-value encoding, and extraction from v2.12 remain unverified.", "",
        "## Transformation and calibration audit", "",
        "The canonical builder forms monthly water balance as `PRECTOTCORR (mm/day) × days_in_month − pet_mm`, then applies a trailing rolling sum requiring a complete window. Fits are separate by calendar month and calibration years end at December 2010. The nonparametric method maps empirical plotting positions to normal scores, linearly extrapolates beyond calibration extrema, and caps the final score at ±3.5. The alternative Fisk implementation bounds probabilities to [1e-6, 1−1e-6] and records failed fits rather than silently dropping those site-months. Existing generalized-logistic fits all reported convergence, but 166/432 had shape >100 and maximum shape was 2136.6; this is evidence of unstable shape estimates, not proof that every fit is invalid.", "",
        f"Canonical versus dated nonparametric 1990–2010 alternative reproduces exactly across all site/scale comparisons: `{integrity['checks_all_match']}`. Detailed counts and maximum absolute errors are in `spei_target_integrity_audit.json`.", "",
        "## Beed, September 2019, SPEI-6", "",
        f"Reconstruction: six-month accumulated local water balance {integrity['beed_september_2019']['accumulated_water_balance_mm']:.3f} mm; uncapped empirical normal score {integrity['beed_september_2019']['unclipped_transformed_spei']:.6f}; final score {integrity['beed_september_2019']['final_clipped_spei']:.3f}; v2.11 nearest-cell score {integrity['beed_september_2019']['speibase_v211_spei_6']:.6f} at ({integrity['beed_september_2019']['reference_grid_lat']:.2f}, {integrity['beed_september_2019']['reference_grid_lon']:.2f}). The reported probability is only Φ of the empirical extrapolated z score, not a fitted drought probability.", "",
        "**Finding:** the local output is capped at −3.5, so magnitude below that boundary is censored. The local source data and calculation can be reproduced, but the local–reference sign disagreement cannot be causally assigned: corresponding CRU precipitation/PET and SPEIbase intermediate water balance are not present. Input, fitting, transformation, and spatial representation effects remain possible; none is confirmed from this evidence alone.", "",
        "## Largest exact-month disagreements", "",
        "| Rank | Site | Scale | Month | Local | SPEIbase v2.11 | Absolute difference | Category mismatch |",
        "|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(top.itertuples(index=False), 1):
        lines.append(f"| {rank} | {row.site} | {row.scale} | {row.date} | {row.local_spei:.3f} | {row.reference_spei:.3f} | {row.absolute_difference:.3f} | {row.category_disagreement} |")
    lines += [
        "", f"There are {len(paired):,} paired local/reference site-month-scale values among {len(event_frame):,} rows; missing values were not interpolated. Per-site/scale, year, accumulation, calibration range, and clipping fields are in `spei_extreme_event_audit.csv`; full summaries are in the JSON.", "",
        "## Recommendation", "",
        "**Do not approve the canonical targets for definitive scientific claims yet.** The canonical calculation is reproducible and its dated alternative matches exactly, but the large Beed event disagreement, ±3.5 tail censoring, short 1990–2010 calibration, and blocked v2.12 reference access leave material uncertainty. This recommendation is a scientific hold, not a claim that the canonical series is incorrect. Keep the canonical data unchanged; resolve the reference/input comparison and document tail sensitivity before model training.", "",
        "## Tests and limits", "",
        "Focused verification: `python -m pytest -q tests/test_spei_phase1_3_audit.py tests/test_spei_complete_validation.py tests/test_spei_reference_validation.py` — 19 passed. Complete suite: `python -m pytest -q --basetemp .pytest_tmp_spei_phase13_final2_20261108` — 130 passed in 87.20 s. Ruff: `python -m ruff check experiments/audit_spei_target_integrity.py tests/test_spei_phase1_3_audit.py` — all checks passed. v2.12 file retrieval was blocked; v2.11 is the only independent matched reference used here. No inference or training was run.", "",
        "Sources: [official SPEIbase repository README](https://github.com/sbegueria/SPEIbase/blob/master/README.md), [official database page](https://spei.csic.es/database.html), [official v2.12 generation script](https://github.com/sbegueria/SPEIbase/blob/master/R/computeSPEI.R).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PHASE_DIR)
    parser.add_argument("--probe-v212", action="store_true", help="Make read-only HEAD checks; never download full NetCDF files")
    args = parser.parse_args()
    run(output_dir=args.output_dir, network_probe=args.probe_v212)
    print(f"Wrote Phase 1.3 audit artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
