"""Compile 5-row systematic ablation table quantifying exact metric degradation."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from data.protocol import DATASET_PROTOCOL_VERSION
from utils.artifacts import load_checkpoint

SCALES = ["spei_3", "spei_6", "spei_12"]
WINDOW_VARIANTS = {"window_3", "window_6", "window_12", "window_24"}


def common_target_rmse(root_dir: Path, variants: list[str]) -> pd.DataFrame:
    """Compute per-seed window ablation RMSE on shared site/target/scale/lead rows."""
    summaries = []
    for seed in sorted(
        {
            int(path.stem.rsplit("_", 1)[1])
            for variant in variants
            for path in (root_dir / variant).glob("test_predictions_seed_*.csv")
        }
    ):
        by_variant = {}
        for variant in variants:
            path = root_dir / variant / f"test_predictions_seed_{seed}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing common-target predictions for {variant}, seed {seed}: {path}")
            frame = pd.read_csv(path)
            required = {"site", "target_month", "scale", "lead", "observed", "prediction_droughtcast"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} missing prediction columns: {sorted(missing)}")
            keys = ["site", "target_month", "scale", "lead"]
            if frame.duplicated(keys).any():
                raise ValueError(f"{path} has duplicate site/target/scale/lead rows")
            by_variant[variant] = frame.set_index(keys)[["observed", "prediction_droughtcast"]]
        if not by_variant:
            continue
        common = set.intersection(*(set(frame.index) for frame in by_variant.values()))
        if not common:
            raise ValueError(f"No common test targets across window variants for seed {seed}")
        common_index = pd.MultiIndex.from_tuples(sorted(common), names=["site", "target_month", "scale", "lead"])
        observed_reference = None
        for variant, frame in by_variant.items():
            aligned = frame.loc[common_index]
            if observed_reference is None:
                observed_reference = aligned["observed"]
            elif not np.allclose(aligned["observed"], observed_reference, equal_nan=True):
                raise ValueError(f"Observed targets differ across variants for seed {seed}")
            error = aligned["prediction_droughtcast"].to_numpy() - aligned["observed"].to_numpy()
            for (site, target_month, scale, lead), value in zip(aligned.index, error):
                summaries.append({"variant": variant, "seed": seed, "site": site, "target_month": target_month, "scale": scale, "lead": lead, "squared_error": float(value ** 2), "n_common_rows": len(aligned)})
    return pd.DataFrame(summaries)


def summarize_ablations(root_dir: Path):
    variants = sorted(
        [
            p.name
            for p in root_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
    )
    if not variants:
        print("No ablation variant directories found.")
        return

    window_variants = sorted(set(variants) & WINDOW_VARIANTS)
    if len(window_variants) > 1:
        aligned = common_target_rmse(root_dir, window_variants)
        aligned = aligned.groupby(["variant", "seed", "site", "target_month", "scale", "lead"], as_index=False).squared_error.mean()
        aligned.to_csv(root_dir / "window_ablation_common_targets.csv", index=False)
        aligned.groupby(["variant", "seed"], as_index=False).agg(
            rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            n_common_rows=("squared_error", "size"),
        ).to_csv(root_dir / "window_ablation_common_rmse.csv", index=False)
        print(f"Saved common-target window ablation metrics to {root_dir / 'window_ablation_common_targets.csv'}")

    aligned_rmse = {}
    if len(window_variants) > 1:
        target_errors = pd.read_csv(root_dir / "window_ablation_common_targets.csv")
        aligned_rmse = {
            variant: group.groupby(["seed", "scale", "lead"]).squared_error.mean()
            .pow(0.5).groupby("seed").mean().tolist()
            for variant, group in target_errors.groupby("variant")
        }

    table_rows = []
    for var in variants:
        var_dir = root_dir / var
        reports = list(var_dir.glob("metrics_seed_*.json"))
        if not reports:
            continue

        rmse_list, mae_list, kge_list, nse_list, skill_list = [], [], [], [], []
        for rep_path in reports:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            dcast = rep.get("methods", {}).get("droughtcast", {})
            for scale in SCALES:
                for lead in ["T+1", "T+2", "T+3", "T+4"]:
                    if scale in dcast and lead in dcast[scale]:
                        info = dcast[scale][lead]
                        if var not in WINDOW_VARIANTS or len(window_variants) <= 1:
                            rmse_list.append(info.get("rmse", np.nan))
                        mae_list.append(info.get("mae", np.nan))
                        kge_list.append(info.get("kge_2012", np.nan))
                        nse_list.append(info.get("nse", np.nan))
                        skill_list.append(info.get("skill_vs_persistence", np.nan))

        if var in aligned_rmse:
            rmse_list = aligned_rmse[var]
        table_rows.append(
            {
                "variant": var,
                "n_seeds": len(reports),
                "rmse_mean": float(np.nanmean(rmse_list)),
                "rmse_std": float(np.nanstd(rmse_list)),
                "mae_mean": float(np.nanmean(mae_list)),
                "mae_std": float(np.nanstd(mae_list)),
                "kge_mean": float(np.nanmean(kge_list)),
                "nse_mean": float(np.nanmean(nse_list)),
                "skill_mean": float(np.nanmean(skill_list)),
            }
        )

    df = pd.DataFrame(table_rows)
    if "full_model" in df["variant"].values:
        baseline_rmse = df.loc[df["variant"] == "full_model", "rmse_mean"].values[0]
        baseline_kge = df.loc[df["variant"] == "full_model", "kge_mean"].values[0]
        df["delta_rmse"] = df["rmse_mean"] - baseline_rmse
        df["delta_kge"] = df["kge_mean"] - baseline_kge

    print(
        "\n================ SYSTEMATIC ABLATION TABLE (OVERALL 4-STEP POOLED) ================"
    )
    print(df.to_string(index=False))
    df.to_csv(root_dir / "ablation_summary.csv", index=False)
    # Also write LaTeX table for direct inclusion in manuscript
    latex_path = root_dir / "ablation_table.tex"
    df.to_latex(latex_path, index=False, float_format="%.4f")
    print(f"\nSaved LaTeX ablation table to {latex_path}")
    if root_dir.name == "ablations":
        _write_ablation_forest(root_dir)
    elif root_dir.name == "region_holdouts":
        _write_leave_region_out(root_dir)


def _current_report(report_path: Path, checkpoint_path: Path, data_hash: str):

    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if (
        report.get("dataset_protocol", {}).get("schema_version")
        != DATASET_PROTOCOL_VERSION
        or checkpoint.get("artifact_metadata", {}).get("dataset_sha256") != data_hash
    ):
        raise ValueError(f"Refusing stale/non-schema-v2 result: {report_path}")
    return report


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def _write_ablation_forest(root_dir: Path):
    manifest = json.loads((root_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_path = Path(manifest.get("dataset_path", "data/monthly_sites.csv"))
    data_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    base_dir = root_dir / "full_model"
    base_paths = sorted(base_dir.glob("metrics_seed_*.json"))
    if len(base_paths) < 5:
        print(
            "Blocked ablation forest export: five current full_model/base_diagram seeds are required"
        )
        return
    manifest_path = root_dir / "manifest.json"
    requested_variants = (
        set(json.loads(manifest_path.read_text(encoding="utf-8")).get("variants", []))
        if manifest_path.is_file()
        else set()
    )
    missing_variants = [
        variant
        for variant in requested_variants - {"full_model"}
        if len(list((root_dir / variant).glob("metrics_seed_*.json"))) < 5
    ]
    if missing_variants:
        print(
            f"Blocked ablation forest export: five current seeds are missing for {sorted(missing_variants)}"
        )
        return
    base_reports = {}
    for report_path in base_paths:
        seed = int(report_path.stem.rsplit("_", 1)[1])
        base_reports[seed] = _current_report(
            report_path, base_dir / f"model_seed_{seed}.pt", data_hash
        )
    variants = sorted(
        path
        for path in root_dir.iterdir()
        if path.is_dir() and path.name != "full_model"
    )
    rows = []
    variants = [base_dir, *variants]
    for folder in variants:
        run_name = folder.name
        by_seed = {}
        for report_path in folder.glob("metrics_seed_*.json"):
            seed = int(report_path.stem.rsplit("_", 1)[1])
            if seed not in base_reports:
                continue
            report = _current_report(
                report_path, folder / f"model_seed_{seed}.pt", data_hash
            )
            deltas = []
            for scale in SCALES:
                for lead in ("T+1", "T+2", "T+3", "T+4"):
                    current = (
                        report.get("methods", {})
                        .get("droughtcast", {})
                        .get(scale, {})
                        .get(lead, {})
                        .get("rmse")
                    )
                    reference = (
                        base_reports[seed]
                        .get("methods", {})
                        .get("droughtcast", {})
                        .get(scale, {})
                        .get(lead, {})
                        .get("rmse")
                    )
                    if current is None or reference is None:
                        raise ValueError(
                            f"Missing test RMSE for {run_name}, seed {seed}, {scale}, {lead}"
                        )
                    deltas.append(float(current - reference))
            by_seed[seed] = float(np.mean(deltas))
        if len(by_seed) < 5:
            continue
        mean, lower, upper = _mean_ci(list(by_seed.values()))
        rows.append(
            {
                "variant": "base_diagram" if run_name == "full_model" else run_name,
                "delta_rmse": mean,
                "lower": lower,
                "upper": upper,
                "n_seeds": len(by_seed),
                "seeds": ",".join(map(str, sorted(by_seed))),
                "aggregation": "mean delta RMSE across 3 SPEI accumulation scales and T+1..T+4 per seed",
            }
        )
    frame = pd.DataFrame(rows)
    if "base_diagram" not in set(frame.variant):
        raise RuntimeError(
            "Ablation forest export is missing the base_diagram reference"
        )
    output = Path("results/ablation_forest.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote current five-seed ablation forest data to {output}")


def _write_leave_region_out(root_dir: Path):
    manifest_path = root_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    dataset_path = Path(manifest.get("dataset_path", "data/monthly_sites.csv"))
    data_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    rows = []
    for folder in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        for report_path in sorted(folder.glob("metrics_seed_*.json")):
            seed = int(report_path.stem.rsplit("_", 1)[1])
            report = _current_report(
                report_path, folder / f"model_seed_{seed}.pt", data_hash
            )
            droughtcast = report.get("methods", {}).get("droughtcast", {})
            for scale in SCALES:
                for lead, metric in droughtcast.get(scale, {}).items():
                    if lead.startswith("T+") and "skill_vs_persistence" in metric:
                        rows.append(
                            {
                                "heldout_region": folder.name,
                                "seed": seed,
                                "scale": scale,
                                "lead": lead,
                                "skill_vs_persistence": float(
                                    metric["skill_vs_persistence"]
                                ),
                            }
                        )
    frame = pd.DataFrame(rows)
    if (
        frame.empty
        or (frame.groupby(["heldout_region", "scale", "lead"]).seed.nunique() < 5).any()
    ):
        print(
            "Blocked leave-region-out export: five current seeds per region, scale, and lead are required"
        )
        return
    output = Path("results/leave_region_out_skill.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote current leave-region-out skill data to {output}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/ablations")
    summarize_ablations(path)
