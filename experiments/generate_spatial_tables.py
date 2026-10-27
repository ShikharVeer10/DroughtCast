"""Generate conference-ready CSV and JSON tables from spatial pipeline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_tables(
    evaluation_json_path: Path | str,
    output_dir: Path | str = "results/spatial_pipeline/tables",
) -> None:
    eval_p = Path(evaluation_json_path)
    if not eval_p.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {eval_p}")

    report = json.loads(eval_p.read_text(encoding="utf-8"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spatial Continuous Metrics Summary
    rows_metrics = []
    for scale, metrics in report.get("per_scale", {}).items():
        ci_rmse = metrics.get("rmse_ci95", [float("nan"), float("nan")])
        ci_mae = metrics.get("mae_ci95", [float("nan"), float("nan")])
        rows_metrics.append(
            {
                "Scale": scale.upper().replace("_", "-"),
                "N_Samples": metrics.get("n_samples"),
                "RMSE": f"{metrics.get('rmse'):.4f}",
                "RMSE_95CI": f"[{ci_rmse[0]:.4f}, {ci_rmse[1]:.4f}]"
                if np.isfinite(ci_rmse[0])
                else "N/A",
                "MAE": f"{metrics.get('mae'):.4f}",
                "MAE_95CI": f"[{ci_mae[0]:.4f}, {ci_mae[1]:.4f}]"
                if np.isfinite(ci_mae[0])
                else "N/A",
                "Bias": f"{metrics.get('bias'):.4f}",
                "Correlation": f"{metrics.get('correlation'):.4f}"
                if np.isfinite(metrics.get("correlation", np.nan))
                else "N/A",
                "NSE": f"{metrics.get('nse'):.4f}"
                if np.isfinite(metrics.get("nse", np.nan))
                else "N/A",
                "KGE": f"{metrics.get('kge'):.4f}"
                if np.isfinite(metrics.get("kge", np.nan))
                else "N/A",
            }
        )
    df_metrics = pd.DataFrame(rows_metrics)
    df_metrics.to_csv(out_dir / "spatial_metrics_summary.csv", index=False)

    # 2. Drought Contingency Scores Summary
    rows_contingency = []
    for scale, thresholds in report.get("thresholds", {}).items():
        for thresh_key, ev in thresholds.items():
            th_val = ev.get("threshold")
            rows_contingency.append(
                {
                    "Scale": scale.upper().replace("_", "-"),
                    "Threshold": th_val,
                    "Regime": "Moderate"
                    if th_val == -1.0
                    else ("Severe" if th_val == -1.5 else "Extreme"),
                    "TP": ev.get("tp"),
                    "FP": ev.get("fp"),
                    "FN": ev.get("fn"),
                    "TN": ev.get("tn"),
                    "POD": f"{ev.get('pod'):.4f}"
                    if np.isfinite(ev.get("pod", np.nan))
                    else "N/A",
                    "FAR": f"{ev.get('far'):.4f}"
                    if np.isfinite(ev.get("far", np.nan))
                    else "N/A",
                    "CSI": f"{ev.get('csi'):.4f}"
                    if np.isfinite(ev.get("csi", np.nan))
                    else "N/A",
                    "F1": f"{ev.get('f1'):.4f}"
                    if np.isfinite(ev.get("f1", np.nan))
                    else "N/A",
                }
            )
    df_contingency = pd.DataFrame(rows_contingency)
    df_contingency.to_csv(out_dir / "drought_detection_summary.csv", index=False)

    # 3. Probabilistic & Conformal Calibration Summary
    rows_prob = []
    for scale, p_info in report.get("probabilistic", {}).items():
        cal = p_info.get("calibrated_event_probability", {})
        cqr = p_info.get("cqr_interval_metrics", {})
        q_score = p_info.get("quantile_scoring", {})
        rows_prob.append(
            {
                "Scale": scale.upper().replace("_", "-"),
                "Brier_Score": f"{cal.get('brier_score'):.4f}"
                if cal.get("brier_score") is not None
                else "N/A",
                "BSS_vs_Climatology": f"{cal.get('brier_skill_score_vs_climatology'):.4f}"
                if cal.get("brier_skill_score_vs_climatology") is not None
                else "N/A",
                "ECE": f"{cal.get('expected_calibration_error'):.4f}"
                if cal.get("expected_calibration_error") is not None
                else "N/A",
                "Calibration_Slope": f"{cal.get('calibration_slope'):.4f}"
                if cal.get("calibration_slope") is not None
                else "N/A",
                "Calibration_Intercept": f"{cal.get('calibration_intercept'):.4f}"
                if cal.get("calibration_intercept") is not None
                else "N/A",
                "Nominal_Coverage": f"{cqr.get('nominal_coverage', 0.8):.2f}",
                "Empirical_Coverage": f"{cqr.get('empirical_coverage'):.4f}"
                if cqr.get("empirical_coverage") is not None
                else "N/A",
                "Mean_Interval_Width": f"{cqr.get('mean_width'):.4f}"
                if cqr.get("mean_width") is not None
                else "N/A",
                "CRPS": f"{q_score.get('crps'):.4f}"
                if q_score.get("crps") is not None
                else "N/A",
                "Pinball_Loss": f"{q_score.get('pinball_loss'):.4f}"
                if q_score.get("pinball_loss") is not None
                else "N/A",
            }
        )
    df_prob = pd.DataFrame(rows_prob)
    df_prob.to_csv(out_dir / "probabilistic_calibration_summary.csv", index=False)

    # 4. Cell Counts & Sample Sizes
    df_counts = pd.DataFrame([report.get("cell_counts", {})])
    df_counts.to_csv(out_dir / "cell_counts.csv", index=False)

    print(f"Generated paper tables in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper tables from spatial evaluation"
    )
    parser.add_argument("evaluation_json", help="Path to spatial evaluation JSON")
    parser.add_argument("--output-dir", default="results/spatial_pipeline/tables")
    args = parser.parse_args()

    generate_tables(args.evaluation_json, args.output_dir)


if __name__ == "__main__":
    main()
