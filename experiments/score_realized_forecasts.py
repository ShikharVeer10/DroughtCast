"""Score ensemble probabilities against realized 2025 observed SPEI."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.probability import brier_score, reliability_table


SPEI_COLUMNS = {"SPEI_3": "spei_3_obs", "SPEI_6": "spei_6_obs", "SPEI_12": "spei_12_obs"}


def main() -> None:
    dataset_path = Path("data/monthly_sites.csv")
    data = pd.read_csv(dataset_path, parse_dates=["date"])
    if data.date.max() < pd.Timestamp("2025-04-01"):
        print(".\\.venv\\Scripts\\python.exe -m data.build_dataset --start 1981 --end 2025")
        raise SystemExit(1)

    predictions_path = Path("results/inference_probabilities_ensemble.csv")
    if not predictions_path.is_file():
        raise FileNotFoundError("Run .\\.venv\\Scripts\\python.exe experiments\\build_probability_ensemble.py first")
    predictions = pd.read_csv(predictions_path, parse_dates=["target_month"])
    prediction_col = "calibrated_probability_mean"
    required = {"site", "target_month", "scale", prediction_col}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Missing ensemble columns: {sorted(missing)}")

    actual_rows = []
    for scale, column in SPEI_COLUMNS.items():
        actual_rows.append(data[["site", "date", column]].rename(columns={"date": "target_month", column: "realized_spei"}).assign(scale=scale))
    actual = pd.concat(actual_rows, ignore_index=True)
    scored = predictions.merge(actual, on=["site", "target_month", "scale"], how="inner", validate="one_to_one")
    scored = scored.loc[scored.target_month.between("2025-01-01", "2025-04-01")].copy()
    if len(scored) != 8 * 4 * len(SPEI_COLUMNS):
        raise ValueError(f"Expected 96 site-month-scale rows, found {len(scored)}")
    scored["event"] = (scored.realized_spei <= -1.0).astype(int)

    train = data.loc[data.date <= pd.Timestamp("2010-12-01")]
    base_rates = {
        scale: float((train[column] <= -1.0).mean())
        for scale, column in SPEI_COLUMNS.items()
    }
    report_rows = []
    overall_train_events = pd.concat(
        [(train[column] <= -1.0) for column in SPEI_COLUMNS.values()],
        ignore_index=True,
    )
    base_rates["OVERALL"] = float(overall_train_events.mean())
    for label, group in [*scored.groupby("scale", sort=True), ("OVERALL", scored)]:
        base_rate = base_rates[label]
        model_brier = brier_score(group[prediction_col].to_numpy(), group.event.to_numpy())
        base_brier = brier_score([base_rate] * len(group), group.event.to_numpy())
        skill = 1 - model_brier / base_brier if base_brier > 0 else float("nan")
        print(f"{label}: n={len(group)}, model Brier={model_brier:.6f}, base-rate Brier={base_brier:.6f}, Brier skill={skill:.6f}")
        print(f"{label} reliability_table: {reliability_table(group[prediction_col].to_numpy(), group.event.to_numpy())}")
        report_rows.append({"scale": label, "n": len(group), "base_rate": base_rate,
                            "model_brier": model_brier, "base_rate_brier": base_brier,
                            "brier_skill_score": skill})
    output = Path("results/realized_scoring_2025.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    pd.DataFrame(report_rows).to_csv(output.with_name("realized_scoring_2025_summary.csv"), index=False)
    if any(row["brier_skill_score"] <= 0 for row in report_rows):
        print("Probabilities are not validated: Brier skill is not positive for every reported group.")
    else:
        print("Brier skill is positive for every reported group; results are evaluated on the specified realized period.")
    print(f"Saved joined realized forecasts to {output}")


if __name__ == "__main__":
    main()
