"""Export calibrated site drought probabilities as a compact paper table.

All values are read from the inference CSV; this module does not recalculate,
round for inference, or synthesize forecasts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = {
    "site",
    "target_month",
    "scale",
    "lead",
    "drought_threshold_spei",
    "calibrated_probability",
}
LEAD_ORDER = ["T+1", "T+2", "T+3", "T+4"]


def build_table(input_path: Path) -> tuple[pd.DataFrame, list[str], float]:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Probability results are missing: {input_path}. Run `python inference.py --site all` first."
        )
    raw = pd.read_csv(input_path)
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError(f"Probability results contain no rows: {input_path}")
    if raw[list(REQUIRED)].isna().any().any():
        raise ValueError(
            f"Probability results contain missing required values: {input_path}"
        )
    thresholds = raw.drought_threshold_spei.unique()
    if len(thresholds) != 1:
        raise ValueError(
            "One table must contain probabilities for exactly one drought threshold"
        )

    month_map: dict[str, str] = {}
    for lead, group in raw.groupby("lead", sort=False):
        months = group.target_month.astype(str).unique()
        if len(months) != 1:
            raise ValueError(f"Target month is inconsistent for {lead}: {list(months)}")
        month_map[lead] = months[0]
    if set(month_map) != set(LEAD_ORDER):
        raise ValueError(f"Expected leads {LEAD_ORDER}, found {sorted(month_map)}")

    counts = raw.groupby(["site", "scale"]).lead.nunique()
    if (counts != len(LEAD_ORDER)).any() or raw.duplicated(
        ["site", "scale", "lead"]
    ).any():
        raise ValueError(
            "Each site and accumulation scale must have exactly one probability for each T+1..T+4 lead"
        )
    table = raw.pivot(
        index=["site", "scale"], columns="lead", values="calibrated_probability"
    )
    table = table.reindex(columns=LEAD_ORDER).reset_index()
    table["scale"] = table.scale.str.replace("SPEI_", "SPEI-", regex=False)
    table["scale"] = pd.Categorical(
        table.scale, categories=["SPEI-3", "SPEI-6", "SPEI-12"], ordered=True
    )
    table = table.sort_values(["site", "scale"], kind="stable").reset_index(drop=True)
    table = table.rename(
        columns={lead: f"{lead} ({month_map[lead]})" for lead in LEAD_ORDER}
    )
    return table, [month_map[lead] for lead in LEAD_ORDER], float(thresholds[0])


def render_latex(
    table: pd.DataFrame, target_months: list[str], threshold: float
) -> str:
    leads = [column for column in table.columns if column.startswith("T+")]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Validation-calibrated probability of drought, defined as SPEI at or below "
        + f"{threshold:g}. Entries are decimal probabilities on [0, 1], shown by configured site, SPEI accumulation scale, and forecast lead. "
        + r"The model uses the six latest available monthly inputs; calibration uses the validation split.}",
        r"\label{tab:calibrated-drought-probabilities}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        "Site & Accumulation scale & " + " & ".join(leads) + r" \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False, name=None):
        site, scale, *probabilities = row
        site_label = str(site).replace("_", r"\_")
        cells = [f"{float(value):.4f}" for value in probabilities]
        lines.append(f"{site_label} & {scale} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.96\textwidth}\footnotesize",
            "Entries are calibrated event probabilities from the saved quantile checkpoint; they are not binary classifications. "
            "The validation-fitted isotonic map is piecewise constant, so tied probabilities across sites can occur. "
            f"Forecast target months are {', '.join(target_months)}. "
            "Sites are the locations configured in this dataset, not a complete state-level coverage.\\\\",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_terminal(table: pd.DataFrame, threshold: float) -> str:
    """Render a fixed-width table so terminals cannot wrap or hide columns."""
    columns = list(table.columns)
    widths = {
        column: max(
            len(column), *(len(f"{float(value):.4f}") for value in table[column])
        )
        if column not in {"site", "scale"}
        else max(len(column), *(len(str(value)) for value in table[column]))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    rule = "-+-".join("-" * widths[column] for column in columns)
    rows = [header, rule]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, row):
            text = str(value) if column in {"site", "scale"} else f"{float(value):.4f}"
            cells.append(text.ljust(widths[column]))
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/inference_probabilities_seed_42.csv"),
    )
    parser.add_argument(
        "--latex-out", type=Path, default=Path("paper/latex/probability_table.tex")
    )
    parser.add_argument(
        "--csv-out", type=Path, default=Path("paper/latex/probability_table.csv")
    )
    args = parser.parse_args()
    table, months, threshold = build_table(args.input)
    args.latex_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.latex_out.write_text(render_latex(table, months, threshold), encoding="utf-8")
    table.to_csv(args.csv_out, index=False, float_format="%.8f")
    print(
        f"Event: SPEI <= {threshold:g} | Values are calibrated probabilities in [0, 1]"
    )
    print(
        "Each row is one configured site and accumulation scale; each lead includes its target month."
    )
    print(render_terminal(table, threshold))
    print(f"\nLaTeX table: {args.latex_out}")
    print(f"Machine-readable table: {args.csv_out}")


if __name__ == "__main__":
    main()
