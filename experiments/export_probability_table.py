
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import pandas as pd

REQUIRED = {
    "site",
    "target_month",
    "scale",
    "lead",
    "drought_threshold_spei",
}
LEAD_ORDER = ["T+1", "T+2", "T+3", "T+4"]


def format_probability(value: float) -> str:
    probability = float(value)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Probability must be strictly between 0 and 1; got {probability!r}")
    # Fixed-point Decimal output avoids scientific notation and preserves repr's
    # round-trip digits, so distinct computed values stay distinct in the table.
    return format(Decimal(repr(probability)), "f")


def format_scale_label(value: str) -> str:
    return str(value).upper().replace("SPEI_", "SPEI-")


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
    probability_column = (
        "calibrated_probability_mean"
        if "calibrated_probability_mean" in raw
        else "calibrated_probability"
    )
    if probability_column not in raw:
        raise ValueError(f"{input_path} needs calibrated_probability or calibrated_probability_mean")
    if raw[list(REQUIRED)].isna().any().any() or raw[probability_column].isna().any():
        raise ValueError(
            f"Probability results contain missing required values: {input_path}"
        )
    if not raw[probability_column].map(lambda value: 0.0 < float(value) < 1.0).all():
        raise ValueError("All probabilities must be strictly between 0 and 1")
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
    table = raw.pivot(index=["site", "scale"], columns="lead", values=probability_column)
    table = table.reindex(columns=LEAD_ORDER).reset_index()
    table["scale"] = table.scale.map(format_scale_label)
    table["scale"] = pd.Categorical(
        table.scale, categories=["SPEI-3", "SPEI-6", "SPEI-12"], ordered=True
    )
    table = table.sort_values(["site", "scale"], kind="stable").reset_index(drop=True)
    table = table.rename(
        columns={lead: f"{lead} ({month_map[lead]})" for lead in LEAD_ORDER}
    )
    table.attrs["ensemble"] = probability_column.endswith("_mean")
    table.attrs["n_seeds"] = int(raw.n_seeds.min()) if "n_seeds" in raw else 1
    calibration_labels = raw["calibration"].dropna().astype(str).unique() if "calibration" in raw else []
    table.attrs["calibration_description"] = (
        calibration_labels[0] if len(calibration_labels) == 1 else "validation-only Platt scaling"
    )
    if table.attrs["ensemble"]:
        ranges = raw.copy()
        ranges["range"] = ranges.apply(
            lambda row: f"{format_probability(row.calibrated_probability_min)} - {format_probability(row.calibrated_probability_max)}",
            axis=1,
        )
        range_table = ranges.pivot(index=["site", "scale"], columns="lead", values="range")
        range_table = range_table.reindex(columns=LEAD_ORDER).reset_index()
        range_table["scale"] = range_table.scale.map(format_scale_label)
        range_table = range_table.sort_values(["site", "scale"], kind="stable").reset_index(drop=True)
        range_table = range_table.rename(columns={lead: f"{lead} ({month_map[lead]})" for lead in LEAD_ORDER})
        table.attrs["range_table"] = range_table
    return table, [month_map[lead] for lead in LEAD_ORDER], float(thresholds[0])


def render_latex(
    table: pd.DataFrame, target_months: list[str], threshold: float
) -> str:
    leads = [column for column in table.columns if column.startswith("T+")]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Validation-calibrated probability of drought, defined as SPEI at or below "
        + f"{threshold:g}. {table.attrs['calibration_description']}, {table.attrs.get('n_seeds', 1)} seed(s); probabilities are estimates, not guarantees.",
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
        cells = [format_probability(value) for value in probabilities]
        lines.append(f"{site_label} & {scale} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.96\textwidth}\footnotesize",
            (
                "Entries are calibrated event probabilities from the saved quantile checkpoint; they are not binary classifications. "
                + f"Forecast target months are {', '.join(target_months)}. "
            "Sites are the locations configured in this dataset, not a complete state-level coverage.\\\\",
            ),
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    if "range_table" in table.attrs:
        lines.extend([r"\paragraph{Across-seed minimum--maximum ranges.}", r"\begin{tabular}{llrrrr}", r"\toprule", "Site & Accumulation scale & " + " & ".join(leads) + r" \\", r"\midrule"])
        for row in table.attrs["range_table"].itertuples(index=False, name=None):
            site, scale, *ranges = row
            site_label = str(site).replace("_", r"\_")
            lines.append(f"{site_label} & {scale} & " + " & ".join(ranges) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_terminal(table: pd.DataFrame, threshold: float) -> str:
    """Render a fixed-width table so terminals cannot wrap or hide columns."""
    month_labels = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
        "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
        "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
    }
    columns = [
        "Site" if column == "site" else "Scale" if column == "scale"
        else f"{column.split(' ')[0]} - {month_labels[column.rsplit('-', 1)[1][:2]]} {column.split('(')[1][:4]}"
        for column in table.columns
    ]
    display_rows = [
        [str(value).replace("_", " ").title() if column == "site"
         else format_scale_label(value) if column == "scale"
         else str(value) if isinstance(value, str) else format_probability(value)
         for column, value in zip(table.columns, row)]
        for row in table.itertuples(index=False, name=None)
    ]
    widths = {
        column: max(len(column), *(len(row[index]) for row in display_rows))
        for index, column in enumerate(columns)
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    rule = "  ".join("-" * widths[column] for column in columns)
    rows = [header, rule]
    rows.extend("  ".join(value.ljust(widths[column]) for column, value in zip(columns, row)) for row in display_rows)
    return "\n".join(rows)


def _assert_current(path: Path) -> None:
    checkpoints = list(Path("results/ablations/quantile_cqr").rglob("model_seed_*.pt"))
    if checkpoints and path.stat().st_mtime < max(c.stat().st_mtime for c in checkpoints):
        raise SystemExit(f"{path} is older than the newest checkpoint; rerun inference.py")
    head = pd.read_csv(path, nrows=50)
    if "calibration" in head and not head["calibration"].astype(str).str.contains("Platt").all():
        raise SystemExit(f"{path} was not produced by the Platt pipeline; rerun inference.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--latex-out", type=Path, default=Path("paper/latex/probability_table.tex")
    )
    parser.add_argument(
        "--csv-out", type=Path, default=Path("paper/latex/probability_table.csv")
    )
    args = parser.parse_args()
    input_path = args.input or (
        Path("results/inference_probabilities_ensemble.csv")
        if Path("results/inference_probabilities_ensemble.csv").is_file()
        else Path("results/inference_probabilities_seed_42.csv")
    )
    _assert_current(input_path)
    table, months, threshold = build_table(input_path)
    args.latex_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.latex_out.write_text(render_latex(table, months, threshold), encoding="utf-8")
    paper_table = table.copy()
    for lead_column in [column for column in paper_table if column.startswith("T+")]:
        paper_table[lead_column] = paper_table[lead_column].map(format_probability)
    paper_table.to_csv(args.csv_out, index=False)
    print(f"Event: SPEI <= {threshold:g} | {table.attrs['calibration_description']}, {table.attrs['n_seeds']} seed(s); probabilities are estimates, not guarantees")
    print(
        "Each row is one configured site and accumulation scale; each lead includes its target month."
    )
    print(render_terminal(table, threshold))
    if "range_table" in table.attrs:
        print("\nAcross-seed min-max ranges:")
        print(render_terminal(table.attrs["range_table"], threshold))
    print(f"\nLaTeX table: {args.latex_out}")
    print(f"Machine-readable table: {args.csv_out}")


if __name__ == "__main__":
    main()
