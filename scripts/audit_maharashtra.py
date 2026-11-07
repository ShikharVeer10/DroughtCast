
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "config.yaml"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"
OUT = ROOT / "results" / "maharashtra"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def grid_audit(cfg: dict) -> tuple[pd.DataFrame, int]:
    sites = cfg["sites"]
    rows = []
    for site, spec in sites.items():
        rows.append(
            {
                "site": site,
                "lat": spec["lat"],
                "lon": spec["lon"],
                "power_lat": round(spec["lat"] * 2) / 2,
                "power_lon": round(spec["lon"] * 2) / 2,
            }
        )
    cells = pd.DataFrame(rows)
    return cells, int(cells[["power_lat", "power_lon"]].drop_duplicates().shape[0])


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = pd.read_csv(DATA, parse_dates=["date"])
    seq_len = int(cfg["data"]["seq_len"])
    forecast_steps = int(cfg["data"]["forecast_steps"])
    accumulation = 12
    required_gap = seq_len + forecast_steps + accumulation
    b = cfg["data"]["split_boundaries"]
    train_end = pd.Timestamp(b["train_end"])
    val_start = pd.Timestamp(b["val_start"])
    val_end = pd.Timestamp(b["val_end"])
    test_start = pd.Timestamp(b["test_start"])
    train_val_gap = (val_start - train_end).days // 30
    val_test_gap = (test_start - val_end).days // 30
    cells, independent_sites = grid_audit(cfg)
    cells.to_csv(OUT / "power_grid_audit.csv", index=False)

    report = f"""# Maharashtra leakage audit

Generated from `configs/config.yaml`, `data/maharashtra/monthly_sites.csv`, and the
current implementation. No test-period values are used to choose a model.

## Split gaps

| boundary | configured dates | gap (months) | required (months) | status |
|---|---|---:|---:|---|
| train -> validation | {train_end.date()} -> {val_start.date()} | {train_val_gap} | {required_gap} | {"PASS" if train_val_gap >= required_gap else "FAIL"} |
| validation -> test | {val_end.date()} -> {test_start.date()} | {val_test_gap} | {required_gap} | {"PASS" if val_test_gap >= required_gap else "FAIL"} |

The test period remains `{b["test_start"]}` onward. Windows are retained only when
all input and target months belong to one split.

## Computation and alignment

* `data/build_dataset.py` uses trailing `rolling(scale)` accumulations and fits
  each calendar-month distribution only on values through 2010-12.
* `spei_*_obs` is copied from the computed SPEI at the same month, so the final
  input month cannot contain a later target month.
* ONI is parsed as a centered three-month season and aligned with a two-month
  lag; DMI/NAO/PDO use a one-month publication lag. The concrete-date regression
  test is `tests/test_phase0_data_integrity.py::test_centered_oni_source_for_concrete_input_month_has_no_future_month`.
* `data/loader.py` fits each normalizer on rows classified as train and reuses it
  for validation and test. Missing inputs are not imputed with future-derived
  values; affected windows are rejected.
* Training checkpoints use validation MSE; CQR and event calibrators are fit on
  validation only.

## Effective spatial sample

NASA POWER's approximately 0.5-degree grid maps the 12 configured sites to
**{independent_sites}** distinct cells. The complete mapping is in
`results/maharashtra/power_grid_audit.csv`; site-year bootstrap statistics must
therefore treat sites sharing a cell as one effective spatial unit.

## Sanity checks and limitations

The repository contains the ONI alignment and chronological-window regression
tests. A shuffled-target/future-input negative-control result is not present in
`results/maharashtra/`; it must be generated before claiming that verification
criterion. Likewise, the requested rolling-origin 1981-2015 tuning cannot be
claimed from this checkout because the Maharashtra table begins in
{data["date"].min().date()} and no tuning trial log exists. These are explicit
FAIL/OPEN items, not silently inferred passes.

## Evidence inventory

SPEIbase comparison: `results/maharashtra/speibase_validation/site_scale_metrics.csv`.
Existing seed-42 and multi-seed test metrics are under `results/maharashtra/`.
They are historical artifacts and are not rewritten or reinterpreted here.
"""
    (ROOT / "LEAKAGE_AUDIT.md").write_text(report, encoding="utf-8")

    summary = pd.read_csv(OUT / "site_skill_t4.csv")
    summary["beats_persistence"] = summary["skill_vs_persistence"] > 0
    grouped = summary.groupby("scale", sort=False)["beats_persistence"].agg(
        ["sum", "count"]
    )
    lines = [
        "# Maharashtra results summary",
        "",
        (
            "All figures below are read from `results/maharashtra/site_skill_t4.csv`; "
            "positive skill means lower RMSE than persistence at T+4."
        ),
        "",
    ]
    for scale, row in grouped.iterrows():
        lines.append(
            f"* **{scale} T+4:** beats persistence at {int(row['sum'])} of {int(row['count'])} site comparisons."
        )
    lines.extend(
        [
            "",
            (
                "This is not a claim of overall superiority. Lead- and scale-specific "
                "RMSE/MAE/R2, bootstrap intervals, and HAC DM tests must be taken from "
                "a regenerated frozen-config evaluation before publication."
            ),
        ]
    )
    (ROOT / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "config": "configs/config.yaml",
        "data_sha256": sha256(DATA),
        "git_commit": git_commit(),
        "test_start": b["test_start"],
        "test_end": b.get("test_end"),
        "seq_len": seq_len,
        "forecast_steps": forecast_steps,
        "independent_power_cells": independent_sites,
    }
    (OUT / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
