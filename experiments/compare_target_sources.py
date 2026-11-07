"""Paired comparison of local and SPEIbase target datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(group: pd.DataFrame, scale: int) -> dict:
    left = pd.to_numeric(group[f"spei_{scale}_local"], errors="coerce").to_numpy()
    right = pd.to_numeric(group[f"spei_{scale}_speibase"], errors="coerce").to_numpy()
    valid = np.isfinite(left) & np.isfinite(right)
    a, b = left[valid], right[valid]
    row = {"n": len(a)}
    if len(a) < 3:
        return row | {"correlation": None, "rmse": None, "bias": None}
    row.update({"correlation": float(np.corrcoef(a, b)[0, 1]),
                "rmse": float(np.sqrt(np.mean((a-b)**2))),
                "bias": float(np.mean(a-b))})
    for name, threshold in (("drought", -1.0), ("severe", -1.5)):
        x, y = a <= threshold, b <= threshold
        row[f"{name}_agreement"] = float(np.mean(x == y))
        row[f"{name}_local_events"] = int(x.sum())
        row[f"{name}_speibase_events"] = int(y.sum())
        row[f"{name}_both_events"] = int((x & y).sum())
    return row


def compare(local_path: Path, speibase_path: Path, output_dir: Path) -> pd.DataFrame:
    local = pd.read_csv(local_path, parse_dates=["date"])
    ref = pd.read_csv(speibase_path, parse_dates=["date"])
    scales = (3, 6, 12)
    left = local[["site", "date", *[f"spei_{s}" for s in scales], "region"]].rename(
        columns={f"spei_{s}": f"spei_{s}_local" for s in scales}
    )
    right = ref[["site", "date", *[f"spei_{s}" for s in scales]]].rename(
        columns={f"spei_{s}": f"spei_{s}_speibase" for s in scales}
    )
    paired = left.merge(right, on=["site", "date"], how="inner", validate="one_to_one")
    rows = []
    for scale in scales:
        for label, group in [("ALL", paired), *list(paired.groupby("region", sort=True)), *list(paired.groupby("site", sort=True))]:
            rows.append({"group": label, "scale": scale, **metrics(group, scale)})
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "target_source_comparison.csv", index=False)
    metadata = {
        "local_dataset": str(local_path),
        "speibase_target_dataset": str(speibase_path),
        "paired_rows": len(paired),
        "comparison_period": [str(paired.date.min().date()), str(paired.date.max().date())] if len(paired) else None,
        "matching": "site and exact month start; intersection of available targets",
        "notes": "Local target used Hargreaves-Samani and fixed-baseline empirical SPEI. SPEIbase targets are nearest 0.5-degree cells. Group label may name a region or site; inspect group membership in config.",
    }
    (output_dir / "target_source_comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", required=True)
    parser.add_argument("--speibase", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    table = compare(Path(args.local), Path(args.speibase), Path(args.output_dir))
    print(table[table.group == "ALL"].to_string(index=False))


if __name__ == "__main__":
    main()
