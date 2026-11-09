"""Generate Phase 5.4 rolling-origin feasibility artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.protocol import MODEL_FEATURES
from experiments.rolling_origin import generate_rolling_origins

OUT = ROOT / "results" / "maharashtra" / "phase5" / "phase5_4_rolling_origin"
DATA = ROOT / "data" / "maharashtra" / "monthly_sites.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["date"])
    rows, exclusions = generate_rolling_origins(frame, features=MODEL_FEATURES)
    rows.to_csv(OUT / "rolling_origin_manifest.csv", index=False)
    exclusions.to_csv(OUT / "exclusion_report.csv", index=False)
    origin_summary = {
        "valid_origins": sorted(rows["origin"].unique().tolist()),
        "number_valid_origins": int(rows["origin"].nunique()),
        "unique_site_month_observations": int(rows[["site", "target_date"]].drop_duplicates().shape[0]),
        "windows_per_origin": rows.groupby("origin")[["site", "input_end"]].nunique().to_dict("index"),
        "windows_per_scale_and_lead": {
            f"{scale}|T+{lead}": int(count)
            for (scale, lead), count in rows.groupby(["scale", "lead"]).size().items()
        },
        "exclusion_count": len(exclusions),
    }
    (OUT / "origin_summary.json").write_text(json.dumps(origin_summary, indent=2) + "\n", encoding="utf-8")
    availability = (
        rows.groupby(["origin", "scale", "lead"], as_index=False)
        .size()
        .rename(columns={"size": "valid_sample_count"})
    )
    availability.to_csv(OUT / "sample_availability.csv", index=False)
    (OUT / "feasibility_report.md").write_text(
        "# Rolling-origin feasibility report\n\n"
        f"Valid origins: **{origin_summary['number_valid_origins']}**. "
        f"Unique site-month target observations: **{origin_summary['unique_site_month_observations']}**. "
        f"Manifest rows count site × scale × lead records and must not be treated as independent observations. "
        f"Excluded records: **{origin_summary['exclusion_count']}**.\n\n"
        "The framework is deterministic and reserves the Phase 5.1.1 final-evaluation period. "
        "It is suitable for evaluation design and sample accounting, but it does not train models, "
        "select hyperparameters, or provide an independent probabilistic calibration partition.\n",
        encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.name != "sha256_manifest.json")
    (OUT / "sha256_manifest.json").write_text(
        json.dumps({path.name: sha256(path) for path in files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
