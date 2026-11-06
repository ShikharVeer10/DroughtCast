from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml


CONFIG_PATH = Path("configs/config.yaml")
SPEI_COLUMNS = [
            "spei_3",
            "spei_6",
            "spei_12",
]
BASELINE_START = "1981-01-01"
BASELINE_END = "2010-12-01"
TEST_START = "2017-01-01"


def load_configured_data() -> tuple[dict, pd.DataFrame]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_config = config["data"]

    dataset_path = Path(data_config["path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Maharashtra dataset not found: {dataset_path}\n"
            "Build it first using:\n"
            "python -m data.build_dataset --config configs/config.yaml"
        )

    frame = pd.read_csv(dataset_path, parse_dates=["date"])
    frame["site"] = frame["site"].astype(str).str.strip().str.lower()
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()

    return config, frame


def validate_dataset(frame: pd.DataFrame,expected_sites: list[str],features: list[str],) -> None:
    required = {"date", "site", *features, *SPEI_COLUMNS}
    missing_columns = sorted(required.difference(frame.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if frame[["date", "site"]].isna().any().any():
        raise ValueError("Dataset contains missing site or date values.")

    duplicates = frame.duplicated(["site", "date"], keep=False)
    if duplicates.any():
        print("\nWARNING: Duplicate site-month observations:")
        print(
            frame.loc[duplicates, ["site", "date"]]
            .sort_values(["site", "date"])
            .to_string(index=False)
        )

    actual_sites = set(frame["site"].unique())
    configured_sites = set(expected_sites)

    missing_sites = sorted(configured_sites - actual_sites)
    unexpected_sites = sorted(actual_sites - configured_sites)

    if missing_sites:
        raise ValueError(f"Configured Maharashtra sites missing: {missing_sites}")

    if unexpected_sites:
        raise ValueError(f"Non-Maharashtra/unconfigured sites found: {unexpected_sites}")

    print(f"Configured Maharashtra sites: {len(configured_sites)}")
    print(f"Sites present in dataset: {len(actual_sites)}")
    print(f"Dataset date range: {frame['date'].min()} to {frame['date'].max()}")
    print(f"Dataset rows: {len(frame)}")
    if len(configured_sites) != 12:
        raise ValueError(f"Expected all 12 configured Maharashtra sites, got {len(configured_sites)}")

    print("\nMonthly continuity checks:")
    for site, group in frame.groupby("site", sort=True):
        dates = pd.DatetimeIndex(group["date"].sort_values())
        expected_dates = pd.date_range(
            dates.min(), dates.max(), freq="MS"
        )
        missing_dates = expected_dates.difference(dates)

        status = "OK" if len(missing_dates) == 0 else "MISSING MONTHS"
        print(
            f"{site:15s} rows={len(group):4d} "
            f"missing_months={len(missing_dates):3d} [{status}]"
        )

        if len(missing_dates):
            print(f"Missing: {list(missing_dates.strftime('%Y-%m'))[:12]}")


def report_feature_quality(frame: pd.DataFrame,features: list[str],) -> None:
    print("\nFEATURE QUALITY BY SITE")
    print("=" * 100)

    for site, group in frame.groupby("site", sort=True):
        print(f"\nSite: {site}")

        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values.dropna()

            if finite.empty:
                print(
                    f"  {feature:22s} "
                    f"missing={values.isna().sum():4d} NO VALID VALUES"
                )
                continue

            std = finite.std(ddof=0)
            variation = "CONSTANT/NEAR-CONSTANT" if std < 1e-8 else "OK"

            print(
                f"  {feature:22s} "
                f"missing={values.isna().sum():4d} "
                f"mean={finite.mean():10.4f} "
                f"std={std:10.4f} "
                f"min={finite.min():10.4f} "
                f"max={finite.max():10.4f} "
                f"[{variation}]"
            )


def report_recent_inputs(frame: pd.DataFrame,features: list[str],months: int = 6,) -> None:
    print(f"\nLATEST {months} INPUT MONTHS BY SITE")
    print("=" * 100)

    for site, group in frame.groupby("site", sort=True):
        recent = (
            group.sort_values("date")
            .tail(months)
            .set_index("date")[features]
        )

        print(f"\nSite: {site}")
        print(recent.to_string())


def report_spei_quality(frame: pd.DataFrame) -> None:
    print("\nOBSERVED SPEI QUALITY BY SITE")
    print("=" * 100)

    for site, group in frame.groupby("site", sort=True):
        group = group.sort_values("date")
        print(f"\nSite: {site}")

        for column in SPEI_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce")
            finite = values.dropna()

            if finite.empty:
                print(f"  {column}: NO VALID VALUES")
                continue

            std = finite.std(ddof=0)
            z_count = (
                int((((finite - finite.mean()) / std).abs() > 3).sum())
                if std > 0
                else 0
            )

            baseline = group.loc[
                group["date"].between(BASELINE_START, BASELINE_END),
                column,
            ]
            baseline = pd.to_numeric(baseline, errors="coerce").dropna()

            later = group.loc[group["date"] >= TEST_START, column]
            later = pd.to_numeric(later, errors="coerce").dropna()

            if len(baseline):
                outside_count = int(((later < baseline.min()) | (later > baseline.max())).sum())
            else:
                outside_count = -1

            print(
                f"  {column}: "
                f"valid={len(finite):4d} "
                f"missing={values.isna().sum():4d} "
                f"mean={finite.mean():8.4f} "
                f"std={std:8.4f} "
                f"min={finite.min():8.4f} "
                f"max={finite.max():8.4f} "
                f"|z|>3={z_count:3d} "
                f"post-2017-outside-baseline={outside_count:3d}"
            )


def main() -> None:
    config, frame = load_configured_data()

    data_config = config["data"]
    expected_sites = [
        str(site).strip().lower()
        for site in data_config["site_filter"]
    ]
    features = list(data_config["features"])

    print("DROUGHTCAST MAHARASHTRA DATA DIAGNOSTICS")
    print("=" * 100)
    print(f"Dataset: {data_config['path']}")
    print(f"Expected sites: {len(expected_sites)}")
    print(f"Configured input features: {len(features)}")

    validate_dataset(frame, expected_sites, features)
    report_feature_quality(frame, features)
    report_recent_inputs(frame, features)
    report_spei_quality(frame)

    print("\nEXTERNAL INDEX ALIGNMENT")
    try:
        from data.protocol import INDEX_ALIGNMENT
        for name, rule in INDEX_ALIGNMENT.items():
            print(f"{name}: lag={rule['lag_months']} month(s); {rule['source']}")
        metadata_path = Path(data_config["path"]).parent / "dataset_metadata.json"
        if metadata_path.is_file():
            import json
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("index_alignment") != INDEX_ALIGNMENT:
                raise ValueError("Dataset index alignment metadata is stale")
    except ImportError:
        raise
    print("\nDIAGNOSTICS COMPLETED")
    print("No data or model files were modified.")

if __name__ == "__main__":
    main()
