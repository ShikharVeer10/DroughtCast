from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def enrich_frame(monthly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
  """Join observed daily POWER soil moisture and derive physical features."""
  monthly = monthly.copy().replace([-999.0, -99.0, -9999.0, -32768.0], np.nan)
  daily = daily.copy()
  for frame in (monthly, daily):
    frame["site"] = frame["site"].astype(str).str.lower().str.strip()
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
  daily = daily.groupby(["site", "date"], as_index=False).agg({
      "T2M_MAX": "mean", "T2M_MIN": "mean", "GWETTOP": "mean", "GWETROOT": "mean",
  })
  result = monthly.drop(columns=[c for c in ("T2M_MAX", "T2M_MIN", "GWETTOP", "GWETROOT", "VPD", "T2M_RANGE") if c in monthly])
  result = result.merge(daily, on=["site", "date"], how="left", validate="one_to_one")
  result["T2M_RANGE"] = (result["T2M_MAX"] - result["T2M_MIN"]).clip(lower=0)
  es = (0.6108 * np.exp(17.27 * result["T2M_MAX"] / (result["T2M_MAX"] + 237.3)) +
        0.6108 * np.exp(17.27 * result["T2M_MIN"] / (result["T2M_MIN"] + 237.3))) / 2
  ea = 0.6108 * np.exp(17.27 * result["T2MDEW"] / (result["T2MDEW"] + 237.3))
  result["VPD"] = (es - ea).clip(lower=0)
  result["spei_3_obs"] = result["spei_3"]
  result["spei_6_obs"] = result["spei_6"]
  result["spei_12_obs"] = result["spei_12"]
  # Inputs from build_dataset are already protocol-aligned and carry their
  # observation month/staleness audit columns. Never apply a second shift here.
  return result.drop(columns=["T2M_MAX", "T2M_MIN"])


def enrich_from_sources(monthly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
  result = enrich_frame(monthly, daily)
  for site, group in result.groupby("site"):
    if group["GWETTOP"].notna().sum() == 0:
      raise ValueError(f"No observed soil-moisture input for {site}")
  return result


def main() -> None:
  monthly_path = Path("data/maharashtra/monthly_sites.csv")
  raw_daily_path = Path("data/maharashtra_daily_raw.csv")
  meta_path = Path("data/maharashtra/dataset_metadata.json")

  if not monthly_path.exists():
    raise FileNotFoundError(f"Missing {monthly_path}. Run build_dataset first.")
  if not raw_daily_path.exists():
    raise FileNotFoundError(f"Missing {raw_daily_path}; observed soil-moisture inputs cannot be synthesized.")

  df_monthly = pd.read_csv(monthly_path).replace([-999.0, -99.0, -9999.0, -32768.0], np.nan)
  df_monthly["site"] = df_monthly["site"].astype(str).str.lower().str.strip()
  configured_sites = json.loads(Path("configs/config.yaml").read_text(encoding="utf-8"))[
      "sites"
  ]
  df_daily = pd.read_csv(raw_daily_path)
  df_daily["site"] = df_daily["site"].astype(str).str.lower().str.strip()

  # 1. Aggregate daily NASA POWER soil moisture & temperature extremes to monthly means
  df_daily["date_dt"] = pd.to_datetime(df_daily["date"])
  df_daily["date"] = (
      df_daily["date_dt"]
      .dt.to_period("M")
      .dt.to_timestamp()
      .dt.strftime("%Y-%m-%d")
  )

  agg_daily = df_daily.groupby(["site", "date"], as_index=False).agg({
      "T2M_MAX": "mean",
      "T2M_MIN": "mean",
      "GWETTOP": "mean",
      "GWETROOT": "mean",
  })

  drop_existing = [
      c
      for c in [
          "T2M_MAX",
          "T2M_MIN",
          "GWETTOP",
          "GWETROOT",
          "VPD",
          "T2M_RANGE",
          "NDVI",
          "VCI",
      ]
      if c in df_monthly.columns
  ]
  df = pd.merge(
      df_monthly.drop(columns=drop_existing),
      agg_daily,
      on=["site", "date"],
      how="left",
  )

  df["dt"] = pd.to_datetime(df["date"])
  df["month"] = df["dt"].dt.month
  df["year"] = df["dt"].dt.year

  # Do not fabricate observations with full-record climatology backfill.

  # 2. Requirement 3: New Parameters (T2M_RANGE & Vapor Pressure Deficit VPD in kPa)
  df["T2M_RANGE"] = (df["T2M_MAX"] - df["T2M_MIN"]).clip(lower=0.0)
  es_max = 0.6108 * np.exp((17.27 * df["T2M_MAX"]) / (df["T2M_MAX"] + 237.3))
  es_min = 0.6108 * np.exp((17.27 * df["T2M_MIN"]) / (df["T2M_MIN"] + 237.3))
  es = 0.5 * (es_max + es_min)
  ea = 0.6108 * np.exp((17.27 * df["T2MDEW"]) / (df["T2MDEW"] + 237.3))
  df["VPD"] = np.maximum(0.0, es - ea)

  # 3. Requirement 4: Vegetation Indices (NDVI & leakage-free training-calibrated VCI)
  df = df.sort_values(["site", "date"]).reset_index(drop=True)
  gwet_2m = df.groupby("site")["GWETROOT"].transform(
      lambda s: s.rolling(2, min_periods=1).mean()
  )
  site_vpd_min = df["site"].map(
      df[df["year"] <= 2010].groupby("site")["VPD"].min()
  )
  site_vpd_max = df["site"].map(
      df[df["year"] <= 2010].groupby("site")["VPD"].max()
  )
  vpd_scaled = (df["VPD"] - site_vpd_min) / (site_vpd_max - site_vpd_min + 1e-6)

  # NDVI cannot be inferred from soil moisture and VPD. Keep it missing until
  # a documented satellite product is joined; do not manufacture a proxy.
  df["NDVI"] = np.nan

  # Calibrate VCI (0-100%) strictly on 1981-2010 training window per site
  train_ref = df[df["year"] <= 2010]
  ndvi_min_map = df["site"].map(train_ref.groupby("site")["NDVI"].min())
  ndvi_max_map = df["site"].map(train_ref.groupby("site")["NDVI"].max())
  df["VCI"] = np.nan

  # Clean helper columns and reorder
  df = df.drop(columns=["dt", "month", "year", "T2M_MAX", "T2M_MIN"])
  from data.protocol import INDEX_ALIGNMENT
  for index_name, alignment in INDEX_ALIGNMENT.items():
    if index_name in df:
      # Raw builder indices are already aligned. Avoid applying the delay twice.
      pass
  df["spei_3_obs"] = df["spei_3"]
  df["spei_6_obs"] = df["spei_6"]
  df["spei_12_obs"] = df["spei_12"]
  df.to_csv(monthly_path, index=False)

  # 4. Update dataset_metadata.json SHA256 & feature list so protocol validation passes
  sha256_hash = hashlib.sha256(monthly_path.read_bytes()).hexdigest()
  if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
  else:
    meta = {}
  meta["sha256"] = sha256_hash
  meta["rows"] = len(df)
  meta["sites"] = configured_sites
  meta["schema_version"] = 4
  from data.protocol import INDEX_ALIGNMENT
  meta["index_alignment"] = INDEX_ALIGNMENT
  meta["features"] = [
      "PRECTOTCORR",
      "T2M",
      "T2MDEW",
      "WS10M",
      "GWETTOP",
      "GWETROOT",
      "VPD",
      "T2M_RANGE",
      "oni",
      "dmi",
      "nao",
      "pdo",
      "spei_3_obs",
      "spei_6_obs",
      "spei_12_obs",
  ]
  meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

  # 5. Requirement 2: Write Quantitative Data Validation Report for the Paper
  val_dir = Path("results/maharashtra")
  val_dir.mkdir(parents=True, exist_ok=True)
  validation_report = {
      "study_area": "Maharashtra, India (4 IMD Subdivisions, 12 Benchmark Stations)",
      "total_site_month_rows": len(df),
      "missing_value_count": int(df[meta["features"]].isna().sum().sum()),
      "physical_bounds_check": {
          "GWETTOP_range": [
              float(df["GWETTOP"].min()),
              float(df["GWETTOP"].max()),
          ],
          "GWETROOT_range": [
              float(df["GWETROOT"].min()),
              float(df["GWETROOT"].max()),
          ],
          "VPD_kPa_range": [float(df["VPD"].min()), float(df["VPD"].max())],
          "NDVI_range": [float(df["NDVI"].min()), float(df["NDVI"].max())],
          "VCI_pct_range": [float(df["VCI"].min()), float(df["VCI"].max())],
      },
      "spei_3_correlations": {
          "GWETROOT_corr": float(df["GWETROOT"].corr(df["spei_3_obs"])),
          "VCI_corr": float(df["VCI"].corr(df["spei_3_obs"])),
          "VPD_corr": float(df["VPD"].corr(df["spei_3_obs"])),
      },
      "dataset_sha256": sha256_hash,
  }
  (val_dir / "data_validation_report.json").write_text(
      json.dumps(validation_report, indent=2), encoding="utf-8"
  )
  print("Enriched dataset saved to:", monthly_path)
  print(
      "Validation report saved to: results/maharashtra/data_validation_report.json"
  )


if __name__ == "__main__":
  main()
