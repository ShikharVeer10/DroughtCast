from __future__ import annotations
from pathlib import Path
import pandas as pd
import requests
import yaml

# 12 Benchmark Stations across Maharashtra's 4 IMD Sub-Regions
MAHARASHTRA_STATIONS = {
    # Marathwada
    "aurangabad": {"lat": 19.8762, "lon": 75.3433, "region": "Marathwada"},
    "latur": {"lat": 18.4088, "lon": 76.5604, "region": "Marathwada"},
    "beed": {"lat": 18.9891, "lon": 75.7601, "region": "Marathwada"},
    "nanded": {"lat": 19.1383, "lon": 77.3210, "region": "Marathwada"},
    # Vidarbha
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "region": "Vidarbha"},
    "akola": {"lat": 20.7002, "lon": 77.0082, "region": "Vidarbha"},
    "amravati": {"lat": 20.9374, "lon": 77.7796, "region": "Vidarbha"},
    # Madhya Maharashtra
    "pune": {"lat": 18.5204, "lon": 73.8567, "region": "Madhya_Maharashtra"},
    "solapur": {"lat": 17.6599, "lon": 75.9064, "region": "Madhya_Maharashtra"},
    "nashik": {"lat": 19.9975, "lon": 73.7898, "region": "Madhya_Maharashtra"},
    # Konkan
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "region": "Konkan"},
    "ratnagiri": {"lat": 16.9902, "lon": 73.3120, "region": "Konkan"},
}

POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMETERS = [
    "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "WS10M",
    "ALLSKY_SFC_SW_DWN", "GWETTOP", "GWETROOT",
]


def normalize_uploaded(frame: pd.DataFrame) -> pd.DataFrame:
  """Normalize a NASA POWER daily frame to the canonical indexed schema."""
  result = frame.copy()
  date_column = "YYYYMMDD" if "YYYYMMDD" in result.columns else "date"
  if date_column not in result.columns:
    raise ValueError("Uploaded data must contain YYYYMMDD or date")
  result.index = pd.to_datetime(result.pop(date_column).astype(str), format="%Y%m%d")
  result.index.name = "date"
  missing = [name for name in PARAMETERS if name not in result.columns]
  if missing:
    raise ValueError(f"Uploaded data missing parameters: {missing}")
  return result.loc[:, list(PARAMETERS)].apply(pd.to_numeric, errors="coerce")


def pet_daily_aggregate(daily: pd.DataFrame, latitude: float) -> pd.DataFrame:
  """Aggregate daily POWER values to monthly totals/means and estimate PET."""
  required = {"PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "T2M_MAX", "T2M_MIN"}
  missing = required.difference(daily.columns)
  if missing:
    raise ValueError(f"Daily data missing parameters: {sorted(missing)}")
  frame = daily.copy()
  frame.index = pd.to_datetime(frame.index)
  grouped = frame.resample("MS").agg({
      "PRECTOTCORR": "sum",
      "T2M": "mean",
      "T2MDEW": "mean",
      "WS10M": "mean",
      "ALLSKY_SFC_SW_DWN": "mean",
      "T2M_MAX": "mean",
      "T2M_MIN": "mean",
  })
  from data.build_dataset import estimate_pet_hargreaves
  grouped["pet_mm"] = estimate_pet_hargreaves(grouped, latitude)
  return grouped


def fetch_station_data(
    lat: float, lon: float, start: int = 19900101, end: int = 20251231
) -> pd.DataFrame:
  """Queries NASA POWER daily API for a specific coordinate point."""
  query_params = {
      "parameters": ",".join(PARAMETERS),
      "community": "ag",
      "longitude": lon,
      "latitude": lat,
      "start": start,
      "end": end,
      "format": "json",
      "units": "metric",
  }

  print(f"Fetching NASA POWER data for Lat: {lat}, Lon: {lon} ({start}-{end})...")
  response = requests.get(POWER_API_URL, params=query_params, timeout=120)
  response.raise_for_status()

  data = response.json()
  parameters_data = data["properties"]["parameter"]

  # Convert dictionary of parameters into a clean pandas DataFrame
  df = pd.DataFrame(parameters_data)
  df.index = pd.to_datetime(df.index, format="%Y%m%d")
  df.index.name = "date"

  # Replace NASA POWER missing value sentinel (-999) with NaN
  df = df.replace([-999.0, -99.0], pd.NA)
  return df


def build_maharashtra_dataset() -> None:
  """Pulls and aggregates data for all 12 Maharashtra stations into monthly CSV."""
  config_path = Path("configs/config.yaml")
  config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
  start = int(config.get("data", {}).get("start_year", 1990))
  end = int(config.get("data", {}).get("end_year", 2025))
  end_month = int(config.get("data", {}).get("end_month", 12))
  all_dfs = []

  for site_id, meta in MAHARASHTRA_STATIONS.items():
    last_day = pd.Timestamp(end, end_month, 1) + pd.offsets.MonthEnd(0)
    df = fetch_station_data(meta["lat"], meta["lon"], start * 10000 + 101, int(last_day.strftime("%Y%m%d")))
    df["site"] = site_id
    df["region"] = meta["region"]
    df["lat"] = meta["lat"]
    df["lon"] = meta["lon"]
    all_dfs.append(df.reset_index())

  combined_df = pd.concat(all_dfs, ignore_index=True)

  # Save raw daily output
  Path("data").mkdir(exist_ok=True)
  combined_df.to_csv("data/maharashtra_daily_raw.csv", index=False)
  print("Successfully saved raw Maharashtra data to data/maharashtra_daily_raw.csv")


if __name__ == "__main__":
  build_maharashtra_dataset()
