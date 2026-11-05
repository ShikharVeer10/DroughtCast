from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import requests

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

PARAMETERS = (
    "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,T2MDEW,WS10M,ALLSKY_SFC_SW_DWN,GWETTOP,GWETROOT"
)


def fetch_station_data(
    lat: float, lon: float, start: int = 19900101, end: int = 20241231
) -> pd.DataFrame:
  """Queries NASA POWER daily API for a specific coordinate point."""
  query_params = {
      "parameters": PARAMETERS,
      "community": "ag",
      "longitude": lon,
      "latitude": lat,
      "start": start,
      "end": end,
      "format": "json",
      "units": "metric",
  }

  print(
      f"🌐 Fetching NASA POWER data for Lat: {lat}, Lon: {lon} ({start}-{end})..."
  )
  response = requests.get(POWER_API_URL, params=query_params)
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
  all_dfs = []

  for site_id, meta in MAHARASHTRA_STATIONS.items():
    df = fetch_station_data(meta["lat"], meta["lon"])
    df["site"] = site_id
    df["region"] = meta["region"]
    df["lat"] = meta["lat"]
    df["lon"] = meta["lon"]
    all_dfs.append(df.reset_index())

  combined_df = pd.concat(all_dfs, ignore_index=True)

  # Save raw daily output
  Path("data").mkdir(exist_ok=True)
  combined_df.to_csv("data/maharashtra_daily_raw.csv", index=False)
  print("✨ Successfully saved raw Maharashtra data to data/maharashtra_daily_raw.csv")


if __name__ == "__main__":
  build_maharashtra_dataset()