import numpy as np
import pandas as pd
import xarray as xr

from data.build_dataset import ingest_speibase
from data.validate_speibase import compare_netcdf


def test_netcdf_speibase_nearest_cell_ingestion_and_validation(tmp_path):
    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    lat = np.array([10.0, 20.0])
    lon = np.array([70.0, 80.0])
    sites = {"test": {"lat": 19.0, "lon": 79.0, "region": "r"}}
    for scale in (3, 6, 12):
        values = np.zeros((len(dates), len(lat), len(lon)), dtype=np.float32)
        values[:, 1, 1] = np.arange(len(dates)) + scale
        ds = xr.Dataset(
            {"spei": (("time", "latitude", "longitude"), values)},
            coords={"time": dates, "latitude": lat, "longitude": lon},
        )
        ds.to_netcdf(tmp_path / f"spei{scale:02d}.nc")
    computed = pd.DataFrame(
        {
            "site": "test",
            "date": dates,
            "spei_3": np.arange(4) + 3,
            "spei_6": np.arange(4) + 6,
            "spei_12": np.arange(4) + 12,
            "feature": 1.0,
        }
    )
    merged = ingest_speibase(computed, tmp_path, sites, "spei")
    assert merged["feature"].eq(1.0).all()
    assert np.allclose(merged["spei_3"], np.arange(4) + 3)
    scores = compare_netcdf(computed, tmp_path, "spei", sites)
    assert len(scores) == 3
    assert scores["n"].eq(4).all()
    assert scores["correlation"].eq(1.0).all()
