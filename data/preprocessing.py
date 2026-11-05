import numpy as np
import pandas as pd


class ClimateNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("Expected [time, feature] data")
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 0), self.std, 1.0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("ClimateNormalizer must be fitted before transform")
        return (np.asarray(data, dtype=float) - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

def compute_biophysical_features(df:pd.DataFrame) -> pd.DataFrame:
    t_max=df["T2M_MAX"]
    t_min=df["T2M_MIN"]
    t_dew=df["T2MDEW"]

    es_max=0.6108*np.exp((17.27*t_max) / (t_max + 237.3))
    es_min=0.6108*np.exp((17.27 * t_min) / (t_min + 237.3))
    es=(es_max + es_min)/2.0
    ea=0.6108* np.exp((17.27 * t_dew)/(t_dew + 237.3))
    df["VPD"]=np.maximum(0.0,es-ea)

    #Vegetation Condition Index(VCI)
    if "ndvi" in df.columns:
        ndvi_min=df.groupby("id")["ndvi"].transform(lambda x: x[x.index < "2025-01-01"].min())
        ndvi_max=df.groupby("id")["ndvi"].transform(lambda x:x[x.index < "2025-01-01"].max())

        df["VCI"]=100.0 * (df["ndvi"] - ndvi_min) / (ndvi_max - ndvi_min + 1e-6)
        df["VCI"]=df["VCI"].clip(0.0,100.0)

    return df
