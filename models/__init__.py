"""DroughtCast model package — canonical class is DroughtCast."""

from models.attention import Attention, TemporalAttention
from models.decoder import CascadedHydrologicalDecoder, Decoder, ForecastDecoder
from models.encoder import (
    CapacityMatchedGRUEncoder,
    CapacityMatchedTransformerEncoder,
    SharedLSTMEncoder,
)
from models.forecaster import DroughtCast, MultiScaleForecaster

__all__ = [
    "DroughtCast",
    "MultiScaleForecaster",
    "SharedLSTMEncoder",
    "CapacityMatchedGRUEncoder",
    "CapacityMatchedTransformerEncoder",
    "TemporalAttention",
    "Attention",
    "ForecastDecoder",
    "Decoder",
    "CascadedHydrologicalDecoder",
]
