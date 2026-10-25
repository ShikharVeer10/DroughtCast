"""Backward-compatible re-exports. Prefer models.forecaster.DroughtCast. """
from models.forecaster import DroughtCast, MultiScaleForecaster

# Historical names now point at the single canonical class.
DroughtCastModel = DroughtCast
MultiTaskAttentionLSTM = DroughtCast
MultiHorizonForecaster = DroughtCast
