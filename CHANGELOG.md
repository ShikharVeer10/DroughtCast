# Changelog

## Phase 1 — Target and leakage fixes (2026-10-26)

- Align ONI, DMI, NAO, and PDO at month *m* to each index's published month *m−1* value; this prevents centered ONI from including month *m* and records source-month provenance for every input window.
- When SPEIbase files are supplied, validate computed SPEI per site and accumulation scale before replacing targets; `results/speibase_validation.csv` records sample count, correlation, RMSE, and bias.
- No scientific values or reported performance metrics were changed.

## Phase 1 — Canonical architecture

- Single model class `DroughtCast` (`MultiScaleForecaster` deprecated alias); removed legacy `DroughtCastModel` / `UncertaintyWeightedMultiTaskLoss` implementations.
- Additive temporal attention `score_t = v^T tanh(W h_t)`; `configs/base_diagram.yaml` locks diagram settings; unknown/dead config keys raise.
- Conformance test writes `results/param_count.json`. README documents the real 20 ablation variants.

## Phase 0 — Data/target integrity (blocking)

- **PET (Hargreaves-Samani):** use `ET0 = 0.0023*(Tmean+17.8)*sqrt(TR)*(Ra/2.45)` with mid-month day-of-year; remove unused surface-radiation term. Same formula in `data/api.py`.
- **SPEI nonparametric standardization:** consistent Gringorten plotting positions with interpolated z-scores and symmetric dry/wet tail extrapolation (removes the asymmetric floor/ceiling for out-of-sample extremes).
- **Information parity:** replace shifted `lag_spei_*` inputs with unshifted `spei_{3,6,12}_obs` (SPEI observed through the last input month).
- **ONI look-ahead:** shift ONI by +1 month so the feature at month `m` uses the value centered on `m-1`.
- **Baselines:** require `statsmodels` and `xgboost`; remove silent Ridge/HistGBM fallbacks; fix VAR `forecast()` history length; replace fake ConvLSTM with GRU and TCN baselines (validation early stopping).
- **Figures:** `make_figures.py` reads only real `results/` artefacts and fails if missing; delete synthetic `np.random` figure inputs.
- **Training:** checkpoint selection uses unweighted validation MSE (logs both weighted loss and MSE).

Any previously reported metrics trained on `lag_spei_*`, unshifted ONI, or the old PET/SPEI formulas are not comparable to post-Phase-0 numbers and must be regenerated.
