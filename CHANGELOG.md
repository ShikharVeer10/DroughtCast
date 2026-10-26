# Changelog

## Phase 1 — Target and leakage fixes (2026-10-26)

- Align ONI, DMI, NAO, and PDO at month *m* to each index's published month *m−1* value; this prevents centered ONI from including month *m* and records source-month provenance for every input window.
- When SPEIbase files are supplied, validate computed SPEI per site and accumulation scale before replacing targets; `results/speibase_validation.csv` records sample count, correlation, RMSE, and bias.
- No scientific values or reported performance metrics were changed.

## Phase 2 — Evaluation integrity (2026-10-26)

- GRU and TCN baselines now average five declared seeds and select checkpoints by validation MSE; both retain equal zero-search tuning budgets alongside fixed classical baseline settings.
- Evaluation rejects exact prediction duplicates, reports a site-year block bootstrap as the primary paired comparison, labels pooled Wilcoxon secondary, enforces DM-HAC bandwidth at least the SPEI accumulation scale, and reports effective sample size and independent drought episodes.
- Probabilistic evaluation reports validation and test coverage; CRPS remains explicitly `crps_approx` for the configured three quantiles. CQR corrections are applied without silent clipping and nonfinite corrections raise an error.
- VAR now forecasts from the actual last k_ar monthly observations in each input window; it no longer repeats the endpoint observation to pad history.
- No existing results were regenerated or reinterpreted.

## Phase 3 — Code consistency (2026-10-26)

- The only model implementation is `MultiScaleForecaster`, publicly aliased as `DroughtCast`; obsolete model-name aliases were removed.
- Rewrote README around the six-phase diagram, corrected accumulation-scale and forecast-horizon terminology, chronology and regeneration commands. Existing strict config validation and base diagram conformance checks remain in force.
- No scientific values or reported performance metrics were changed.

## Phase 4 — Paper figures (2026-10-26)

- Added a shared serif/Okabe-Ito style, config-driven F1 architecture diagram, vector PDF and 300-dpi PNG export, and exact plotted-data CSV.
- Added `make_paper_figures.py --check`; F2–F10 are blocked by missing or inadequate current result exports. Existing legacy results were not used to fabricate required five-seed, quantile, bootstrap, SPEIbase, interpretability, or regional evidence.
- Added schema-v2 dataset provenance gates for training/evaluation and a current run manifest for multi-seed summaries. Evaluation exports per-site T+4 skill; SPEIbase scatter pairs are saved when reference files are supplied.
- Rebuilt the cached monthly dataset offline under schema v2 and verified seed-42 base-diagram training/evaluation. The complete baseline pass was stopped after several minutes of CPU-bound GRU/TCN seed fitting; no partial evaluation table is treated as a completed result.

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

## Calibrated probability table (2026-10-26)

- Added a reproducible exporter that formats the real all-site inference CSV as a 24-row site-by-accumulation-scale paper table with T+1..T+4 decimal-probability columns, LaTeX booktabs output, and a companion CSV. The current export contains eight configured sites and probabilities for SPEI <= -1.0; forecast months and threshold are read from the input file.
- Removed the unused legacy `make_figures.py`; `make_paper_figures.py` is the project's referenced figure entry point. Restored the shared figure-style module required by that entry point and its QA checks.
- No probabilities or evaluation metrics were edited by hand; table values are read directly from `results/inference_probabilities_seed_42.csv`.
