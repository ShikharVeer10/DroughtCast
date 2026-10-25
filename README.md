# DroughtCast

DroughtCast forecasts monthly SPEI at three **accumulation scales** (SPEI-3, SPEI-6, SPEI-12). Every head predicts the same forecast horizon **T+1…T+4**. Never confuse accumulation scales with lead times.

## Canonical architecture (`configs/base_diagram.yaml`)

| Phase | Component | Spec |
|------|-----------|------|
| 1 | Input | `[B, 6, 12]` — 12 climate, ocean and drought-state variables over T−6…T−1 |
| 2 | Encoder | `LSTM(64) → Dropout(0.2) → LSTM(32)` → all hidden states `[B, 6, 32]` |
| 3 | Attention | One additive temporal attention: `score_t = vᵀ tanh(W h_t)`, softmax over 6 steps → `[B, 6, 1]` |
| 4 | Context | `c = Σ_t α_t h_t`, `c ∈ ℝ³²`, **shared** by all heads |
| 5 | Heads | Three separate decoders (SPEI-3 / SPEI-6 / SPEI-12); each maps only `c` → 4 leads |
| 6 | Output | Predictions `[B, 3, 4]` |

The single model class is `models.forecaster.DroughtCast` (`MultiScaleForecaster` is a deprecated alias). Architecture conformance is locked by `tests/test_architecture_conformance.py`.

## Data

Default builder: NASA POWER monthly climate + NOAA teleconnections (ONI shifted +1 month to avoid look-ahead). SPEI is computed from P−PET (Hargreaves-Samani with `Ra/2.45`, mid-month DOY) via a nonparametric Gringorten transform calibrated through 2010-12. Observed SPEI features are **unshifted** `spei_3_obs` / `spei_6_obs` / `spei_12_obs` (information parity with persistence).

```powershell
python -m data.build_dataset --start 1981 --end 2024
python train.py --config configs/base_diagram.yaml
python evaluate.py --config configs/base_diagram.yaml
```

Leak-free splits (unchanged): train ≤2010-12, val 2012-01…2015-12, test ≥2017-01, with 12-month gaps. Checkpoint selection uses **unweighted validation MSE**.

## Ablation variants (20)

Run with `python -m experiments.run_ablations`. Current rows:

| # | Variant | What changes |
|---|---------|--------------|
| 1 | `no_attention` | `attention: none` (last hidden state) |
| 2 | `head_specific_attention` | separate additive attention per scale |
| 3 | `separate_encoders` | one encoder stack per scale |
| 4 | `gru` | capacity-matched GRU 64→32 |
| 5 | `transformer` | Transformer + positional encoding |
| 6 | `variable_gate` | learned input-variable gate |
| 7 | `fixed_task_weights` | equal task losses |
| 8 | `uncertainty_task_weights` | Kendall uncertainty weighting |
| 9 | `no_extreme_weighting` | `extreme_weight: 1.0` |
| 10 | `extreme_weight_2` | `extreme_weight: 2.0` |
| 11 | `huber_point_loss` | Huber instead of MSE |
| 12 | `quantile_cqr` | three-quantile output + CQR |
| 13 | `window_3` | `seq_len: 3` |
| 14 | `window_12` | `seq_len: 12` |
| 15 | `window_24` | `seq_len: 24` |
| 16 | `without_indices` | drop ONI/DMI/NAO/PDO |
| 17 | `without_observed_spei` | drop `spei_*_obs` |
| 18 | `climate_only` | climate variables only |
| 19 | `without_climate` | indices + SPEI obs only |
| 20 | `single_site_guntur` | single-site train/eval |

Optional Phase-2 upgrades (all **off** in `base_diagram.yaml`; each becomes an ablation row when enabled): residual heads, static descriptors, input extras (month sincos / water balance), multi-quantile CRPS, multi-scale consistency penalty.

## Development

```powershell
pip install -r requirements.txt
pytest -q
python -m experiments.run_all --skip-build
python -m experiments.run_region_holdouts
python -m experiments.export_interpretability --checkpoint checkpoints/best_research_model.pt
python make_figures.py
```

Figures read only real artefacts under `results/` and fail if they are missing. Do not invent metrics; print tables from actual runs only. `base_diagram.yaml` is the reference model in every results table.

## Gridded spatial evidence

The repository now includes a strict spatial-evidence path for public gridded
products. Source provenance is recorded in `data/spatial_sources.json`.
Download and harmonize public boundaries, Copernicus DEM, and SPEIbase (or
another documented gridded target) before producing a continuous map. A valid
NetCDF evidence product must contain `time`, `lat`, `lon`, and matching
`obs_spei_3/6/12` plus `pred_spei_3/6/12` variables. Run:

```powershell
python -m experiments.validate_spatial_evidence data/spatial_grid.nc
python -m experiments.plot_spatial_evidence data/spatial_grid.nc --scale spei_6
```

The validator refuses missing or incompatible fields and reports common-grid
RMSE, MAE, bias, POD, FAR, and CSI. It does not interpolate the eight point
sites into a raster. Spatial metrics may be reported as generalization evidence
only when the grid test dates and target construction are held out from model
training.

## Assumptions

- Point sites, not areal averages; Murray–Darling is a basin proxy.
- Computed SPEI ≠ official SPEIbase unless `--speibase` is supplied and validated.
- Attention weights describe model behaviour, not causation.
- See `CHANGELOG.md` when a fix changes reported numbers.
