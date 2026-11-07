# DroughtCast: Maharashtra monthly drought forecasting

This repository contains a research prototype for forecasting monthly SPEI at 12 configured Maharashtra locations. It is **not currently a validated operational drought-warning system or a conference-ready result**: the locally calculated targets have only moderate agreement with SPEIbase, severe-event agreement is weak at many sites, and the older evaluation/checkpoint artifacts do not match the current data protocol.

## Canonical study configuration

`configs/config.yaml` is the canonical configuration. It lists exactly 12 locations: Aurangabad, Latur, Beed, Nanded (Marathwada); Nagpur, Akola, Amravati (Vidarbha); Pune, Solapur, Nashik (Madhya Maharashtra); Mumbai, Ratnagiri (Konkan).

The active model uses 6 monthly input steps, 12 configured features, and predicts 4 leads (T+1 through T+4) for SPEI-3, SPEI-6, and SPEI-12. The configured experiment predicts q10, q50, and q90 for each scale and lead using pinball loss. These are forecast quantiles, not drought probabilities. The q10–q90 interval is nominally the central 80% forecast interval. The shared LSTM and attention implementation is documented in `models/forecaster.py` and `tests/test_architecture_conformance.py`; model outputs have one scale head per target scale. The point model in `configs/base_diagram.yaml` is a separate experiment and must use a distinct checkpoint.

The configured dataset spans January 1990 through October 2026. The chronological evaluation boundaries are train through December 2010, validation January 2011–December 2015, and test January 2017–December 2025. The 2016 gap is excluded; all input and target months in a window must belong to the same split. A 2026 input period is not a 2026 test result; the test ends in 2025. A four-lead model produces only the next four monthly targets, not a full-year 2027 forecast.

## Data and scientific limitations

The builder uses NASA POWER monthly climate fields, daily POWER data where complete daily aggregation is available, and NOAA CPC/PSL teleconnection series. Precipitation is treated as a daily mean rate in mm/day and multiplied by calendar days for monthly totals. PET is a monthly Hargreaves–Samani approximation from temperature fields, not FAO-56 Penman–Monteith. VPD and temperature range are derived from temperature/dewpoint and extrema. SPEI targets use fixed monthly calibration through December 2010. The index alignment protocol is versioned in `data/protocol.py`; exact historical publication timestamps for all indices are not supplied by their source tables, so the stated lags are conservative proxies, not verified release-time reconstruction.

`docs/source_data.md` describes the source products, units, processing, and limitations. Local SPEIbase comparison artifacts under `results/maharashtra/speibase_validation/` compare nearest 0.5-degree cells for 1990–2024. Those results show imperfect target agreement and should be treated as a target-validity warning, not proof that either target is interchangeable with the other. There is no independent PET comparison artifact currently available. Periods outside the SPEIbase record are not reference validated.

## Rebuild and evaluate

Run from the repository root. Rebuilding requires NASA/NOAA source access; use explicit refresh flags to replace caches. Existing results are preserved; write new analyses to a new output directory.

```powershell
python -m data.build_dataset --config configs/config.yaml
python train.py --config configs/config.yaml --seed 42
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/maharashtra_2026/current_model.pt
```

Before treating these commands as a reproducible study, first confirm the builder completes its data-quality checks and that the resulting checkpoint's recorded dataset hash, protocol version, features, sites, split dates, and architecture match the dataset and configuration. The existing checkpoints were produced under earlier schemas/configurations and are expected to be rejected by current validation.

Run tests with `python -m pytest -q`. Data source refresh and large SPEIbase validation may require network access and substantial disk space. Current checked-in code and local result artifacts do not establish that the model beats persistence, reliable event probabilities, spatial transfer, or forecast skill on future 2027 observations.

## Drought event probabilities

For an event definition such as moderate drought (SPEI ≤ −1.0), event probability is a separate post-processing product. q10/q50/q90 are quantiles; they are not probabilities of drought. Any calibrated event-risk table must identify its SPEI scale, threshold, forecast origin, lead, calibrator, validation fit split, event counts, and checkpoint/dataset provenance. The checked-in `drought_no_drought_probabilities_2027.csv` is explicitly uncalibrated and is not a validated probability forecast.

## Reproducibility artifacts

Data protocol metadata is written alongside the dataset. New checkpoints include hashes and configuration metadata. Existing files under `results/` are preserved historical artifacts; their schema/config/dataset provenance must be checked before use. Do not combine metrics from incompatible checkpoints or datasets.
