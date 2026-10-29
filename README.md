# DroughtCast

DroughtCast forecasts monthly SPEI at three **accumulation scales**: SPEI-3, SPEI-6, and SPEI-12. Every scale head predicts the same forecast horizon, T+1 through T+4.

## Reference architecture

The canonical configuration is [`configs/base_diagram.yaml`](configs/base_diagram.yaml). Its six phases are:

1. Six monthly input steps, each with 12 climate, teleconnection, and observed SPEI features.
2. `LSTM(64) -> Dropout(0.2) -> LSTM(32)`, returning `[B, 6, 32]`.
3. Additive temporal attention over six steps, with weights `[B, 6, 1]` that sum to one.
4. One shared context vector `c in R^32`.
5. Three separate accumulation-scale decoders, each reading only `c` and producing four leads.
6. Point forecasts with output shape `[B, 3, 4]`, trained with plain MSE.

`MultiScaleForecaster` is the sole model implementation; `DroughtCast` is its public alias. Architecture conformance is checked in `tests/test_architecture_conformance.py`.

## Data and leakage controls

`data/build_dataset.py` builds monthly data from NASA POWER climate fields and NOAA indices. PET uses the monthly Hargreaves-Samani approximation and mid-month day-of-year for extraterrestrial radiation. SPEI is standardized using calibration data through December 2010. Each feature month *m* uses the published ONI, DMI, NAO, and PDO values for month *m−1*. The observed state features `spei_3_obs`, `spei_6_obs`, and `spei_12_obs` use SPEI observed through the final input month, matching persistence information.

Chronological windows retain the train period through 2010-12, validation 2012-01 through 2015-12, and test from 2017-01. The intervening gaps are intentionally excluded. Feature normalization and SPEI calibration use training data only.

Training and evaluation require dataset metadata schema version 2. Rebuild older local datasets before running either command. Paper figures also require a current run manifest so stale metrics are not plotted as current evidence.

```powershell
python -m data.build_dataset --start 1981 --end 2024
python train.py --config configs/base_diagram.yaml
python evaluate.py --config configs/base_diagram.yaml
```

## Generate a new site forecast

Run inference from the project root to print calibrated probabilities for every configured site, or replace `all` with a single site name from `configs/config.yaml`:

  ```powershell
  python inference.py --site all
  ```

The command loads the current dataset-matched q10/q50/q90 checkpoint at `results/ablations/quantile_cqr/model_seed_42.pt`, reads the latest six real monthly rows for each configured site from `data/monthly_sites.csv`, applies each site's saved train-only normalizer, and fits a separate Platt-scaling calibrator (logistic regression on a piecewise-normal z-score of q10/q50/q90) for each accumulation scale and lead using validation data. The terminal tables print q10/q50/q90 forecasts and the calibrated probability that SPEI is at or below -1.0. The same rows are saved to `results/inference_probabilities_seed_42.csv`; pass `--output-csv PATH` to choose another destination. SPEI-3/6/12 are accumulation scales; T+1..T+4 are forecast leads.

Export a compact research-paper table (one row per configured site and SPEI accumulation scale, with separate columns for T+1..T+4) from those exact inference results:

```powershell
python experiments/export_probability_table.py
```

This prints decimal probabilities (0 to 1) in the terminal and writes `paper/latex/probability_table.tex` plus `paper/latex/probability_table.csv`. The LaTeX version is a `table*` using booktabs and includes the event definition, calibration description, and target months.

Audit the probability output before reporting it:

```powershell
python experiments/audit_probability_outputs.py
```

This checks the dataset hash, duplicate site/scale/lead rows, quantile ordering, and the number of distinct raw versus calibrated probabilities. Isotonic calibration is piecewise constant, so repeated calibrated values are expected; the audit confirms whether the underlying q10/q50/q90 forecasts are distinct.

Compare validation-only isotonic and logistic calibration on the untouched test period with:

```powershell
python -m experiments.compare_probability_calibration
```

Regenerate after rerunning `experiments/compare_probability_calibration.py`.

The checked-in monthly observations currently end in December 2024, so the default command forecasts January through April 2025; it does not provide a live forecast for the current date. To forecast from newer observations, pass a CSV containing at least the latest six consecutive month-start rows for the site, with `date`, an optional `site` column, and every feature listed under `data.features` in the config. Values must follow the project's feature definitions, including already lag-aligned teleconnection indices and observed SPEI state features. Keep the training dataset and its saved normalizers unchanged, and pass the separate input file:

```powershell
python inference.py --site jaipur --input-csv path\to\latest_monthly_inputs.csv
```

Use `--drought-threshold -1.5` to obtain a probability for the severe-drought threshold instead. Calibration is refit on the validation split for the selected threshold. It is an empirically calibrated model probability, not a guarantee of occurrence.

To generate chronological test-period calibration diagnostics and per-case calibrated probabilities for the research paper, run:

```powershell
python -m experiments.evaluate_event_calibration --config results/ablations/quantile_cqr/config.yaml --checkpoint results/ablations/quantile_cqr/model_seed_42.pt --output results/probability_calibration/calibration_seed_42.json --predictions-output results/probability_calibration/predictions_seed_42.csv --threshold -1.0
```

To compare computed SPEI against local SPEIbase NetCDF files while building, pass `--speibase PATH`; the per-site and per-scale validation table is written to `results/speibase_validation.csv`.

## Reproducibility

Install dependencies with `pip install -r requirements.txt`. Run `pytest -q` for the test suite. Training checkpoints are selected using unweighted validation MSE; the training log includes weighted loss and unweighted MSE.

Evaluation output is written to `results/test_metrics.json`. Classical baseline failures produce a warning and the failed baseline is omitted from that run’s result table. GRU and TCN predictions average five seeds with validation early stopping. Probabilistic metrics label three-quantile CRPS as `crps_approx`.

Paper figures are generated by `python make_paper_figures.py`. See [`paper/figures/README.md`](paper/figures/README.md) for figure sources and unavailable result artifacts. The generator reads only real files under `results/` and fails if a required input is absent.

## Spatial evidence

The separate gridded evidence workflow is configured in `configs/spatial_config.yaml` and uses `data.spatial_preprocess`, `experiments.run_spatial_pipeline`, and `experiments.plot_spatial_pipeline`. Gridded inference evaluates each valid cell independently; point-site predictions are not interpolated into rasters. SPEIbase is the target source and never enters the input features.

## Interpretation

Attention weights describe model behaviour and do not establish causality. Computed SPEI is not equivalent to official SPEIbase unless it has been explicitly compared against SPEIbase.
