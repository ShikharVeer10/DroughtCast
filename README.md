# DroughtCast

DroughtCast is a research prototype for monthly, multi-scale SPEI forecasting. SPEI-3/6/12 are **accumulation scales**; the four outputs are forecast leads T+1 through T+4.

## Data status and provenance

The formerly tracked `data/spei_targets.nc` and `data/era5_inputs.nc` were generated from synthetic data, and `spei03.nc`, `spei06.nc`, and `spei12.nc` were invalid HTML responses. They have been removed, along with the checkpoint trained on synthetic targets. `data/era5_inputs.csv` contained NASA POWER annual rows labelled month 13 and has been removed. Current experiments use NASA POWER and NOAA source data, with SPEI computed from those inputs; targets are not official SPEIbase observations.

The default builder downloads monthly NASA POWER precipitation, temperature, dew point, wind, and solar radiation for the eight sites in `configs/config.yaml` for 1981–2024. NOAA PSL/CPC ONI, DMI, NAO, and PDO series are fetched separately and parsed with explicit format checks. DMI/NAO/PDO are year rows with twelve monthly values; ONI is a three-month seasonal table, and its value is assigned to the season's central month as an approximation (not a true monthly index). The pipeline fails loudly if endpoint formats change. URLs were checked against live text endpoints on 2026-10-15; check again for future runs.

**PET assumption:** NASA POWER monthly `T2M`, `T2M_MAX`, and `T2M_MIN` are used in a monthly Hargreaves-Samani approximation. The monthly extrema are not full daily series, and this is not FAO-56 Penman-Monteith. SPEI is computed from monthly precipitation minus estimated PET, accumulated over 3/6/12 months, then transformed separately by calendar month using a nonparametric Gringorten transform fitted through 2010-12 only. These computed indices are not official SPEIbase observations.

```powershell
python -m data.build_dataset --start 1981 --end 2024
python train.py
python evaluate.py
```

### Local FastAPI downloader

To use NASA POWER's **daily** API or import downloaded daily CSVs, install requirements and launch the local service:

```powershell
pip install -r requirements.txt
uvicorn data.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Submit `POST /build` with JSON such as `{"start":19950101,"end":20241231,"sites":["guntur"]}`. It fetches/caches daily POWER data plus NOAA indices and creates `data/monthly_sites.csv`. Or use `POST /upload/{site}` in the docs page to upload a daily NASA POWER CSV for one configured site, then call `POST /build`. The service needs outbound HTTPS for any data not already cached. The POWER Daily API has a minimum date of 1981-01-01. This route uses daily T2M, T2M_MAX, and T2M_MIN to estimate daily Hargreaves-Samani PET, then aggregates to months; it remains an approximation and does not implement FAO-56 PM.
For the full 1981 training calibration, request data starting at 1981 (e.g. `{"start":19810101,"end":20241231,...}`).

NASA POWER requests require no API key. NOAA source endpoints are listed in `data/build_dataset.py`; endpoint schemas are external and should be rechecked before each new collection. Output has site and date keys; month-13 annual rows are excluded. NASA POWER precipitation is treated as mm/day and converted to monthly totals. Solar radiation is treated as kWh/m²/day and converted for PET.

Optional official SPEIbase targets can be selected with `python -m data.build_dataset --speibase PATH [--speibase-variable VARIABLE]`. The directory should contain `spei03.nc`, `spei06.nc`, and `spei12.nc`; the builder extracts the nearest grid cell for each configured site. Validate computed and official values with `python -m data.validate_speibase data/monthly_sites.csv PATH [--variable VARIABLE]`. Reference metrics depend on obtaining official files and must be recorded after running; this repository does not claim that comparison is complete.

## Protocol

Input feature order defaults to five climate variables, four teleconnection indices, and three lagged SPEI scales (12 total). `input_dim` is inferred from those columns. Site windows are contiguous monthly windows; inputs and targets must all fall within one partition: train through 2010-12, validation 2012-01 through 2015-12, or test from 2017-01 onward. The intervening periods are excluded as 12-month gaps. The per-site normalizer uses train rows only and is saved to `data/meta.json`. Missing target values remain masks; missing input windows are excluded. Test metrics are only computed by `evaluate.py`, after selecting a checkpoint on validation loss.

The default matches the uploaded architecture: shared LSTM(64)-Dropout-LSTM(32), one temporal attention distribution and context vector, then three SPEI-scale heads. Head-specific/no attention, variable gating, separate encoders, GRU, Transformer, input-group, and window-length variants are available in the ablation runner. The default publication run remains point-output squared error. Optional three-quantile output (`model.quantiles: [0.1, 0.5, 0.9]`) uses monotone outputs and pinball loss; `training.task_weighting: uncertainty` enables learned uncertainty weights. Evaluation estimates CQR corrections from validation forecasts and reports before/after test interval coverage and width. Because the windows overlap and form a serially dependent monthly series, standard split-conformal exchangeability guarantees may not apply; treat coverage as empirical. These options require fresh multi-seed experiments before being claimed as empirical contributions.

## Honest results

The current experiment has five seeds and compares DroughtCast with persistence, seasonal climatology, Ridge, and histogram gradient boosting on identical windows. The generated table is [results/results_table.md](results/results_table.md); machine-readable per-seed and mean/std files are `results/per_seed.csv` and `results/summary.csv`. At T+1, mean RMSE is 1.183 (SPEI-3), 0.929 (SPEI-6), and 0.762 (SPEI-12). Ridge is lower for all three (1.090, 0.761, 0.686 respectively); DroughtCast also loses to persistence for SPEI-6/12. Do not claim predictive superiority. These are computed SPEI targets and point-site aggregates, not official SPEIbase validation.

## Security and history cleanup

A NASA API key was committed in the previous `fetch_data.py` and has been removed from the working tree. Treat it as compromised and revoke it with NASA if still active. Purge it from all published git history before sharing this repository (for example, use `git filter-repo` with a replacement rule), then force-push rewritten history and notify collaborators. A working-tree edit does not erase a secret from old commits.

## Assumptions and limitations

- NOAA URLs and parsers are checked for expected parseable rows at run time, but their upstream format stability is not guaranteed.
- Site coordinates are representative point locations, not area averages; the Murray-Darling coordinate is a broad basin proxy.
- Computed SPEI calibration is site- and calendar-month-specific over the available period, which should be disclosed in any paper.
- Attention and gate scores describe model behaviour and are not causal evidence.
- IMD, ERA5-Land, CHIRPS, and TerraClimate alternative input mappings are not implemented or validated.
- No data redistribution licence is asserted here; follow NASA POWER, NOAA, and any SPEIbase terms and cite providers.

## Development

Run `python -m pytest -q` for the validation suite. To reproduce the five-seed temporal experiment from cached raw responses, run `python -m experiments.run_all --skip-build`; to rebuild the dataset first, omit `--skip-build`. This writes each seed checkpoint/report and generates `results/summary.csv`, `results/per_seed.csv`, and `results/results_table.md`. Run resumable architecture ablations with `python -m experiments.run_ablations --seeds 11 22 33 44 55`; add `--force` to rerun completed cells. Run the predeclared leave-one-region-out grid with `python -m experiments.run_region_holdouts --seeds 11 22 33 44 55`. Export attention and Integrated Gradients with `python -m experiments.export_interpretability --checkpoint CHECKPOINT`, then generate PNG/PDF figures using `python make_figures.py`. Figures and attributions describe model behavior and are not causal evidence. Generated raw data, checkpoints, metadata, and result artifacts are git-ignored.
