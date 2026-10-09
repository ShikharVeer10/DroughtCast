# DroughtCast 🌦️

**Multi-horizon drought forecasting with a shared LSTM encoder, temporal attention, and task-specific SPEI prediction heads.**

DroughtCast is a research-oriented PyTorch project for forecasting drought conditions at multiple accumulation horizons. It uses recent climate observations to predict the Standardized Precipitation–Evapotranspiration Index (SPEI) for **3-month, 6-month, and 12-month horizons**, with four forecast steps produced by each head.

> **Implementation note:** The current CSV data loader uses five climate features and a six-timestep history. The architecture illustration describes a 12-variable input, so that illustration should be treated as a conceptual design unless the code and configuration are updated to use 12 input features.

## Contents

- [Overview](#overview)
- [Model architecture](#model-architecture)
- [Data and targets](#data-and-targets)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference and visualisation](#inference-and-visualisation)
- [Configuration](#configuration)
- [Current implementation notes](#current-implementation-notes)
- [Limitations and next steps](#limitations-and-next-steps)

## Overview

DroughtCast learns temporal patterns from historical climate inputs and uses a shared representation to predict three related drought indicators:

| Prediction head | Target | Intended interpretation |
|---|---|---|
| SPEI-3 | `spei_3` | Short-term moisture conditions |
| SPEI-6 | `spei_6` | Medium-term moisture conditions |
| SPEI-12 | `spei_12` | Longer-term moisture conditions |

For each target, the model returns four values corresponding to forecast steps **T+1, T+2, T+3, and T+4**. The exact calendar meaning of these steps depends on the frequency and alignment of the supplied data; the current project is configured around a short historical lookback and a four-step output window.

### Main features

- Shared two-layer LSTM encoder for learning temporal features.
- Temporal attention for assigning different weights to historical steps.
- Three separate forecast decoders for SPEI-3, SPEI-6, and SPEI-12.
- Sliding-window dataset construction from climate and SPEI time series.
- Smooth L1 training loss, AdamW optimisation, gradient clipping, and learning-rate scheduling.
- Evaluation utilities for RMSE, MAE, NSE, and KGE, plus a persistence baseline.
- Plotting utilities for comparing predictions with target values.

## Model architecture

The implemented forecasting pipeline is:

```text
Historical climate features
        │
        ▼
Sliding windows (6 historical steps)
        │
        ▼
Shared LSTM encoder
  LSTM( input_dim → 64 )
        │
      Dropout
        │
        ▼
  LSTM( 64 → 32 )
        │
        ▼
Temporal attention
  score each encoded time step
  softmax-normalise attention weights
        │
        ▼
Context vector (weighted sequence summary)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   SPEI-3 decoder  SPEI-6 decoder  SPEI-12 decoder
        │              │              │
        ▼              ▼              ▼
   4 predictions  4 predictions  4 predictions
```

### 1. Historical input window

The active loader reads the following five monthly climate features from `data/era5_inputs.csv`:

- `PRECTOTCORR` — corrected precipitation
- `T2M` — near-surface air temperature
- `T2MDEW` — dew-point temperature
- `WS10M` — wind speed at 10 metres
- `ALLSKY_SFC_SW_DWN` — all-sky surface shortwave downward radiation

With `seq_len: 6`, one sample contains six consecutive observations. With five variables per observation, the intended input tensor shape is:

```text
[batch_size, 6, 5]
```

### 2. Shared LSTM encoder

`models/encoder.py` defines `SharedLSTMEncoder`. It processes the input sequence through an LSTM with 64 hidden units, dropout, and a second LSTM with 32 hidden units. The output retains an encoded representation for each historical time step.

### 3. Temporal attention

`models/attention.py` defines `TemporalAttention`. A small feed-forward scoring network calculates a score for each encoded time step. Softmax turns these scores into attention weights that sum to one across the time dimension. The context vector is the weighted sum of the encoder outputs.

The attention weights are returned alongside the forecasts and can be inspected to understand which positions in the input window receive more model attention. They are model attention scores, not proof of causal importance.

### 4. Multi-head forecast decoders

`models/forecaster.py` uses the same context vector with three separately parameterised `ForecastDecoder` instances:

- `decoder_spei3` predicts four values for `spei_3`.
- `decoder_spei6` predicts four values for `spei_6`.
- `decoder_spei12` predicts four values for `spei_12`.

Each decoder maps the 32-dimensional context representation through a small feed-forward network to four outputs. The three heads share the learned encoder and attention representation while learning different target mappings.

## Data and targets

The main training data loader expects:

```text
data/
├── era5_inputs.csv
└── spei_targets.nc
```

`era5_inputs.csv` must contain the five column names listed above. `spei_targets.nc` must contain three variables:

- `spei_3`
- `spei_6`
- `spei_12`

The loader aligns the available input and target lengths, replaces missing values with zero, standardises the climate features, and creates sliding windows. For each window, the six historical input steps are paired with four future target steps for each SPEI variable.

**Data preparation matters:** the current implementation fits the normaliser on the complete input series before windows are built. For rigorous out-of-sample evaluation, split data chronologically first and fit preprocessing statistics on the training partition only; reuse those statistics for validation, testing, and inference.

### Data download helpers

- `fetch_data.py` requests monthly climate observations from the NASA POWER API and saves the result as `data/era5_inputs.csv`.
- `download_data.py` can request monthly ERA5 data from the Copernicus Climate Data Store (CDS), or generate a small synthetic NetCDF dataset when invoked with `--synthetic`.
- SPEI targets must be provided or computed separately. A valid target file containing the three expected variable names is required by the main loader.

External data providers may require accounts, API credentials, and acceptance of their terms. Check dataset licences and usage conditions before redistributing downloaded data.

## Repository structure

```text
DroughtCast/
├── configs/
│   └── config.yaml                 # Experiment and model settings
├── data/
│   ├── dataset.py                  # PyTorch dataset wrapper
│   ├── loader.py                   # Input/target loading and window creation
│   ├── preprocessing.py            # Climate feature normalisation
│   ├── era5_inputs.csv             # Climate input data
│   ├── era5_inputs.nc              # NetCDF climate data
│   └── spei_targets.nc             # SPEI targets for three horizons
├── models/
│   ├── attention.py                # Temporal attention
│   ├── decoder.py                  # Four-step forecast decoder
│   ├── encoder.py                  # Shared LSTM encoder
│   ├── forecaster.py               # Multi-horizon model assembly
│   └── model.py                    # Alternative multi-head attention model
├── evaluation/
│   └── persistence.py              # Baseline predictors
├── utils/
│   ├── metrics.py                  # Regression and hydrology metrics
│   └── visualisation.py            # Forecast and attention plotting helpers
├── checkpoints/
│   └── best_research_model.pt      # Saved model checkpoint
├── plots/
│   └── forecast_comparison.png     # Example forecast comparison plot
├── download_data.py                # ERA5 download / synthetic data helper
├── fetch_data.py                  # NASA POWER data fetch helper
├── train.py                        # Training entry point
├── evaluation.py                   # Evaluation entry point
├── inference.py                    # Inference entry point
├── plot.py                         # Prediction comparison plot
├── requirements.txt
└── README.md
```

Some files shown in this structure may be generated or replaced locally as part of data preparation and training.

## Installation

Python 3.10+ is recommended. Use a virtual environment to isolate dependencies.

```bash
git clone https://github.com/ShikharVeer10/DroughtCast.git
cd DroughtCast

python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project's listed dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project dependencies include PyTorch, NumPy, xarray, NetCDF4, Matplotlib, Seaborn, PyYAML, and the Copernicus CDS API client.

## Data preparation

### Option A — use the repository's prepared data

Check that `data/era5_inputs.csv` contains all five expected feature columns and that `data/spei_targets.nc` contains `spei_3`, `spei_6`, and `spei_12`.

### Option B — download climate inputs

For NASA POWER inputs, review the parameters and date range in `fetch_data.py`, configure credentials if the API requires them, and run:

```bash
python fetch_data.py
```

For ERA5 monthly means via Copernicus CDS, configure the CDS API credentials for your environment and run:

```bash
python download_data.py
```

The CDS helper downloads `data/era5_inputs.nc`; the primary training loader, however, reads `data/era5_inputs.csv`. Convert the downloaded NetCDF inputs to the required CSV schema before training, or adapt the loader to consume NetCDF directly. Prepare the SPEI target file separately.

### Option C — generate synthetic data for pipeline experiments

```bash
python download_data.py --synthetic
```

This helper writes synthetic NetCDF files for testing basic data workflows. It does **not** create the CSV file expected by the main training loader, and its synthetic target values are not meaningful drought observations. Use real, aligned data for scientific evaluation.

## Training

The current training entry point uses `configs/config.yaml` for model construction and data-window settings:

```bash
python train.py
```

The script loads the CSV climate inputs and SPEI targets, then trains the shared encoder and three output heads. Its current training loop uses:

- **Loss:** sum of Smooth L1 losses for SPEI-3, SPEI-6, and SPEI-12.
- **Optimiser:** AdamW with weight decay.
- **Learning-rate scheduler:** `ReduceLROnPlateau`.
- **Gradient clipping:** maximum gradient norm of `1.0`.
- **Checkpoint:** `checkpoints/best_research_model.pt`, updated when the running average training loss improves.

Note that the configuration file currently lists 15 epochs and a learning rate of `0.001`, while `train.py` directly sets 60 epochs, batch size 32, and learning rate `1e-3`. The values inside the script determine these settings until the training code is changed to read them from the configuration.

## Evaluation

Run:

```bash
python evaluation.py
```

The evaluation script loads the checkpoint and reports the following metrics for each SPEI output:

| Metric | Meaning |
|---|---|
| RMSE | Root Mean Squared Error; lower is better |
| MAE | Mean Absolute Error; lower is better |
| NSE | Nash–Sutcliffe Efficiency; closer to 1 is generally better |
| KGE | Kling–Gupta Efficiency; closer to 1 is generally better |

The script also includes a persistence-baseline comparison. Treat these results as preliminary unless evaluation uses a strictly held-out chronological test set. The current script evaluates the dataset returned by the loader and does not create a train/validation/test split itself.

## Inference and visualisation

### Plot predictions

After training, compare predictions with target values:

```bash
python plot.py
```

The script saves a plot to `plots/forecast_comparison.png`.

### Inference entry point

An inference script is provided:

```bash
python inference.py
```

Review `inference.py` before relying on it in production. The current version is not fully aligned with the training pipeline: it imports a different model class (`MultiHorizonMultiHeadForecaster` from `models.model`), while the training script saves weights for the class in `models.forecaster`. The NetCDF input schema also differs from the CSV schema used in training. These interfaces should be made consistent before using the script for reproducible forecasts.

## Configuration

The main configuration is `configs/config.yaml`:

```yaml
experiment:
  name: "Multi-Horizon Multi-Head Drought Forecasting"
  seed: 42

data:
  seq_len: 6
  input_dim: 5
  forecast_steps: 4

model:
  hidden_dim_1: 64
  hidden_dim_2: 32
  dropout: 0.2

training:
  batch_size: 32
  epochs: 15
  learning_rate: 0.001
  checkpoint_dir: "checkpoints"
```

The `input_dim` must match the number of feature columns supplied to the model. The configured `seq_len` controls the historical lookback window, and `forecast_steps` controls the number of values produced by every decoder.

## Current implementation notes

The repository is actively evolving. Before interpreting outputs as validated drought forecasts, address these consistency and research-quality points:

1. **Input dimensionality:** the diagram depicts 12 climate/ocean variables, but the current loader builds five input features and the configuration sets `input_dim: 5`.
2. **Preprocessing leakage:** normalisation currently fits across the entire available time series. Fit preprocessing only on the training period for honest temporal evaluation.
3. **Data pipeline alignment:** `download_data.py` writes NetCDF while the principal training loader expects CSV climate data. Use one documented schema end-to-end.
4. **Inference/training model mismatch:** align `inference.py` with the exact class, feature order, normalisation statistics, and checkpoint used by `train.py`.
5. **Evaluation protocol:** create chronological train, validation, and test splits; report metrics on held-out data and compare against baselines using the same target and sample alignment.
6. **Reproducibility:** apply the configured seed consistently to Python, NumPy, and PyTorch, and record data periods, preprocessing parameters, model settings, and checkpoint provenance.
7. **Data provenance:** document the source, units, spatial location, temporal resolution, and processing procedure for every input and target variable.

## Limitations and next steps

DroughtCast is a research prototype, not a validated operational early-warning system. Forecast performance depends on data quality, spatial and temporal alignment, training methodology, and independent validation. Attention weights provide a view into model behaviour but should not be interpreted as causal explanations.

Potential next steps include:

- Unify the CSV/NetCDF input interfaces and model class used for training and inference.
- Add chronological train/validation/test splits and save normalisation parameters with the model checkpoint.
- Make all training settings configurable through `config.yaml`.
- Validate forecasts against held-out observations and persistence/climatology baselines for all three SPEI targets.
- Add spatial evaluation and uncertainty estimates if the project is extended beyond a single time series.

## Citation and data usage

If you use DroughtCast in research, cite this repository and describe the exact commit, data sources, preprocessing procedure, and evaluation split used. No formal publication citation or software licence is specified here yet; check the repository for updates before reuse or redistribution.

## Acknowledgements

DroughtCast builds on the PyTorch ecosystem and climate-data resources from NASA POWER and the Copernicus Climate Data Store. Refer to each provider's current terms and data documentation when using their data.
