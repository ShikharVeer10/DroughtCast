# Repository structure

The repository uses a small set of stable command-line entry points at the
root and keeps implementation modules in focused packages. The root scripts
are intentionally retained because the Makefile, experiment runners, tests,
and published regeneration commands invoke them directly.

## Runtime code

- `data/`: dataset construction, preprocessing, loaders, API processing, and
  optional spatial data preparation.
- `models/`: the canonical `MultiScaleForecaster`/`DroughtCast` model and its
  encoder, attention, decoder, and configuration validation modules.
- `evaluation/`: baseline implementations and evaluation helpers.
- `utils/`: metrics, statistics, conformal calibration, artifacts, and
  interpretability utilities.
- `experiments/`: reproducible training/evaluation orchestration, ablations,
  calibration audits, probability exports, and spatial workflows.

## Root command-line entry points

- `train.py`: train a model.
- `evaluate.py`: evaluate chronological test-period predictions.
- `inference.py`: generate site or all-site probabilistic forecasts.
- `make_paper_figures.py`: generate paper figures from real result files.
- `inspect_data.py`: inspect the configured dataset.

## Research assets

- `configs/`: validated experiment and architecture configuration.
- `checkpoints/`: trained model checkpoints.
- `results/`: generated metrics, predictions, provenance, and calibration
  artifacts.
- `paper/`: figure style, generated figures, plotted-data exports, and LaTeX
  assets.
- `tests/`: protocol, architecture, probability, spatial, and figure tests.

## Generated or local-only directories

- `output/` and `plots/`: auxiliary generated artifacts retained for backward
  compatibility with existing workflows.
- `.venv/`, `__pycache__/`, and `.pytest_cache/`: local Python tooling caches;
  they are not part of the research source tree.

Do not move the root entry points or rename package directories without also
updating the Makefile, README commands, experiment subprocess calls, and test
imports.
