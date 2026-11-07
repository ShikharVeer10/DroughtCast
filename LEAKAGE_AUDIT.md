# Maharashtra leakage audit

Generated from `configs/config.yaml`, `data/maharashtra/monthly_sites.csv`, and the
current implementation. No test-period values are used to choose a model.

## Split gaps

| boundary | configured dates | gap (months) | required (months) | status |
|---|---|---:|---:|---|
| train -> validation | 2010-12-01 -> 2012-01-01 | 13 | 22 | FAIL |
| validation -> test | 2015-12-01 -> 2017-01-01 | 13 | 22 | FAIL |

The test period remains `2017-01-01` onward. Windows are retained only when
all input and target months belong to one split.

## Computation and alignment

* `data/build_dataset.py` uses trailing `rolling(scale)` accumulations and fits
  each calendar-month distribution only on values through 2010-12.
* `spei_*_obs` is copied from the computed SPEI at the same month, so the final
  input month cannot contain a later target month.
* ONI is parsed as a centered three-month season and aligned with a two-month
  lag; DMI/NAO/PDO use a one-month publication lag. The concrete-date regression
  test is `tests/test_phase0_data_integrity.py::test_centered_oni_source_for_concrete_input_month_has_no_future_month`.
* `data/loader.py` fits each normalizer on rows classified as train and reuses it
  for validation and test. Missing inputs are not imputed with future-derived
  values; affected windows are rejected.
* Training checkpoints use validation MSE; CQR and event calibrators are fit on
  validation only.

## Effective spatial sample

NASA POWER's approximately 0.5-degree grid maps the 12 configured sites to
**12** distinct cells. The complete mapping is in
`results/maharashtra/power_grid_audit.csv`; site-year bootstrap statistics must
therefore treat sites sharing a cell as one effective spatial unit.

## Sanity checks and limitations

The repository contains the ONI alignment and chronological-window regression
tests. A shuffled-target/future-input negative-control result is not present in
`results/maharashtra/`; it must be generated before claiming that verification
criterion. Likewise, the requested rolling-origin 1981-2015 tuning cannot be
claimed from this checkout because the Maharashtra table begins in
1990-01-01 and no tuning trial log exists. These are explicit
FAIL/OPEN items, not silently inferred passes.

## Evidence inventory

SPEIbase comparison: `results/maharashtra/speibase_validation/site_scale_metrics.csv`.
Existing seed-42 and multi-seed test metrics are under `results/maharashtra/`.
They are historical artifacts and are not rewritten or reinterpreted here.
