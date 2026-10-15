# DroughtCast model card

## Intended use

Research comparison of monthly forecasts for SPEI-3/6/12 accumulation scales at eight configured point sites. Not validated for operational warning or decisions.

## Architecture

Six monthly feature vectors (12 inputs) → shared LSTM(64) → dropout → LSTM(32) → one temporal softmax attention distribution and shared context → three scale-specific four-lead forecast heads. Default point outputs predict T+1…T+4.

## Training/evaluation

Seeds 11, 22, 33, 44, 55; chronological train through 2010, validation 2012–2015, test 2017–2024, with gaps. Normalizers and target calibration use training era only. Checkpoints are selected by validation loss. Baselines use the same windows.

## Data and caveats

NASA POWER and NOAA climate indices. SPEI targets are computed from monthly precipitation minus approximate Hargreaves-Samani PET (monthly aggregate inputs), not official SPEIbase. Point locations do not represent full states/regions. ONI is seasonal and mapped to a central month as an approximation. Attribution/attention is not causal evidence.

## Performance

See generated per-scale/per-lead tables in `results/summary.csv` and site-level details in each seed JSON. The model loses to Ridge at all scales at T+1 in this run; it does not support general superiority claims.

## Limitations

Only eight sites, one temporal holdout, approximate targets, and no official target validation. Quantile/CQR code is optional and has only smoke-level evaluation; do not describe it as validated calibration or deploy without domain validation and stronger independent evaluation.
