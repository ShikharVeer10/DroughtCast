# Submission readiness checklist

## Implemented and exercised

- Diagram-aligned default: shared LSTM encoder, one temporal attention distribution/context, three SPEI accumulation-scale heads, four forecast leads.
- Date-aligned chronological splits, gaps, train-only normalization, and train-era-only target calibration.
- Five fixed-seed in-region temporal experiment; same-window persistence, seasonal climatology, Ridge, and histogram gradient-boosting baselines.
- Per-scale/per-lead and per-site metrics, drought-event scores, site-year paired bootstrap intervals, and monthly panel HAC DM statistics.
- Automated tests and CI; leave-region-out mode and ablation-grid scripts; one exploratory Australia holdout run.
- Optional quantile/CQR training, Captum Integrated Gradients (with local fallback), attention export, and publication PNG/PDF figure tooling; smoke-tested, not yet supported by multi-seed scientific conclusions.

## Still required before an international conference submission

- Run and report official SPEIbase comparison for all locations/scales, and justify point-cell alignment and PET method with a domain expert.
- Run five seeds for each predeclared region holdout and selected architecture ablations; current cross-region result is only one Australia seed.
- Freeze an analysis plan and code/data versions; verify all source units, endpoint payload hashes, licenses, and derived-data provenance.
- Optional quantile, validation-calibrated CQR, and learned uncertainty task weighting are implemented; run multi-seed experiments and add CRPS/coverage results before claiming an uncertainty contribution.
- Conformal coverage must be interpreted empirically because overlapping monthly windows violate simple exchangeability assumptions.
- Conduct full related-work review, establish a defensible novelty claim, and obtain climate/hydrology collaborator review.
- See `results/summary.csv` (regenerate after the schema-v3 rebuild).
- IEEE Xplore inclusion depends on acceptance by an IEEE-sponsored conference/journal and compliance with its current author/submission requirements; software work cannot guarantee acceptance or indexing.
