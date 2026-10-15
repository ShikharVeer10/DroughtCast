# Paper outline and submission strategy

## Working title options

1. **Leak-Free Multi-Scale Drought Forecasting Across Climatic Regions with Calibrated Uncertainty**
2. **Evaluating Recurrent and Attention Models for Monthly SPEI Forecasting Across Regions**
3. **A Reproducible Benchmark for Multi-Scale Drought Forecasting from Climate and Teleconnection Signals**

Use a benchmark or evaluation framing unless held-out gains and the proposed interpretability claims are supported by completed experiments.

## Abstract scaffold

State the drought monitoring problem and define SPEI-3/6/12 as accumulation scales with T+1…T+4 forecast leads. Describe site/month inputs, explicit temporal and regional splits, baselines, probabilistic calibration, and the metrics. Insert only measured results with uncertainty across at least five seeds. If skill is not better than persistence or gradient boosting, report that result and frame the contribution around calibration, transfer, or reproducibility only where demonstrated.

## Sections

### 1. Introduction

Motivate early warning across drought accumulation scales; explain why autocorrelation makes persistence a necessary benchmark. Contributions must be conditional on evidence.

### 2. Related work

Cover drought indices and SPEI, statistical persistence/climatology/AR models, deep temporal forecasting, multi-task learning, probabilistic forecasting and conformal calibration, and cross-region climate evaluation.

**Gap to test:** many drought ML studies use in-sample or randomly split windows, inconsistent lead/scale terminology, and incomplete persistence comparisons. Verify this claim with a systematic literature review before asserting novelty.

### 3. Problem and data

For site (s), month (t), climate and teleconnection vector (x_{s,t}), and SPEI scale (q\in\{3,6,12\}), predict (y_{s,t+h}^{(q)}) for (h\in\{1,2,3,4\}). Document source versions, units, coordinate handling, PET assumptions, index transforms, missingness, and regional site selection. Distinguish computed SPEI from official SPEIbase.

### 4. Method

The shared encoder is LSTM(64), dropout, LSTM(32). The diagram-aligned default uses one attention distribution shared across all task heads:

\[
\alpha_j=\frac{\exp(v^\top\tanh(W h_j))}{\sum_{k=1}^{L}\exp(v^\top\tanh(W h_k))},\qquad
c=\sum_{j=1}^{L}\alpha_jh_j.
\]

The same context (c) is supplied to the SPEI-3/6/12 decoders. Optional variable-gate, head-specific attention, GRU, and Transformer settings are ablations, not the main architecture. The reported baseline run uses point outputs and squared-error loss. An optional three-quantile head, pinball loss, and validation-only CQR evaluation are implemented, but no results from these options should be claimed until fresh multi-seed runs are completed.

### 5. Protocol

Chronological train through 2010-12, validation 2012–2015, and test from 2017, with stated gaps; only windows fully contained in one partition are eligible. Fit per-site normalizers on training rows only. Add region holdouts and event-period analyses only after their split logic is implemented and frozen. Run at least five fixed seeds.

### 6. Results

Table RMSE/MAE/NSE/KGE-2012, persistence skill, and drought-event POD/FAR/CSI/F1 by scale and lead. Report mean ± standard deviation across five seeds, site-year paired bootstrap confidence intervals, panel-month HAC Diebold–Mariano tests with Harvey correction, sample counts, and all baselines. For quantile runs, include quantile CRPS approximation and pre/post CQR interval coverage and width.

### 7. Ablations and interpretability

Compare attention modes, feature gate, shared/separate encoders, fixed/uncertainty task weights, extreme weighting, point/quantile losses, input groups, sequence lengths, and recurrent/Transformer baselines. Use Integrated Gradients and attention summaries across seeds and known events. Attention and gate weights are not causal evidence; any ENSO/IOD recovery claim requires predeclared tests against known regional teleconnections.

### 8. Limitations

Cover point-site sampling, approximate PET, computed SPEI calibration, data source revisions, missing official SPEIbase comparison, regional representativeness, few extreme events, autocorrelation, transfer limits, and non-causal explanations.

### 9. Broader impact

Discuss potential early-warning value, false alarms and missed droughts, unequal regional data quality, uncertainty communication, and the need for local validation before operational use.

### 10. Reproducibility checklist

Record commit, data provenance and hashes, split dates, seed list, preprocessing metadata, configs, checkpoint selection, package/container versions, compute, all baseline code, and the exact scripts used to produce tables/figures.

## Venue strategy

- **Main-track ML (NeurIPS/ICML/ICLR/KDD/AAAI):** requires a clear methodological contribution and strong multi-region, multi-seed evidence against tuned classical and deep baselines.
- **Applied ML / AI for social good:** suitable if evaluation is rigorous and the operational decision context is concrete.
- **Climate Change AI workshop:** reasonable for an early reproducible benchmark with honest limitations.
- **AGU/EGU or a hydrology/climate journal:** better fit if data provenance, physical index validation, regional interpretation, and domain collaboration dominate.

## Likely reviewer concerns and evidence needed

1. **Is SPEI valid?** Compare computed values to official SPEIbase per site/scale; report correlation and RMSE, and fully disclose PET approximation. A NetCDF/CSV validator is available, but this comparison remains to be run.
2. **Does the model beat simple methods?** Evaluate persistence, seasonal climatology, Ridge, and boosted trees on identical windows; report losses honestly.
3. **Does attention add value?** Include shared/no-attention ablations and uncertainty across seeds.
4. **Does it transfer geographically?** The loader now supports source-train-only leave-region-out evaluation and source-fitted normalization. Run at least five seeds for every held-out region before making transfer claims; one Australia smoke run is exploratory only.
5. **Are uncertainty intervals calibrated?** Show validation calibration, untouched test coverage and width, and event-conditioned coverage.
6. **Are climate links causal?** Avoid causal wording; compare stable attributions against known events and independent domain evidence.
7. **Are sample sizes adequate?** Show site-by-period counts, effective event counts, paired uncertainty intervals, and limitations from overlapping forecast windows.
