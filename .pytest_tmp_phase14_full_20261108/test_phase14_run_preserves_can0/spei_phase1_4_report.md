# Phase 1.4 — SPEI methodological validation and discrepancy attribution

This analysis reads the canonical dataset and existing Phase 1.2/1.3 artifacts. It does not modify canonical SPEI, the config, features, model code, or previous outputs; no model was run.

## Method comparison

| Area | Local implementation | Official SPEI/SPEIbase behavior | Classification |
|---|---|---|---|
| water balance and units | POWER PRECTOTCORR monthly mean rate (mm/day) x days_in_month - Hargreaves-Samani monthly PET total (mm/month) | CRU TS precipitation (mm/month) - CRU PET (mm/day) x month days in SPEIbase functions.R | mathematically equivalent unit conversion; data and PET methods intentionally differ |
| trailing accumulation | unweighted trailing rolling sum, full scale window required | default rectangular kernel, shift 0, right-aligned weighted sum over scale months | equivalent for complete regular monthly input |
| calendar-month strata | separate calibration sample for each calendar month | separate cycle/month fit for each time series | equivalent |
| calibration period | fixed 1990-01 through 2010-12 | SPEIbase invokes SPEI::spei without ref.start/ref.end, so package defaults use all supplied observations (v2.11 through 2024) | intentional methodological alternative |
| distribution and parameter fit | canonical uses empirical plotting positions p=(rank-0.44)/(n+0.12), inverse-normal scores, linear tail extrapolation; no fitted parametric distribution | log-Logistic (generalized logistic type III) with ub-pwm by default | intentional methodological alternative; not equivalent to official SPEI |
| probability transform | empirical score interpolation/extrapolation then normal score; Fisk alternative uses its fitted CDF; final local score clipping | log-Logistic CDF followed by qnorm; SPEIbase calls na.rm=TRUE | normal-score end step equivalent in concept; underlying CDF differs |
| zero, negative, and support | canonical empirical transform accepts real water balances including negatives and zeros; alternative Fisk maps values at/below its estimated location to probability floor 1e-6 | log-Logistic has a fitted location parameter and can represent negative balance; zero-mass correction is applied to Gamma/PearsonIII, not default log-Logistic | intentional distribution behavior difference |
| missing values and spin-up | rolling sum requires every month in the window; NaNs propagate; calibration drops missing accumulated values; first scale-1 outputs are missing | spei.nc calls SPEI::spei(..., na.rm=TRUE); rolling windows are right aligned; fit excludes NA values | equivalent in window propagation on regular monthly inputs; local 20-sample minimum is stricter than package's four-sample early guard |
| fit failures | canonical empirical transform has no optimizer; less than 20 calibration observations leaves missing; Fisk failures are stored in diagnostics and target rows are missing | package requires nonzero spread and at least 4 observations, checks L-moment validity and may return missing values | intentional alternative plus different failure thresholds; no silent Fisk drops in stored alternative diagnostics |
| extreme clipping | canonical empirical and Fisk scores clipped at +/-3.5 | SPEIbase SPEI::spei standardizes qnorm(CDF) without a fixed +/-3.5 clipping step | intentional implementation choice, material tail censoring measured below |
| alternative Fisk PWM estimator | three-parameter Fisk PWM based on plotting positions (rank-0.35)/n | ub-pwm by default; pp-pwm is a separate supported method | intentional alternative estimator; not official default |

The local nonparametric transform is an empirical standardization: it maps sample plotting positions to normal scores and interpolates/extrapolates values. This is a defensible alternative for a fixed historical reference, but it is not the parametric log-logistic SPEI estimator used by the official package. It will respond differently to skew, tail shape, and values outside the calibration sample. The short monthly calibration sample makes its empirical tails especially sensitive; linear tail extrapolation followed by clipping further limits extreme-value interpretation.

The official SPEI package defaults to log-Logistic with unbiased PWM, rectangular trailing accumulation, and all supplied observations unless `ref.start`/`ref.end` are provided. SPEIbase calls it with `na.rm=TRUE`. The local fixed 1990–2010 calibration is intentional for a forecasting protocol; it differs from the v2.11 all-series fit. Fisk alternatives are closer in family but use the project's plotting-position PWM equations rather than SPEI's ub-pwm default.

**Defect assessment:** no confirmed defect was found in the configured canonical nonparametric path. A potential API issue remains: an unrecognized `method` passed to `_standardize_accumulation` falls through to its custom log-logistic branch instead of raising. It is not used by the configured canonical method and was not changed. The alternative Fisk fit fails for 53 calendar/site/scale combinations; those cases are explicitly present as failed in diagnostics and missing in the alternative targets. The generalized-logistic fit's convergence flags do not establish stable shape estimates: 166/432 shapes exceed 100 and the maximum is 2,136.6.

## Clipping sensitivity

| Scale | Valid local n | Below -3.5 | Above +3.5 | Clipped % | Uncapped min/max | Category changes | Pearson change vs ref | Bias change | RMSE change | Category-agreement change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 5280 | 14 | 25 | 0.739% | -10.800/9.240 | 0 | 0.01762 | -0.00197 | -0.04842 | +0.00000 |
| 6 | 5244 | 19 | 14 | 0.629% | -6.563/7.747 | 0 | 0.00668 | -0.00015 | -0.01777 | +0.00000 |
| 12 | 5172 | 21 | 9 | 0.580% | -7.032/4.864 | 0 | 0.00557 | +0.00193 | -0.01506 | +0.00000 |

Counts are over local site-month targets; metric deltas use exact paired SPEIbase v2.11 months. The seven drought/wet category boundaries are unchanged by ±3.5 clipping, so clipping can change numerical RMSE/bias/correlation while leaving categories identical. See the CSV for per-site metrics and full pre/post results.

## Top 30 disagreement reconstruction

The selected top 30 include 8 sites, all three scales, 14 years, and both signs of local-minus-reference difference. 25 of 30 accumulated values lie outside their calendar-month 1990–2010 calibration ranges; 0 contain missing local precipitation/PET inputs. Each CSV row contains the scale-window precipitation rates/totals, PET, monthly water balance, accumulated balance, calibration n/range, empirical tail slopes, local uncapped/clipped score, reference value, coordinates, and discrepancy.

| Rank | Site | Scale | Month | Accumulated balance (mm) | Calibration range (mm) | Raw SPEI | Clipped SPEI | v2.11 | Absolute difference | Outside calibration? |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | beed | 6 | 2019-09 | -925.5 | -838.2 to -282.4 | -3.635 | -3.500 | 1.545 | 5.045 | True |
| 2 | nanded | 6 | 2019-09 | -684.9 | -631.7 to -105.2 | -3.625 | -3.500 | 0.805 | 4.305 | True |
| 3 | mumbai | 3 | 2021-04 | -602.6 | -679.4 to -621.7 | 2.846 | 2.846 | -1.433 | 4.279 | True |
| 4 | ratnagiri | 3 | 1993-02 | -374.7 | -384.4 to -243.8 | -0.927 | -0.927 | -5.178 | 4.251 | False |
| 5 | nashik | 3 | 2021-03 | -498.3 | -565.6 to -512.6 | 3.211 | 3.211 | -0.988 | 4.199 | True |
| 6 | beed | 6 | 2019-08 | -1179.7 | -1024.3 to -598.8 | -4.546 | -3.500 | 0.530 | 4.030 | True |
| 7 | beed | 3 | 2019-08 | -390.2 | -296.0 to 42.9 | -3.163 | -3.163 | 0.845 | 4.008 | True |
| 8 | mumbai | 3 | 2021-03 | -528.9 | -580.8 to -537.9 | 2.389 | 2.389 | -1.552 | 3.941 | True |
| 9 | beed | 12 | 2019-09 | -1994.4 | -1831.5 to -1143.5 | -2.824 | -2.824 | 1.033 | 3.857 | True |
| 10 | mumbai | 3 | 2022-03 | -532.2 | -580.8 to -537.9 | 2.224 | 2.224 | -1.523 | 3.747 | True |

For Beed September 2019, the six-month local balance is -925.462 mm versus a September calibration range of -838.170 to -282.368 mm; it is below range. The reconstructed raw empirical score is -3.635126, clipped to -3.500, versus v2.11 1.544839. This establishes that the local six-month balance is outside its local calibration tail and that the local score is censored. It does not reveal the CRU source balance; the sign discrepancy remains unexplained.

## Precipitation and PET input availability

The local cached input is NASA POWER monthly data. Precipitation is `PRECTOTCORR` in mm/day and converted to monthly total by multiplying by days in month. PET is Hargreaves–Samani from monthly mean temperatures/extrema, multiplied to a monthly total. Existing caches do not contain CRU TS input grids, ERA5-Land or CHIRPS point series. The v2.11 artifact contains matched SPEI values but not the underlying precipitation/PET.

A small independent input comparison was therefore **not numerically possible** in this run. Official candidate sources were identified: CHIRPS v3 provides 0.05° precipitation from 1981 to near-present, but it blends satellite and station data and its v3 gauge undercatch correction changes levels; ERA5-Land monthly means provide precipitation and potential evaporation on the 0.1° CDS grid from 1950 onward, but potential evaporation's sign/definition must be checked before treating it as crop-reference PET. CRU TS v4.09 provides `pre` (mm/month) and `pet` (mm/day) at 0.5° and is the source family for v2.11, so it can help reproduce the reference inputs but is not an independent validation product. No source values were fabricated or downloaded.

## Method sensitivity and drought-category stability

The method-sensitivity CSV compares the existing 1981–2010 nonparametric, 1990–2010 Fisk, and 1981–2010 Fisk targets against canonical targets on common valid months, per site/scale and pooled. The category CSV reports canonical/reference, alternative/reference, and clipped/unclipped seven-class, drought-only, and extreme-drought agreement. Failed Fisk fits stay missing, and sample counts are explicit. Summary metrics below are pooled `ALL` rows:

| Alternative | Scale | n | Pearson r vs canonical | Bias | MAE | RMSE | Category agreement | Drought false positives | Missed comparator droughts | Extreme-event disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nonparametric_1981_2010 | 3 | 5280 | 0.9866 | +0.0414 | 0.1420 | 0.1850 | 0.8345 | 80 | 62 | 111 |
| fisk_1990_2010 | 3 | 4548 | 0.8718 | -0.2045 | 0.5412 | 0.9237 | 0.6025 | 513 | 8 | 573 |
| fisk_1981_2010 | 3 | 4878 | 0.9123 | -0.0802 | 0.4210 | 0.6716 | 0.6378 | 384 | 30 | 413 |
| nonparametric_1981_2010 | 6 | 5244 | 0.9851 | +0.0674 | 0.1518 | 0.2019 | 0.8251 | 72 | 102 | 104 |
| fisk_1990_2010 | 6 | 4810 | 0.9042 | -0.1038 | 0.4051 | 0.7138 | 0.6468 | 459 | 3 | 410 |
| fisk_1981_2010 | 6 | 5062 | 0.9191 | +0.0089 | 0.3423 | 0.5597 | 0.6750 | 305 | 54 | 303 |
| nonparametric_1981_2010 | 12 | 5172 | 0.9840 | +0.0841 | 0.1623 | 0.2166 | 0.7956 | 49 | 138 | 93 |
| fisk_1990_2010 | 12 | 4992 | 0.9751 | -0.0184 | 0.2546 | 0.3497 | 0.6745 | 309 | 1 | 230 |
| fisk_1981_2010 | 12 | 5172 | 0.9743 | +0.1016 | 0.2267 | 0.2870 | 0.7253 | 126 | 72 | 133 |

| Scale | Canonical vs v2.11 pairs | Seven-class agreement | Drought-only agreement | False local droughts | Missed v2.11 droughts | Extreme-drought agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 5014 | 0.5901 | 0.3365 | 314 | 512 | 0.0068 |
| 6 | 4980 | 0.6022 | 0.3884 | 280 | 397 | 0.0000 |
| 12 | 4908 | 0.5896 | 0.3406 | 321 | 378 | 0.0098 |

## Limitations and recommendation

Numerical agreement with v2.11 does not validate either product as truth. Local/reference input differences remain unmeasured; v2.12 is not available in this workspace. The canonical fixed reference has only about 20–21 values per calendar month, does not use the official default log-logistic ub-pwm fit, and clips tails. The signed Beed disagreement is partially characterized but not causally explained. Category robustness should be read from per-site/per-scale values, not pooled rows alone.

**Recommendation: RETAIN canonical targets unchanged, but do not yet regard them as cleared for research-grade model training or strong performance claims.** A future approved phase should retrieve raw/reference input data or another independent precipitation/PET source, quantify PET sensitivity, and decide whether a conventional log-logistic target should be added as a parallel protocol. This phase makes no automatic target change.

## Artifacts and verification

Generated: `spei_methodology_comparison.json`, `spei_clipping_sensitivity.csv`, `spei_top30_discrepancy_reconstruction.csv`, `spei_input_data_comparison.csv`, `spei_method_sensitivity.csv`, `spei_category_stability.csv`, and this report. Tests and Ruff results are recorded in the final execution summary.

Sources: [SPEIbase functions.R](https://github.com/sbegueria/SPEIbase/blob/master/R/functions.R), [SPEI package implementation](https://github.com/sbegueria/SPEI/blob/master/R/spei.R), [SPEI CRAN manual](https://cran.r-project.org/web/packages/SPEI/SPEI.pdf), [CRU TS variables and units](https://crudata.uea.ac.uk/cru/data/hrg/), [CHIRPS v3 documentation](https://www.chc.ucsb.edu/data/chirps3), [ERA5-Land dataset documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means).
