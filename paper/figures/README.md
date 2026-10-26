# Paper figures

Regenerate with `python make_paper_figures.py --figure F1`; audit all ten inputs with `python make_paper_figures.py --check`. The figures currently on disk were generated from the working tree based on commit `a85e9ae`.

| Figure | Content | Source files | Regeneration command | Git commit | Status |
|---|---|---|---|---|---|
| F1 | Six-phase reference architecture | `configs/base_diagram.yaml` | `python make_paper_figures.py --figure F1` | `a85e9ae` | Generated as PDF, 300-dpi PNG, and exact plotted rows in `data/F1_architecture.csv`. |
| F2 | Site map colored by T+4 skill vs persistence | `results/site_skill_t4.csv`, `configs/config.yaml`, Natural Earth coastlines | `python make_paper_figures.py --figure F2` | `a85e9ae` | Per-site skill is available; blocked because Cartopy and local Natural Earth coastline data are unavailable. |
| F3 | Skill by forecast lead, scale, and method with five-seed 95% CI | `results/per_seed.csv`, `results/run_manifest.json` | `python make_paper_figures.py --figure F3` | `a85e9ae` | Blocked until results are regenerated under the current dataset alignment protocol. |
| F4 | Named drought episode forecasts and quantile bands | `results/case_study_predictions.csv` | `python make_paper_figures.py --figure F4` | `a85e9ae` | Blocked; no timestamped forecast quantile export is present. |
| F5 | Reliability and interval coverage by lead | `results/reliability_and_coverage.csv` | `python make_paper_figures.py --figure F5` | `a85e9ae` | Blocked; current calibration files do not provide CQR coverage by lead in the required joined schema. |
| F6 | POD, FAR, CSI by threshold, scale, and lead with bootstrap intervals | `results/drought_event_bootstrap.csv` | `python make_paper_figures.py --figure F6` | `a85e9ae` | Blocked; current drought summaries lack lead-specific bootstrap intervals. |
| F7 | Ablation RMSE deltas against base_diagram with 95% CI | `results/ablation_forest.csv` | `python make_paper_figures.py --figure F7` | `a85e9ae` | Blocked; only one ablation variant is present and no base_diagram comparison row exists. |
| F8 | Attention by lead, scale, season and validation occlusion | `results/attention_occlusion.csv` | `python make_paper_figures.py --figure F8` | `a85e9ae` | Blocked; current interpretability export is aggregate and uses retired `lag_spei_*` feature names. |
| F9 | Computed SPEI vs SPEIbase scatter | `results/speibase_validation_points.csv`, `results/speibase_validation.csv` | `python make_paper_figures.py --figure F9` | `a85e9ae` | Blocked; SPEIbase reference files were not available to produce the validation table. |
| F10 | Leave-region-out skill by held-out region | `results/leave_region_out_skill.csv` | `python make_paper_figures.py --figure F10` | `a85e9ae` | Blocked; current holdout artifact has only South Asia. |

After regenerating any figure, update its row with the source revision from `git rev-parse HEAD`.
