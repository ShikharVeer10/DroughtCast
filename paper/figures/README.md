# Figure generation

The current figure entry point is `make_paper_figures.py`; regenerate available figures with:

```powershell
python make_paper_figures.py --all --strict
```

Use `python make_paper_figures.py --check` to list missing or schema-inadequate inputs. Figures F2–F10 require current chronological test/evaluation exports under `results/`; they are skipped when those files are absent. The generator writes PDF, PNG, and plotted-data CSV files under this directory. Do not treat legacy pre-fix outputs as current results.

| Figure | Main input files |
|---|---|
| F1 architecture | `configs/base_diagram.yaml` |
| F2 site skill map | `results/site_skill_t4.csv`, `configs/config.yaml`, `results/evaluation_manifest.json` |
| F3 skill by lead | `results/per_seed.csv`, `results/run_manifest.json` |
| F4 case-study forecasts | `results/case_study_predictions.csv`, `results/run_manifest.json` |
| F5 reliability and coverage | `results/reliability_and_coverage.csv`, `results/run_manifest.json` |
| F6 drought-event metrics | `results/drought_event_bootstrap.csv`, `results/run_manifest.json` |
| F7 ablations | `results/ablation_forest.csv`, `results/run_manifest.json` |
| F8 attention and occlusion | `results/attention_occlusion.csv`, `results/run_manifest.json` |
| F9 SPEIbase validation | `results/speibase_validation_points.csv`, `results/speibase_validation.csv` |
| F10 leave-region-out | `results/leave_region_out_skill.csv`, `results/run_manifest.json` |

Probability results are tabulated separately by `python experiments/export_probability_table.py`; this uses `results/inference_probabilities_seed_42.csv` and writes `paper/latex/probability_table.tex` and `.csv`.

Repository revision inspected for this documentation: `d51f9ce`.
