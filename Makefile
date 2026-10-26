.PHONY: results figures paper-assets probability-table probability-audit calibration-compare

results:
	python experiments/run_all.py --seeds 42 101 2024 2025 2026

figures:
	python make_paper_figures.py --all --strict

paper-assets:
	python make_paper_figures.py --all --strict
	python experiments/export_probability_table.py

probability-table:
	python experiments/export_probability_table.py

probability-audit:
	python experiments/audit_probability_outputs.py

calibration-compare:
	python -m experiments.compare_probability_calibration
