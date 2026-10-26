.PHONY: results figures paper-assets probability-table

results:
	python experiments/run_all.py --seeds 42 101 2024 2025 2026

figures:
	python make_paper_figures.py --all --strict

paper-assets:
	python make_paper_figures.py --all --strict
	python experiments/export_probability_table.py

probability-table:
	python experiments/export_probability_table.py
