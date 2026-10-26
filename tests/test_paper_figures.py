import ast
import importlib.util
import json
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from paper.figure_style import DOUBLE_COLUMN_IN, SINGLE_COLUMN_IN

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("make_paper_figures", ROOT / "make_paper_figures.py")
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


def test_missing_input_audit_reports_commands_and_creates_no_images(monkeypatch, capsys):
    with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
        tmp_path = Path(temp_dir)
        monkeypatch.setattr(figures, "ROOT", tmp_path)
        monkeypatch.setattr(figures, "OUT", tmp_path / "paper" / "figures")
        monkeypatch.setattr(figures, "DATA", figures.OUT / "data")
        monkeypatch.setattr(figures, "PROVENANCE", figures.OUT / "provenance")
        monkeypatch.setattr("sys.argv", ["make_paper_figures.py", "--check"])
        assert figures.main() != 0
        output = capsys.readouterr().out
        assert "results/test_predictions_seed_*.csv" in output
        assert "run python evaluate.py" in output
        assert not list((tmp_path / "paper").rglob("*.pdf"))
        assert not list((tmp_path / "paper").rglob("*.png"))


def test_figure_code_has_no_random_number_or_random_module_usage():
    paths = [ROOT / "make_paper_figures.py", ROOT / "paper" / "figure_style.py"]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "np.random" not in source
        assert not re.search(r"\bimport\s+random\b|\brandom\.", source)


def test_ieee_width_constants_and_saved_figures_have_data_and_provenance(monkeypatch):
    assert SINGLE_COLUMN_IN == pytest.approx(3.5)
    assert DOUBLE_COLUMN_IN == pytest.approx(7.16)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.0))
    assert fig.get_size_inches()[0] == pytest.approx(3.5)
    with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
        tmp_path = Path(temp_dir)
        monkeypatch.setattr(figures, "OUT", tmp_path)
        monkeypatch.setattr(figures, "DATA", tmp_path / "data")
        monkeypatch.setattr(figures, "PROVENANCE", tmp_path / "provenance")
        monkeypatch.setattr(figures, "ROOT", ROOT)
        figures._save(fig, "test_figure", pd.DataFrame({"value": [1.0]}))
        assert (tmp_path / "test_figure.pdf").is_file()
        assert (tmp_path / "test_figure.png").is_file()
        assert (tmp_path / "data" / "test_figure.csv").is_file()
        provenance = json.loads((tmp_path / "provenance" / "test_figure.json").read_text())
        assert "git_commit" in provenance
        assert "config_sha256" in provenance
