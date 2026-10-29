import os
from pathlib import Path

import pandas as pd
import pytest

from experiments.export_probability_table import _assert_current


def test_assert_current_rejects_isotonic_calibration(tmp_path, monkeypatch):
    path = tmp_path / "probabilities.csv"
    pd.DataFrame({"calibration": ["isotonic"]}).to_csv(path, index=False)
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter(()))

    with pytest.raises(SystemExit, match="not produced by the Platt pipeline"):
        _assert_current(path)


def test_assert_current_accepts_newer_validation_only_platt_csv(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_seed_42.pt"
    checkpoint.touch()
    os.utime(checkpoint, (0, 0))
    path = tmp_path / "probabilities.csv"
    pd.DataFrame({"calibration": ["validation-only Platt scaling"]}).to_csv(
        path, index=False
    )
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda self, pattern: iter([checkpoint])
        if str(self) == "results/ablations/quantile_cqr"
        else iter(()),
    )

    _assert_current(path)
