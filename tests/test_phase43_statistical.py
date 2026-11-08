import numpy as np
import pandas as pd

from evaluation.statistical import (
    BootstrapConfig,
    holm_adjust,
    paired_cluster_bootstrap,
)


def test_cluster_bootstrap_is_reproducible_and_paired():
    rows = []
    for site in ("a", "b", "c"):
        for model, value in (("a", 1.0), ("b", 2.0)):
            rows.append({"site": site, "target_month": "2026-11", "model": model,
                         "prediction": value, "observed": 0.0})
    frame = pd.DataFrame(rows)
    config = BootstrapConfig(iterations=100, seed=7)
    first = paired_cluster_bootstrap(frame, "a", "b", config=config)
    second = paired_cluster_bootstrap(frame, "a", "b", config=config)
    assert first == second


def test_holm_preserves_nan_and_orders_adjustments():
    result = holm_adjust(np.array([0.01, 0.04, np.nan]))
    assert np.isnan(result[2])
    assert result[0] <= result[1]


def test_bootstrap_warns_on_insufficient_blocks():
    frame = pd.DataFrame([
        {"site": "a", "target_month": "2026-11", "model": "a", "prediction": 1, "observed": 0},
        {"site": "a", "target_month": "2026-11", "model": "b", "prediction": 2, "observed": 0},
    ])
    result = paired_cluster_bootstrap(frame, "a", "b", config=BootstrapConfig(iterations=10))
    assert np.isnan(result["estimate"])
    assert result["warning"]
