import numpy as np
import pandas as pd

from experiments.phase5_7_robustness import (
    adjust_bh,
    block_keys,
    cluster_bootstrap,
    percentile_interval,
)


def _frame():
    rows = []
    for site in ("a", "b"):
        for origin in ("2013-01", "2013-02"):
            for lead in (1, 2):
                rows.append({"site": site, "forecast_origin": origin, "observed": 0.0, "prediction": 1.0, "lead": lead})
    return pd.DataFrame(rows)


def test_blocks_and_resampling_are_reproducible():
    frame = _frame()
    assert len(block_keys(frame)) == 4
    first = cluster_bootstrap(frame, "mae", replicates=20, seed=3)
    second = cluster_bootstrap(frame, "mae", replicates=20, seed=3)
    assert np.array_equal(first, second)
    assert percentile_interval(first) == percentile_interval(second)


def test_small_and_multiple_testing_behaviors():
    interval = percentile_interval(np.array([1.0]))
    assert np.isnan(interval[0]) and np.isnan(interval[1])
    assert adjust_bh([0.01, 0.04, 0.2])[0] <= 0.03
