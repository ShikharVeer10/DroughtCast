import numpy as np

from utils.metrics import calculate_kge, drought_event_metrics
from utils.stats import count_drought_episodes, dm_hac, paired_block_bootstrap_rmse_ci


def test_kge_stable_for_zero_mean_index():
    y = np.array([-1.0, 0.0, 1.0, 0.0])
    assert np.isfinite(calculate_kge(y, y))
    assert calculate_kge(y, y) == 1.0
    assert np.isfinite(calculate_kge(np.zeros_like(y), y))


def test_drought_event_confusion_scores():
    result = drought_event_metrics(np.array([-1.5, 0.0, -0.5]), np.array([-1.2, -1.1, 0.2]))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)
    assert result["pod"] == 0.5 and result["far"] == 0.0


def test_paired_statistics_return_finite_values():
    y = np.linspace(-1, 1, 40)
    pred = y + 0.2
    baseline = y + 0.5
    dm = dm_hac((pred-y)**2, (baseline-y)**2, horizon=2)
    ci = paired_block_bootstrap_rmse_ci(pred, baseline, y, np.repeat(np.arange(8), 5), n_boot=100)
    assert np.isfinite(dm["statistic"])
    assert ci["rmse_difference_ci95"][1] < 0


def test_dm_hac_bandwidth_at_least_accumulation_scale():
    loss = np.sin(np.arange(30, dtype=float))
    result = dm_hac(loss, np.zeros_like(loss),
                    horizon=1, accumulation_scale=12)
    assert result["hac_bandwidth"] == 12


def test_counts_independent_drought_runs_by_site_and_contiguity():
    metadata = [
        {"site": "a", "target_dates": ["2017-01-01"]},
        {"site": "a", "target_dates": ["2017-02-01"]},
        {"site": "a", "target_dates": ["2017-04-01"]},
    ]
    targets = np.array([[[-1.2]], [[-1.4]], [[-1.1]]])
    mask = np.ones_like(targets, dtype=bool)
    assert count_drought_episodes(metadata, targets, mask, 0) == 2
