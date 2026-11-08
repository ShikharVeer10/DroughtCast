import numpy as np
import pandas as pd

from experiments.phase2_3_diagnostics import (
    block_bootstrap_metrics,
    classification_summary,
    drought_categories,
    event_summary,
    matched_sample_counts,
    quantile_summary,
    regression_summary,
    site_metric_summary,
    training_history_summary,
)


def test_drought_category_boundaries_and_missing_values():
    values = np.array([-2.1, -2, -1.5, -1, -0.999, 1, 1.5, 2, np.nan])
    assert drought_categories(values).tolist() == [0, 0, 1, 2, 3, 4, 5, 6, -1]


def test_regression_metrics_and_missing_pair_mask():
    metrics = regression_summary(np.array([1.0, 2.0, np.nan]), np.array([1.5, 2.0, 4.0]))
    assert metrics["n"] == 2
    assert metrics["mae"] == 0.25
    assert metrics["bias"] == 0.25


def test_severe_event_metrics_are_not_severe_category_metrics():
    result = event_summary(np.array([-2.2, -1.7, -1.2, 0.0]), np.array([-1.7, -1.2, -1.7, 0.0]))
    assert result["actual_events"] == 2
    assert result["predicted_events"] == 2
    assert result["tp"] == 1 and result["fn"] == 1 and result["fp"] == 1
    classes = classification_summary(np.array([-2.2, -1.7, -1.2]), np.array([-2.2, -1.2, -1.7]))
    assert classes["severe_category_actual"] == 1
    assert classes["severe_category_predicted"] == 1
    assert classes["severe_category_recall"] == 0


def test_quantile_coverage_crossing_and_pinball():
    y = np.array([0.0, 1.0, -1.0])
    q = np.array([[-1.0, 0.0, 1.0], [0.0, 0.5, 1.5], [-0.5, -1.5, 0.0]])
    result = quantile_summary(y, q)
    assert result["crossing_count"] == 1
    assert result["n"] == 3
    assert 0 <= result["q10_q90_coverage"] <= 1
    assert result["pinball_q50"] >= 0


def test_two_year_block_bootstrap_is_reproducible():
    frame = pd.DataFrame({
        "target_year": np.repeat(np.arange(2017, 2022), 2),
        "q50": np.arange(10, dtype=float),
        "observed": np.arange(10, dtype=float) + np.tile([0.0, 1.0], 5),
    })
    a = block_bootstrap_metrics(frame, value_col="q50", target_col="observed", block_years=2, resamples=50, seed=7)
    b = block_bootstrap_metrics(frame, value_col="q50", target_col="observed", block_years=2, resamples=50, seed=7)
    assert a == b
    assert a["status"] == "ok"
    assert a["n_years"] == 5


def test_absent_class_is_retained_in_class_counts():
    result = classification_summary(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
    assert result["class_counts_actual"] == [0, 0, 0, 3, 0, 0, 0]
    assert len(result["confusion_matrix"]) == 7


def test_history_recomputes_best_checkpoint_from_validation_only_score():
    history = pd.DataFrame({
        "profile": ["p"] * 3,
        "seed": [42] * 3,
        "epoch": [1, 2, 3],
        "validation_primary_site_mean_mse_skill": [0.4, 0.3, 0.35],
        "best_epoch_so_far": [1, 1, 1],
        "train_pinball_loss": [2.0, 1.5, 1.0],
    })
    result = training_history_summary(history).iloc[0]
    assert result.best_epoch_recomputed == 1
    assert result.best_epoch_recorded == 1
    assert result.training_loss_reduced
    assert result.validation_pinball_loss == "NOT_SAVED"


def test_site_aggregation_preserves_site_level_spread():
    frame = pd.DataFrame({"site": ["a", "a", "b", "ALL"], "rmse": [1.0, 3.0, 2.0, 9.0]})
    summary = site_metric_summary(frame, "rmse").set_index("site")
    assert summary.loc["a", "mean"] == 2.0
    assert summary.loc["b", "mean"] == 2.0
    assert "ALL" not in summary.index


def test_sample_count_comparison_flags_missing_or_mismatched_groups():
    reference = pd.DataFrame({"site": ["a", "b"], "n": [10, 9]})
    candidate = pd.DataFrame({"site": ["a", "c"], "n": [10, 9]})
    result = matched_sample_counts(reference, candidate, ("site",), "n").set_index("site")
    assert bool(result.loc["a", "same_n"])
    assert not bool(result.loc["b", "same_n"])
    assert not bool(result.loc["c", "same_n"])
