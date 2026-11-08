import csv
import inspect
import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.run_phase2_2_neural import _train_epoch, run
from experiments.training_diagnostics import TrainingDiagnostics, make_diagnostics


def test_diagnostics_are_disabled_by_default():
    assert inspect.signature(_train_epoch).parameters["diagnostics"].default is None
    assert inspect.signature(run).parameters["diagnostics_dir"].default is None
    assert make_diagnostics(None) is None


def test_epoch_metrics_are_detached_and_include_scale_lead_diagnostics(tmp_path):
    observer = TrainingDiagnostics(
        tmp_path,
        run_id="synthetic",
        profile="test-only-fixture",
        seed=7,
        effective_config={"model": {"variable_gate": False}},
        extreme_weight=1.5,
        extreme_threshold=-1.0,
    )
    targets = torch.zeros((2, 3, 4))
    mask = torch.ones((2, 3, 4), dtype=torch.bool)
    predictions = {}
    for scale_index, scale in enumerate(("spei_3", "spei_6", "spei_12")):
        for lead in range(4):
            targets[:, scale_index, lead] = torch.tensor(
                [10 * scale_index + lead, 10 * scale_index + lead + 2]
            )
        q = torch.stack(
            [targets[:, scale_index, :] - 1, targets[:, scale_index, :] + 0.5,
             targets[:, scale_index, :] + 1], dim=-1
        )
        predictions[scale] = q.clone().requires_grad_()
    targets[0, 0, 0] = -1.5
    # Deliberately cross one synthetic interval to verify crossing accounting.
    with torch.no_grad():
        predictions["spei_3"][0, 0] = torch.tensor([-2.0, -1.0, 0.0])
        predictions["spei_6"][0, 1] = torch.tensor([0.0, -1.0, 1.0])
    original_targets = targets.clone()
    original_predictions = {key: value.detach().clone() for key, value in predictions.items()}

    observer.add_predictions("train", 1, predictions, targets, mask)
    observer.finish_epoch(1)

    torch.testing.assert_close(targets, original_targets)
    for key, value in predictions.items():
        torch.testing.assert_close(value, original_predictions[key])
        assert value.grad is None
    with (tmp_path / "epoch_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    first = next(row for row in rows if row["target"] == "spei_3" and row["lead"] == "T+1")
    assert first["n_valid"] == "2"
    assert first["actual_spei_le_minus_1_5"] == "1"
    assert first["q10_q90_coverage"] == "1.0"
    assert first["diagnostic_status"] == "ok"
    spei6_t3 = next(
        row for row in rows if row["target"] == "spei_6" and row["lead"] == "T+3"
    )
    assert spei6_t3["target_mean"] == "13.0"
    spei6_t2 = next(
        row for row in rows if row["target"] == "spei_6" and row["lead"] == "T+2"
    )
    assert spei6_t2["quantile_crossing_count"] == "1"
    config_records = [
        json.loads(line)
        for line in (tmp_path / "effective_configurations.jsonl").read_text().splitlines()
    ]
    assert config_records[0]["effective_config"]["model"]["variable_gate"] is False


def test_nonfinite_values_are_counted_and_empty_cells_are_explicit(tmp_path):
    observer = TrainingDiagnostics(
        tmp_path,
        run_id="synthetic-missing",
        profile="test-only-fixture",
        seed=9,
        effective_config={},
        extreme_weight=1.0,
        extreme_threshold=-1.5,
    )
    targets = torch.full((1, 3, 4), float("nan"))
    mask = torch.ones((1, 3, 4), dtype=torch.bool)
    predictions = {
        scale: np.full((1, 4, 3), np.nan)
        for scale in ("spei_3", "spei_6", "spei_12")
    }

    observer.add_predictions("validation", 3, predictions, targets, mask)
    observer.finish_epoch(3)
    with (tmp_path / "epoch_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["n_nonfinite_target"] == "1" for row in rows)
    assert all(row["n_nonfinite_prediction"] == "1" for row in rows)
    assert all(row["diagnostic_status"] == "no_valid_observations" for row in rows)
    assert all(row["median_rmse"] == "" for row in rows)


def test_primary_endpoint_diagnostics_use_supplied_predictions_only(tmp_path):
    observer = TrainingDiagnostics(
        tmp_path,
        run_id="synthetic-endpoint",
        profile="test-only-fixture",
        seed=3,
        effective_config={},
        extreme_weight=1.0,
        extreme_threshold=-1.5,
    )
    targets = torch.ones((2, 3, 4))
    mask = torch.ones((2, 3, 4), dtype=torch.bool)
    metadata = [
        {"site": "akola", "last_observed_spei": {"spei_3": 2.0}},
        {"site": "beed", "last_observed_spei": {"spei_3": 3.0}},
    ]
    predictions = {"spei_3": np.zeros((2, 4, 3))}
    dataset = SimpleNamespace(Y=targets, mask=mask, metadata=metadata)

    observer.add_primary_endpoint(4, dataset, predictions, ["akola", "beed"], macro_skill=0.75)

    assert len(observer.endpoint_rows) == 4
    t3_akola = next(row for row in observer.endpoint_rows if row["site"] == "akola" and row["lead"] == "T+3")
    assert t3_akola["model_mse"] == 1.0
    assert t3_akola["persistence_mse"] == 1.0
    assert t3_akola["skill"] == 0.0
    assert t3_akola["site_macro_skill_for_epoch"] == 0.75


def test_gradient_observer_reads_gradients_without_changing_them(tmp_path):
    class DiagnosticOnlyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.lstm1 = torch.nn.Linear(1, 1)
            self.encoder.lstm2 = torch.nn.Linear(1, 1)
            self.attentions = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.decoders = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(3)])

    model = DiagnosticOnlyModule()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients_before = [parameter.grad.clone() for parameter in model.parameters()]
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    observer = TrainingDiagnostics(
        tmp_path,
        run_id="synthetic-gradients",
        profile="test-only-fixture",
        seed=4,
        effective_config={},
        extreme_weight=1.0,
        extreme_threshold=-1.5,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer_before = optimizer.state_dict()

    snapshot = observer.before_optimizer_step(model)
    observer.after_optimizer_step(model, snapshot)
    observer.finish_epoch(1)

    for parameter, gradient, value in zip(
        model.parameters(), gradients_before, parameters_before, strict=True
    ):
        torch.testing.assert_close(parameter.grad, gradient)
        torch.testing.assert_close(parameter, value)
    assert all(row["finite_gradient_steps"] == 1 for row in observer.gradient_totals.values())
    assert optimizer.state_dict() == optimizer_before
    with (tmp_path / "optimization_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        optimization_rows = list(csv.DictReader(stream))
    assert optimization_rows
    assert all(float(row["parameter_update_l2_mean"]) == 0.0 for row in optimization_rows)
