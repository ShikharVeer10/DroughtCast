"""Opt-in, read-only epoch diagnostics for the Phase 2.2 training loop."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCALES = ("spei_3", "spei_6", "spei_12")
QUANTILES = (0.1, 0.5, 0.9)


def make_diagnostics(directory: Path | None, *args, **kwargs):
    """Return no observer by default; create one only for an explicit path."""
    if directory is None:
        return None
    return TrainingDiagnostics(directory, *args, **kwargs)


def _array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _empty_totals() -> dict[str, float]:
    return defaultdict(float)


class TrainingDiagnostics:
    """Collect detached summaries without returning values to the training graph."""

    def __init__(
        self,
        directory: Path,
        *,
        run_id: str,
        profile: str,
        seed: int,
        effective_config: dict[str, Any],
        extreme_weight: float,
        extreme_threshold: float,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.profile = profile
        self.seed = int(seed)
        self.extreme_weight = float(extreme_weight)
        self.extreme_threshold = float(extreme_threshold)
        self.epoch_totals: dict[tuple[str, str, int], dict[str, float]] = {}
        self.gradient_totals: dict[str, dict[str, float]] = {}
        self.endpoint_rows: list[dict[str, Any]] = []
        self.step_count = 0
        with (self.directory / "effective_configurations.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps({
                "run_id": run_id,
                "profile": profile,
                "seed": seed,
                "effective_config": effective_config,
            }, allow_nan=False) + "\n")

    def add_predictions(
        self,
        split: str,
        epoch: int,
        predictions: dict[str, Any],
        targets,
        mask,
    ) -> None:
        """Accumulate train or validation summaries from already-computed outputs."""
        y = _array(targets).astype(np.float64, copy=False)
        valid_mask = _array(mask).astype(bool, copy=False)
        for scale_index, scale in enumerate(SCALES):
            q_values = _array(predictions[scale]).astype(np.float64, copy=False)
            if q_values.shape != (len(y), 4, 3):
                raise ValueError(f"Unexpected diagnostic output shape for {scale}: {q_values.shape}")
            for lead in range(4):
                target = y[:, scale_index, lead]
                quantiles = q_values[:, lead, :]
                selected = valid_mask[:, scale_index, lead]
                finite_target = np.isfinite(target)
                finite_prediction = np.isfinite(quantiles).all(axis=1)
                valid = selected & finite_target & finite_prediction
                values = self.epoch_totals.setdefault(
                    (split, scale, lead + 1), _empty_totals()
                )
                values["n_total"] += len(target)
                values["n_masked"] += int((~selected).sum())
                values["n_nonfinite_target"] += int((selected & ~finite_target).sum())
                values["n_nonfinite_prediction"] += int((selected & ~finite_prediction).sum())
                values["n_valid"] += int(valid.sum())
                if not valid.any():
                    continue

                observed = target[valid]
                q = quantiles[valid]
                median = q[:, 1]
                weights = np.where(observed <= self.extreme_threshold, self.extreme_weight, 1.0)
                errors = observed[:, None] - q
                pinball = np.maximum(
                    np.asarray(QUANTILES)[None, :] * errors,
                    (np.asarray(QUANTILES)[None, :] - 1.0) * errors,
                )
                denominator = float(weights.sum())
                values["weight_sum"] += denominator
                values["pinball_sum"] += float((pinball.sum(axis=1) * weights).sum())
                for qi, name in enumerate(("q10", "q50", "q90")):
                    values[f"{name}_pinball_sum"] += float((pinball[:, qi] * weights).sum())
                values["target_sum"] += float(observed.sum())
                values["target_sq_sum"] += float(np.square(observed).sum())
                values["median_sum"] += float(median.sum())
                values["median_sq_sum"] += float(np.square(median).sum())
                values["cross_sum"] += float((observed * median).sum())
                values["abs_error_sum"] += float(np.abs(median - observed).sum())
                values["squared_error_sum"] += float(np.square(median - observed).sum())
                covered = (observed >= q[:, 0]) & (observed <= q[:, 2])
                values["covered_count"] += int(covered.sum())
                values["interval_width_sum"] += float((q[:, 2] - q[:, 0]).sum())
                crossings = (q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])
                values["crossing_count"] += int(crossings.sum())
                actual_event = observed <= -1.5
                predicted_event = median <= -1.5
                values["actual_severe_events"] += int(actual_event.sum())
                values["predicted_severe_events"] += int(predicted_event.sum())
                values["severe_true_positives"] += int((actual_event & predicted_event).sum())

    def add_primary_endpoint(
        self, epoch: int, dataset, predictions: dict[str, np.ndarray], sites: list[str], macro_skill: float
    ) -> None:
        y = dataset.Y.numpy()[:, 0, 2:4]
        mask = dataset.mask.numpy()[:, 0, 2:4].astype(bool)
        median = predictions["spei_3"][:, 2:4, 1]
        site_ids = np.asarray([str(row["site"]) for row in dataset.metadata])
        persistence = np.asarray([
            float(row["last_observed_spei"]["spei_3"]) for row in dataset.metadata
        ])
        for lead_index, lead in enumerate((3, 4)):
            for site in sites:
                valid = (
                    (site_ids == site)
                    & mask[:, lead_index]
                    & np.isfinite(y[:, lead_index])
                    & np.isfinite(median[:, lead_index])
                    & np.isfinite(persistence)
                )
                observed = y[valid, lead_index]
                model_values = median[valid, lead_index]
                baseline_values = persistence[valid]
                model_mse = float(np.mean(np.square(observed - model_values))) if len(observed) else None
                persistence_mse = float(np.mean(np.square(observed - baseline_values))) if len(observed) else None
                skill = (
                    1.0 - model_mse / persistence_mse
                    if model_mse is not None and persistence_mse and persistence_mse > 0
                    else None
                )
                self.endpoint_rows.append({
                    "run_id": self.run_id, "profile": self.profile, "seed": self.seed,
                    "epoch": epoch, "target": "spei_3", "lead": f"T+{lead}",
                    "site": site, "n_valid": len(observed), "model_mse": model_mse,
                    "persistence_mse": persistence_mse, "skill": skill,
                    "site_macro_skill_for_epoch": float(macro_skill),
                })

    def before_optimizer_step(self, model) -> dict[str, list[torch.Tensor]]:
        groups = self._parameter_groups(model)
        snapshots: dict[str, list[torch.Tensor]] = {}
        self.step_count += 1
        for name, parameters in groups.items():
            grads = [p.grad.detach() for p in parameters if p.grad is not None]
            finite = [g for g in grads if torch.isfinite(g).all()]
            row = self.gradient_totals.setdefault(name, _empty_totals())
            row["steps"] += 1
            row["nonfinite_gradient_tensors"] += len(grads) - len(finite)
            norm = float(torch.sqrt(sum(torch.sum(g.double() ** 2) for g in finite)).cpu()) if finite else None
            if norm is not None:
                row["gradient_norm_sum"] += norm
                row["gradient_norm_max"] = max(row.get("gradient_norm_max", 0.0), norm)
                row["finite_gradient_steps"] += 1
            snapshots[name] = [p.detach().clone() for p in parameters]
        return snapshots

    def after_optimizer_step(self, model, snapshots: dict[str, list[torch.Tensor]]) -> None:
        groups = self._parameter_groups(model)
        for name, parameters in groups.items():
            deltas = [p.detach() - before for p, before in zip(parameters, snapshots[name], strict=True)]
            finite = [delta for delta in deltas if torch.isfinite(delta).all()]
            row = self.gradient_totals[name]
            row["nonfinite_update_tensors"] += len(deltas) - len(finite)
            magnitude = float(torch.sqrt(sum(torch.sum(delta.double() ** 2) for delta in finite)).cpu()) if finite else None
            if magnitude is not None:
                row["update_norm_sum"] += magnitude
                row["update_norm_max"] = max(row.get("update_norm_max", 0.0), magnitude)
                row["finite_update_steps"] += 1

    @staticmethod
    def _parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
        named = list(model.named_parameters())
        return {
            "lstm_layer_1": [p for name, p in named if "lstm1" in name],
            "lstm_layer_2": [p for name, p in named if "lstm2" in name],
            "temporal_attention": [p for name, p in named if "attentions" in name],
            **{f"{scale}_head": list(model.decoders[index].parameters()) for index, scale in enumerate(SCALES)},
        }

    def finish_epoch(self, epoch: int) -> None:
        rows = []
        for (split, scale, lead), values in sorted(self.epoch_totals.items()):
            count = int(values["n_valid"])
            weight_sum = values.get("weight_sum", 0.0)
            target_mean = values["target_sum"] / count if count else None
            median_mean = values["median_sum"] / count if count else None
            target_var = max(values["target_sq_sum"] / count - target_mean**2, 0.0) if count else None
            median_var = max(values["median_sq_sum"] / count - median_mean**2, 0.0) if count else None
            covariance = values["cross_sum"] / count - target_mean * median_mean if count else None
            pearson = (
                covariance / np.sqrt(target_var * median_var)
                if count > 1 and target_var and median_var
                else None
            )
            rows.append({
                "run_id": self.run_id, "profile": self.profile, "seed": self.seed,
                "epoch": epoch, "split": split, "target": scale, "lead": f"T+{lead}",
                "n_valid": count, "n_masked": int(values["n_masked"]),
                "n_nonfinite_target": int(values["n_nonfinite_target"]),
                "n_nonfinite_prediction": int(values["n_nonfinite_prediction"]),
                "target_mean": target_mean, "target_variance": target_var,
                "median_prediction_mean": median_mean, "median_prediction_variance": median_var,
                "prediction_target_variance_ratio": median_var / target_var if count and target_var else None,
                "median_pearson": pearson,
                "median_mae": values["abs_error_sum"] / count if count else None,
                "median_rmse": np.sqrt(values["squared_error_sum"] / count) if count else None,
                "pinball_sum_quantiles_weighted": values["pinball_sum"] / weight_sum if weight_sum else None,
                "pinball_q10_weighted": values["q10_pinball_sum"] / weight_sum if weight_sum else None,
                "pinball_q50_weighted": values["q50_pinball_sum"] / weight_sum if weight_sum else None,
                "pinball_q90_weighted": values["q90_pinball_sum"] / weight_sum if weight_sum else None,
                "q10_q90_coverage": values["covered_count"] / count if count else None,
                "q10_q90_mean_width": values["interval_width_sum"] / count if count else None,
                "quantile_crossing_count": int(values["crossing_count"]),
                "quantile_crossing_rate": values["crossing_count"] / count if count else None,
                "actual_spei_le_minus_1_5": int(values["actual_severe_events"]),
                "predicted_q50_le_minus_1_5": int(values["predicted_severe_events"]),
                "severe_event_recall": values["severe_true_positives"] / values["actual_severe_events"] if values["actual_severe_events"] else None,
                "diagnostic_status": "no_valid_observations" if not count else "ok",
            })
        self._append_csv("epoch_metrics.csv", rows)
        self._append_csv("primary_endpoint.csv", self.endpoint_rows)
        optimization_rows = []
        for component, values in self.gradient_totals.items():
            steps = int(values["steps"])
            optimization_rows.append({
                "run_id": self.run_id, "profile": self.profile, "seed": self.seed,
                "epoch": epoch, "component": component, "optimizer_steps": steps,
                "gradient_norm_mean": values["gradient_norm_sum"] / values["finite_gradient_steps"] if values.get("finite_gradient_steps") else None,
                "gradient_norm_max": values.get("gradient_norm_max"),
                "parameter_update_l2_mean": values["update_norm_sum"] / values["finite_update_steps"] if values.get("finite_update_steps") else None,
                "parameter_update_l2_max": values.get("update_norm_max"),
                "nonfinite_gradient_tensors": int(values["nonfinite_gradient_tensors"]),
                "nonfinite_update_tensors": int(values["nonfinite_update_tensors"]),
            })
        self._append_csv("optimization_diagnostics.csv", optimization_rows)
        self.epoch_totals.clear()
        self.gradient_totals.clear()
        self.endpoint_rows.clear()

    def _append_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.directory / name
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)
