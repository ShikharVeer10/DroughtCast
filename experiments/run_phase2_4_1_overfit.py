"""Training-only fixed-subset overfit diagnostic; never touches official runs."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from data.loader import load_site_table
from data.protocol import MODEL_FEATURES, SPLIT_BOUNDARIES
from models.forecaster import MultiScaleForecaster
from train import pinball_loss

PHASE22_RUN = "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0"
PHASE22_RESULTS = Path("results/maharashtra/phase2/neural") / PHASE22_RUN
NORMALIZER = Path("data/normalizers/maharashtra_protocol10.json")
SCALES = ("spei_3", "spei_6", "spei_12")
QUANTILES = (0.1, 0.5, 0.9)
SEED = 2401
SITES = (
    "akola", "amravati", "aurangabad", "beed", "latur", "mumbai",
    "nagpur", "nanded", "nashik", "pune", "ratnagiri", "solapur",
)
WINDOWS_PER_SITE = 8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_subset(dataset, *, seed: int = SEED, windows_per_site: int = WINDOWS_PER_SITE) -> list[int]:
    """Select time-spread, all-target-valid windows per site, training data only."""
    rng = np.random.default_rng(seed)
    site_ids = np.asarray([str(row["site"]) for row in dataset.metadata])
    selected: list[int] = []
    for site in SITES:
        candidates = np.flatnonzero(site_ids == site)
        candidates = candidates[
            dataset.mask[candidates].all(dim=(1, 2)).cpu().numpy()
            & torch.isfinite(dataset.Y[candidates]).all(dim=(1, 2)).cpu().numpy()
            & torch.isfinite(dataset.X[candidates]).all(dim=(1, 2)).cpu().numpy()
        ]
        if len(candidates) < windows_per_site:
            raise RuntimeError(f"Not enough fully valid training windows for {site}: {len(candidates)}")
        # Select one randomly within each chronological bin to spread dates.
        bins = np.array_split(candidates, windows_per_site)
        selected.extend(int(rng.choice(group)) for group in bins if len(group))
    if len(selected) != len(SITES) * windows_per_site or len(set(selected)) != len(selected):
        raise RuntimeError("Subset selection produced an unexpected or duplicate window count")
    return selected


def _metric_rows(dataset, indexes: list[int], outputs: dict[str, np.ndarray], epoch: int) -> list[dict[str, Any]]:
    y_all = dataset.Y[indexes].cpu().numpy()
    mask_all = dataset.mask[indexes].cpu().numpy().astype(bool)
    sites = np.asarray([dataset.metadata[index]["site"] for index in indexes])
    rows = []
    for scale_index, scale in enumerate(SCALES):
        q_values = outputs[scale]
        for lead in range(4):
            y = y_all[:, scale_index, lead]
            valid = mask_all[:, scale_index, lead] & np.isfinite(y)
            prediction = q_values[:, lead, 1]
            q10, q90 = q_values[:, lead, 0], q_values[:, lead, 2]
            for site in ("ALL", *SITES):
                chosen = valid & ((sites == site) if site != "ALL" else True)
                if not chosen.any():
                    continue
                yt = y[chosen].astype(float)
                yp = prediction[chosen].astype(float)
                residual = yp - yt
                y_var, p_var = float(np.var(yt)), float(np.var(yp))
                corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 and y_var > 0 and p_var > 0 else np.nan
                rows.append({
                    "epoch": epoch, "site": site, "target": scale, "lead": lead + 1,
                    "n": len(yt), "target_mean": float(yt.mean()),
                    "target_std": float(np.std(yt, ddof=1)) if len(yt) > 1 else 0.0,
                    "prediction_mean": float(yp.mean()),
                    "prediction_std": float(np.std(yp, ddof=1)) if len(yp) > 1 else 0.0,
                    "prediction_variance_ratio": p_var / y_var if y_var > 0 else np.nan,
                    "bias": float(residual.mean()), "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "r2": float(1 - np.sum(residual**2) / np.sum((yt - yt.mean())**2)) if y_var > 0 else np.nan,
                    "pearson": corr,
                    "q10_q90_coverage": float(np.mean((yt >= q10[chosen]) & (yt <= q90[chosen]))),
                    "q10_q90_width_mean": float(np.mean(q90[chosen] - q10[chosen])),
                    "quantile_crossing_count": int(np.sum((q_values[chosen, lead, 0] > q_values[chosen, lead, 1]) | (q_values[chosen, lead, 1] > q_values[chosen, lead, 2]))),
                })
    return rows


def _predict(model, x: torch.Tensor) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        prediction, _ = model(x.to(dtype=torch.float32))
    result = {scale: prediction[scale].cpu().numpy() for scale in SCALES}
    for scale, values in result.items():
        if values.shape != (len(x), 4, 3) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid output for {scale}: shape={values.shape}")
    return result


def _group_norm(parameters) -> float:
    squares = [torch.sum(p.grad.detach() ** 2) for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0


def _parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    named = list(model.named_parameters())
    return {
        "lstm_layer_1": [p for name, p in named if "lstm1" in name],
        "lstm_layer_2": [p for name, p in named if "lstm2" in name],
        "temporal_attention": [p for name, p in named if "attentions" in name],
        **{f"{scale}_head": list(model.decoders[index].parameters()) for index, scale in enumerate(SCALES)},
    }


def _render_report(run_id: str, audit: dict, history: pd.DataFrame, metrics: pd.DataFrame) -> str:
    first = history.iloc[0]
    last = history.iloc[-1]
    lines = [
        "# Phase 2.4.1 - Controlled Training-Subset Overfit Diagnostic", "",
        f"Run: `{run_id}`. Status: **{audit['conclusion']}**.", "",
        "## Scope and safeguards", "",
        "This is a diagnostic fit on 96 fully valid windows drawn only from the protocol-10 training split (8 from each of 12 configured sites). It used a fresh model and checkpoint. The loader reads the source CSV, then truncates rows after the training cutoff before quality checks, preprocessing, and window construction; only training windows and labels were materialized for this run. Validation and test values were not used for fitting, selection, or the diagnostic status. No official checkpoint, dataset, normalizer, configuration, or model source was modified.",
        "", "## Effective implementation and settings", "",
        f"Architecture: `{audit['effective_architecture']}`. The model uses 6 input months x 15 features and three scale-specific heads; each emits four leads x q10/q50/q90. The actual variable gate is disabled (`variable_gate: false`); the ignored `use_variable_gate` config key was not enabled.",
        f"Training used AdamW with the saved Phase 2.2 settings: learning rate {audit['training_config']['learning_rate']}, weight decay {audit['training_config']['weight_decay']}, batch size {audit['training_config']['batch_size']}, dropout {audit['training_config']['dropout']}, gradient clip {audit['training_config']['grad_clip']}, and {audit['epochs']} fixed epochs. Loss was the existing three-scale sum of weighted pinball losses. The validation-driven scheduler and early stopping were omitted because validation data are prohibited in this experiment; the training duration was fixed in advance.",
        "", "## Subset composition", "",
        "Selected input/target dates and original training-window indices are in `selected_subset.json`. The subset selection is deterministic, time-spread within each site, and retains only windows with finite inputs and all 12 valid target cells.",
        "", "## Fit diagnostics", "",
        f"Aggregate pinball loss changed from {first.train_pinball_loss:.4f} at epoch 1 to {last.train_pinball_loss:.4f} at epoch {int(last.epoch)}. Median RMSE was {first.median_rmse_all_scales_leads:.4f} at the first epoch and {last.median_rmse_all_scales_leads:.4f} at the last; median prediction/target variance ratio changed from {first.mean_prediction_variance_ratio:.4f} to {last.mean_prediction_variance_ratio:.4f}.",
        "Per-scale, per-lead, per-site metrics and prediction distributions are in `per_scale_lead_metrics.csv`; group gradient norms and parameter update magnitudes are in `gradient_update_diagnostics.csv`.",
        "", "## Objective alignment", "",
        "The training objective sums quantile pinball losses across all three SPEI scales and four leads. Official checkpoint selection instead maximizes validation site-macro MSE skill for SPEI-3 T+3/T+4. That is an objective mismatch: the diagnostic measures whether the training objective can fit its subset, not whether the selected validation endpoint or drought-event recall improves. This experiment did not alter or evaluate that selection rule.",
        "", "## Interpretation", "",
        audit["interpretation"],
        "This experiment cannot establish generalization, calibrated uncertainty, drought-event skill, or performance on validation/test periods. It is not a benchmark result and must not be reported as such.",
        "", "## Reproducibility", "",
        f"Dataset SHA-256: `{audit['dataset_sha256']}`. Normalizer SHA-256: `{audit['normalizer_sha256']}`. Diagnostic seed: {audit['seed']}. Environment/source hashes are in `experiment_metadata.json`; final fresh diagnostic weights are in `diagnostic_checkpoint.pt`.",
        "", "## Verification", "",
        f"Focused tests: {audit['verification']['focused_tests']}. Full pytest: {audit['verification']['full_pytest']}. Ruff: {audit['verification']['ruff']}.",
    ]
    return "\n".join(lines) + "\n"


def run() -> Path:
    metadata = json.loads((PHASE22_RESULTS / "neural_experiment_metadata.json").read_text(encoding="utf-8"))
    config_path = Path(metadata["configuration_path"])
    dataset_path = Path(metadata["dataset_path"])
    if file_sha256(dataset_path) != metadata["dataset_sha256"]:
        raise RuntimeError("Canonical dataset hash differs from the frozen Phase 2.2 run")
    if file_sha256(NORMALIZER) != metadata["normalizer_sha256"]:
        raise RuntimeError("Verified protocol-10 normalizer hash differs from Phase 2.2")
    if list(metadata["feature_order"]) != list(MODEL_FEATURES) or metadata["split_boundaries"] != SPLIT_BOUNDARIES:
        raise RuntimeError("Feature order or split protocol differs from the frozen experiment")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["data"]["features"] != list(MODEL_FEATURES):
        raise RuntimeError("Configured input feature order changed")

    run_script_hash = file_sha256(Path(__file__))
    run_id = f"phase2_4_1_train_only_seed{SEED}_{metadata['dataset_sha256'][:10]}_{metadata['normalizer_sha256'][:10]}_{run_script_hash[:8]}"
    output = Path("results/maharashtra/phase2/diagnostics") / run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    datasets = load_site_table(
        dataset_path,
        features=list(MODEL_FEATURES),
        seq_len=6,
        forecast_steps=4,
        metadata_path=NORMALIZER,
        site_filter=list(config["data"]["site_filter"]),
        site_specs=config.get("sites"),
        split_config=SPLIT_BOUNDARIES,
        include_splits=("train",),
    )
    train = datasets["train"]
    if len(train) != metadata["window_counts"]["train"]:
        raise RuntimeError(f"Unexpected train window count: {len(train)}")
    indexes = select_subset(train)
    x = train.X[indexes].clone()
    y = train.Y[indexes].clone()
    mask = train.mask[indexes].clone()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not mask.all():
        raise RuntimeError("Selected training subset contains non-finite inputs/targets or masked targets")

    config_model = json.loads(json.dumps(metadata["selected_config"]))
    model = MultiScaleForecaster(config_model)
    effective_gate_enabled = model.variable_gate is not None
    profile = metadata["selected_parameters"]
    training_cfg = config_model["training"]
    # Fixed settings from the saved Phase 2.2 selected profile; no scheduler or
    # validation-driven stopping is used in this training-only diagnostic.
    batch_size = int(profile["batch_size"])
    learning_rate = float(profile["learning_rate"])
    weight_decay = float(profile["weight_decay"])
    extreme_weight = float(training_cfg["extreme_weight"])
    extreme_threshold = float(training_cfg["extreme_threshold"])
    grad_clip = float(training_cfg["grad_clip"])
    epochs = int(training_cfg["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(
        TensorDataset(x, y, mask), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )
    groups = _parameter_groups(model)
    history_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    finite_loss_failures = 0
    finite_gradient_failures = 0
    global_step = 0

    initial_outputs = _predict(model, x)
    initial_metrics = _metric_rows(train, indexes, initial_outputs, epoch=0)
    metric_rows.extend(initial_metrics)

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss_sum, seen = 0.0, 0
        gradient_accumulator = {name: [] for name in groups}
        update_accumulator = {name: [] for name in groups}
        batch_losses: dict[str, list[float]] = {scale: [] for scale in SCALES}
        for batch_x, batch_y, batch_mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch_x)
            losses = [
                pinball_loss(
                    prediction[scale], batch_y[:, scale_index], batch_mask[:, scale_index],
                    QUANTILES, extreme_weight, extreme_threshold,
                )
                for scale_index, scale in enumerate(SCALES)
            ]
            loss = torch.stack(losses).sum()
            if not torch.isfinite(loss):
                finite_loss_failures += 1
                raise RuntimeError(f"Non-finite loss in diagnostic at epoch {epoch}")
            for scale, value in zip(SCALES, losses, strict=True):
                batch_losses[scale].append(float(value.detach()))
            loss.backward()
            for name, parameters in groups.items():
                norm = _group_norm(parameters)
                gradient_accumulator[name].append(norm)
                if not np.isfinite(norm):
                    finite_gradient_failures += 1
            global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            if not np.isfinite(global_norm):
                finite_gradient_failures += 1
                raise RuntimeError(f"Non-finite gradient norm in diagnostic at epoch {epoch}")
            before = {name: [p.detach().clone() for p in parameters] for name, parameters in groups.items()}
            optimizer.step()
            for name, parameters in groups.items():
                delta_sq = sum(torch.sum((p.detach() - old) ** 2) for p, old in zip(parameters, before[name], strict=True))
                update_accumulator[name].append(float(torch.sqrt(delta_sq)))
            batch_n = len(batch_x)
            weighted_loss_sum += float(loss.detach()) * batch_n
            seen += batch_n
            global_step += 1
        outputs = _predict(model, x)
        metric_rows.extend(_metric_rows(train, indexes, outputs, epoch=epoch))
        rows_this_epoch = [row for row in metric_rows if row["epoch"] == epoch and row["site"] == "ALL"]
        aggregate_rmse = float(np.mean([row["rmse"] for row in rows_this_epoch]))
        aggregate_prediction_ratio = float(np.nanmean([row["prediction_variance_ratio"] for row in rows_this_epoch]))
        history_rows.append({
            "epoch": epoch,
            "train_pinball_loss": weighted_loss_sum / max(seen, 1),
            "pinball_spei3": float(np.mean(batch_losses["spei_3"])),
            "pinball_spei6": float(np.mean(batch_losses["spei_6"])),
            "pinball_spei12": float(np.mean(batch_losses["spei_12"])),
            "median_rmse_all_scales_leads": aggregate_rmse,
            "mean_prediction_variance_ratio": aggregate_prediction_ratio,
            "prediction_mean_all_scales_leads": float(np.mean([row["prediction_mean"] for row in rows_this_epoch])),
            "prediction_std_all_scales_leads": float(np.mean([row["prediction_std"] for row in rows_this_epoch])),
            "target_mean_all_scales_leads": float(np.mean([row["target_mean"] for row in rows_this_epoch])),
            "target_std_all_scales_leads": float(np.mean([row["target_std"] for row in rows_this_epoch])),
            "min_quantile": float(min(outputs[scale][..., 0].min() for scale in SCALES)),
            "max_quantile": float(max(outputs[scale][..., 2].max() for scale in SCALES)),
            "max_quantile_crossing_count": int(sum(row["quantile_crossing_count"] for row in rows_this_epoch)),
            "optimizer_steps_cumulative": global_step,
        })
        for name in groups:
            update_rows.append({
                "epoch": epoch, "component": name,
                "gradient_norm_mean": float(np.mean(gradient_accumulator[name])),
                "gradient_norm_max": float(np.max(gradient_accumulator[name])),
                "parameter_update_l2_mean_per_batch": float(np.mean(update_accumulator[name])),
                "parameter_update_l2_sum_epoch": float(np.sum(update_accumulator[name])),
                "batches": len(update_accumulator[name]),
            })

    selected_rows = []
    for original_index in indexes:
        metadata_row = train.metadata[original_index]
        selected_rows.append({
            "original_train_window_index": int(original_index),
            "site": metadata_row["site"],
            "input_dates": metadata_row["dates"],
            "target_dates": metadata_row["target_dates"],
            "target_values_by_scale_lead": train.Y[original_index].tolist(),
        })
    output.mkdir(parents=True)
    (output / "selected_subset.json").write_text(json.dumps({
        "seed": SEED, "windows_per_site": WINDOWS_PER_SITE,
        "selection_rule": "seeded random choice from chronological bins within each site's fully valid training windows",
        "split": "train only", "n_windows": len(indexes), "samples": selected_rows,
    }, indent=2), encoding="utf-8")
    history_frame = pd.DataFrame(history_rows)
    metric_frame = pd.DataFrame(metric_rows)
    update_frame = pd.DataFrame(update_rows)
    history_frame.to_csv(output / "training_history.csv", index=False)
    metric_frame.to_csv(output / "per_scale_lead_metrics.csv", index=False)
    update_frame.to_csv(output / "gradient_update_diagnostics.csv", index=False)

    first = history_frame.iloc[0]
    last = history_frame.iloc[-1]
    rmse_improvement = 1 - float(last.median_rmse_all_scales_leads) / float(first.median_rmse_all_scales_leads)
    loss_reduction = 1 - float(last.train_pinball_loss) / float(first.train_pinball_loss)
    ratio_final = float(last.mean_prediction_variance_ratio)
    # Apply a transparent descriptive decision rule; all values derive from
    # training-only data and are not used to tune or choose a checkpoint.
    successful = loss_reduction >= 0.75 and rmse_improvement >= 0.50 and ratio_final >= 0.25
    partial = loss_reduction >= 0.50 and (rmse_improvement >= 0.20 or ratio_final >= 0.10)
    conclusion = "OVERFIT SUCCESSFUL" if successful else "OVERFIT PARTIAL" if partial else "OVERFIT FAILED"
    interpretation = (
        f"Training-only fit reduced aggregate pinball loss by {loss_reduction:.1%} and median RMSE by {rmse_improvement:.1%}; "
        f"the final mean prediction/target variance ratio was {ratio_final:.3f}. "
        + ("These diagnostics indicate the model can fit this fixed subset and produce non-collapsed variation." if successful
           else "The model learned some subset structure but did not meet the predeclared joint fit/variation criteria." if partial
           else "The model did not meet the predeclared criteria for fitting this subset; this points toward an optimization/objective implementation investigation, but is not alone proof of a defect.")
    )
    audit = {
        "conclusion": conclusion,
        "interpretation": interpretation,
        "run_id": run_id,
        "seed": SEED,
        "subset_size": len(indexes),
        "subset_windows_per_site": WINDOWS_PER_SITE,
        "subset_sites": list(SITES),
        "subset_target_fully_valid": True,
        "subset_contains_only_training_windows": True,
        "validation_or_test_materialized": False,
        "source_csv_loaded_then_truncated_at_training_end": True,
        "validation_or_test_values_used_for_fit_selection_or_status": False,
        "dataset_sha256": metadata["dataset_sha256"],
        "normalizer_sha256": metadata["normalizer_sha256"],
        "config_sha256": metadata["configuration_sha256"],
        "phase22_selected_profile": metadata["selected_profile"],
        "effective_architecture": "Shared LSTM(64)->Dropout(0.3)->LSTM(32), shared additive attention, three direct monotone quantile heads",
        "effective_variable_gate_enabled": effective_gate_enabled,
        "training_config": {
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "batch_size": batch_size, "dropout": float(profile["dropout"]),
            "grad_clip": grad_clip, "extreme_weight": extreme_weight,
            "extreme_threshold": extreme_threshold, "loss": "sum of three scale-specific weighted pinball losses",
        },
        "epochs": epochs,
        "validation_scheduler_or_early_stopping_used": False,
        "predeclared_descriptive_rules": {
            "successful": "pinball reduction >=75%, median RMSE reduction >=50%, final mean variance ratio >=0.25",
            "partial": "pinball reduction >=50% and (RMSE reduction >=20% or variance ratio >=0.10)",
            "selection_use": "descriptive status only; no checkpoint/parameter selection",
        },
        "outcomes": {
            "initial_pinball_loss": float(first.train_pinball_loss),
            "final_pinball_loss": float(last.train_pinball_loss),
            "pinball_reduction_fraction": loss_reduction,
            "initial_median_rmse": float(first.median_rmse_all_scales_leads),
            "final_median_rmse": float(last.median_rmse_all_scales_leads),
            "median_rmse_improvement_fraction": rmse_improvement,
            "final_mean_prediction_variance_ratio": ratio_final,
            "finite_loss_failures": finite_loss_failures,
            "nonfinite_gradient_norm_occurrences": finite_gradient_failures,
            "quantile_crossings": int(metric_frame.loc[
                (metric_frame.epoch > 0) & (metric_frame.site == "ALL"),
                "quantile_crossing_count",
            ].sum()),
        },
        "phase24_masked_infinity_defect_separate": "still present; no canonical targets contain infinity and no data/model correction was applied here",
        "phase24_gate_ambiguity_separate": "use_variable_gate key is ignored; model reads variable_gate=false, so the gate is disabled",
        "no_official_artifacts_modified": True,
    }
    torch.save(model.state_dict(), output / "diagnostic_checkpoint.pt")
    code_files = [Path(__file__), Path("models/forecaster.py"), Path("train.py"), Path("data/loader.py")]
    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "device": "cpu", "source_sha256": {str(path): file_sha256(path) for path in code_files},
        "checkpoint_sha256": file_sha256(output / "diagnostic_checkpoint.pt"),
    }
    (output / "experiment_metadata.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (output / "diagnostic_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "diagnostic_report.md").write_text(_render_report(run_id, audit, history_frame, metric_frame), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(run())
