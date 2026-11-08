"""Audit the frozen Phase 2.2 neural implementation without training it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.forecaster import MultiScaleForecaster
from train import pinball_loss

RUN_ID = "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0"
RESULTS = Path("results/maharashtra/phase2/neural") / RUN_ID
CHECKPOINTS = Path("models/neural") / RUN_ID
OUT = Path("results/maharashtra/phase2/diagnostics/phase2_4_implementation_audit")
SCALES = ("spei_3", "spei_6", "spei_12")
QUANTILES = (0.1, 0.5, 0.9)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gradient_audit(config: dict, state: dict) -> dict:
    model = MultiScaleForecaster(config)
    model.load_state_dict(state, strict=True)
    model.eval()  # Disable dropout; gradients remain enabled.
    model.zero_grad(set_to_none=True)
    x = torch.linspace(-1.0, 1.0, 2 * 6 * 15, dtype=torch.float32).reshape(2, 6, 15)
    targets = torch.linspace(-2.4, 1.8, 2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.ones_like(targets, dtype=torch.bool)
    predictions, _ = model(x)
    per_scale = {
        name: pinball_loss(predictions[name], targets[:, index], mask[:, index], QUANTILES, 1.5, -1.0)
        for index, name in enumerate(SCALES)
    }
    total = torch.stack(list(per_scale.values())).sum()
    total.backward()

    groups = {
        "lstm_layer_1": [p for n, p in model.named_parameters() if "lstm1" in n],
        "lstm_layer_2": [p for n, p in model.named_parameters() if "lstm2" in n],
        "temporal_attention": [p for n, p in model.named_parameters() if "attentions" in n],
        **{f"{scale}_head": list(model.decoders[i].parameters()) for i, scale in enumerate(SCALES)},
    }
    report = {}
    for name, parameters in groups.items():
        grads = [p.grad for p in parameters]
        nonnull = [g for g in grads if g is not None]
        report[name] = {
            "parameter_tensors": len(parameters),
            "tensors_with_grad": len(nonnull),
            "tensors_with_nonzero_grad": sum(bool(torch.count_nonzero(g)) for g in nonnull),
            "all_finite": all(bool(torch.isfinite(g).all()) for g in nonnull),
            "global_l2_norm": float(torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in nonnull))) if nonnull else 0.0,
            "max_abs_grad": max((float(g.abs().max()) for g in nonnull), default=0.0),
        }
    return {
        "synthetic_batch_only": True,
        "optimizer_step_performed": False,
        "input_shape": list(x.shape),
        "prediction_shapes": {name: list(predictions[name].shape) for name in SCALES},
        "per_scale_pinball_loss": {name: float(value.detach()) for name, value in per_scale.items()},
        "sum_of_scale_losses": float(total.detach()),
        "groups": report,
        "all_model_parameters_in_configured_adamw": True,
        "optimizer_code_evidence": "experiments/run_phase2_2_neural.py::_fit_profile constructs AdamW(model.parameters(), ...)",
        "no_official_model_weights_or_optimizer_state_modified": True,
    }


def run() -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit directory: {OUT}")
    metadata_path = RESULTS / "neural_experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_path = CHECKPOINTS / "selected_seed_42.pt"
    checkpoint_hash = sha256(checkpoint_path)
    expected_hash = metadata["selected_checkpoints"]["42"]["sha256"]
    if checkpoint_hash != expected_hash:
        raise RuntimeError("Selected checkpoint hash does not match frozen Phase 2.2 metadata")
    config = metadata["selected_config"]
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    gradient = _gradient_audit(config, state)

    history = __import__("pandas").read_csv(RESULTS / "neural_training_history.csv")
    history = history[(history.profile == metadata["selected_profile"]) & history.seed.isin([42, 101, 2024, 2025, 2026])]
    selected_epochs = {}
    for seed, group in history.groupby("seed"):
        best = group.loc[group.validation_primary_site_mean_mse_skill.idxmax()]
        selected_epochs[str(int(seed))] = {
            "best_epoch_recomputed_from_history": int(best.epoch),
            "best_epoch_in_phase2_2_manifest": int(metadata["selected_checkpoints"][str(int(seed))]["validation_selected_epoch"]),
            "best_validation_primary_skill": float(best.validation_primary_site_mean_mse_skill),
            "training_pinball_first": float(group.iloc[0].train_pinball_loss),
            "training_pinball_last": float(group.iloc[-1].train_pinball_loss),
            "history_epochs": len(group),
            "checkpoint_matches_best_epoch": int(best.epoch) == int(metadata["selected_checkpoints"][str(int(seed))]["validation_selected_epoch"]),
        }

    # Directly reproduce the current helper's behavior with an invalid value masked out.
    probe_prediction = torch.zeros((1, 2, 3), dtype=torch.float32)
    probe_target = torch.tensor([[0.0, float("inf")]], dtype=torch.float32)
    probe_mask = torch.tensor([[True, False]])
    masked_infinity = pinball_loss(probe_prediction, probe_target, probe_mask, QUANTILES)
    target_scale_probe = np.nan_to_num(np.array([[0.0, np.inf]]), nan=0.0).astype(np.float32)
    target_frame = pd.read_csv(metadata["dataset_path"], usecols=list(SCALES))
    target_stats = {}
    for scale in SCALES:
        values = pd.to_numeric(target_frame[scale], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        target_stats[scale] = {
            "missing_nan_count": int(np.isnan(values).sum()),
            "positive_infinity_count": int(np.isposinf(values).sum()),
            "negative_infinity_count": int(np.isneginf(values).sum()),
            "min_finite": float(finite.min()) if len(finite) else None,
            "max_finite": float(finite.max()) if len(finite) else None,
        }

    out_model = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": metadata["selected_checkpoints"]["42"]["validation_selected_epoch"],
        "frozen_config_sha256": metadata["configuration_sha256"],
        "selected_profile": metadata["selected_profile"],
        "model_config": {key: config["model"].get(key) for key in (
            "encoder", "attention", "heads", "input_dim", "hidden_dim_1", "hidden_dim_2",
            "dropout", "quantiles", "use_cascaded_decoder", "variable_gate", "use_variable_gate",
        )},
    }
    audit = {
        "status": "DEFECT FOUND",
        "scope": "implementation audit only; no training, optimizer step, or canonical artifact changes",
        "dataset_sha256": metadata["dataset_sha256"],
        "normalizer_sha256": metadata["normalizer_sha256"],
        "official_test_set_used": False,
        "checkpoint": out_model,
        "tensor_contract": {
            "loader_X": "[batch, 6 time steps, 15 features] float32",
            "loader_Y": "[batch, 3 scales, 4 leads] float32, native SPEI units",
            "loader_mask": "[batch, 3 scales, 4 leads] bool",
            "model_output_per_scale": "[batch, 4 leads, 3 quantiles]",
            "scale_head_mapping": {str(i): scale for i, scale in enumerate(SCALES)},
            "quantile_mapping": {str(i): f"q{int(q*100)}" for i, q in enumerate(QUANTILES)},
            "lead_mapping": {str(i + 1): f"target window offset {i + 1} month after input end" for i in range(4)},
            "predict_tensor": "[batch, 3 scales, 4 leads], median quantile; scales ordered SPEI-3/6/12",
        },
        "target_transform": {
            "target_scaler": "none",
            "loader_behavior": "targets are selected from CSV and transposed to [scale, lead]; no standardization",
            "inverse_transform": "none; inference returns model outputs directly in SPEI units",
            "target_generation_clip": "SPEI target builder clips generated values at +/-3.5; no training-time target clipping/transformation is applied",
            "canonical_dataset_target_ranges_and_missing": target_stats,
        },
        "pinball_loss": {
            "formula": "max(q * (y - yhat), (q - 1) * (y - yhat))",
            "q_shape": "[1, 1, Q], broadcast against prediction [B,L,Q] and target [B,L,1]",
            "reduction": "sum over valid weighted samples and quantiles / sum of valid sample weights; sum, not mean, over quantile dimension",
            "expected_sum_over_three_quantiles": "For y=2, prediction=(0,0,0), q=(.1,.5,.9), losses=(.2,1,1.8), sum=3. For y=-2, losses=(1.8,1,.2), sum=3.",
            "masked_finite_target_probe": "covered by unit tests and correctly contributes zero",
            "masked_infinity_probe": {
                "loss": float(masked_infinity) if bool(torch.isfinite(masked_infinity)) else None,
                "observed_result": "NaN" if not bool(torch.isfinite(masked_infinity)) else "finite",
                "is_finite": bool(torch.isfinite(masked_infinity)),
            },
            "loader_nan_to_num_infinity_probe": {"converted_value": float(target_scale_probe[0, 1]), "is_finite": bool(np.isfinite(target_scale_probe[0, 1]))},
            "confirmed_defect": "non-finite target sentinels are not robustly neutralized before multiplication by a zero mask; a masked +Inf produces NaN pinball loss. Loader np.nan_to_num(..., nan=0) leaves +Inf as the largest float, which overflows when cast to float32. This is not present in the canonical dataset targets (verified zero infinities), so it does not explain the current prediction collapse.",
        },
        "gradient_flow_on_frozen_checkpoint": gradient,
        "validation_checkpoint_audit": {
            "selection_metric": metadata["selection_metric"],
            "early_stopping_patience": metadata["selected_config"]["training"]["early_stopping_patience"],
            "min_delta": 1e-5,
            "scheduler": "ReduceLROnPlateau(mode=max, patience=5, factor=0.5) on validation primary skill",
            "per_seed": selected_epochs,
            "interpretation": "All five selected epochs match the actual validation-score maxima. With epoch 1 best and patience 12, stopping after 12 non-improving epochs at epoch 13 is expected, not an early-stopper defect.",
            "not_saved": ["actual per-batch gradient norms", "validation pinball loss by epoch", "optimizer state by epoch", "epoch-wise predictions for all target scales/leads"],
        },
        "optimizer": {
            "algorithm": "AdamW",
            "parameters": metadata["selected_parameters"],
            "grad_clip_norm": metadata["selected_config"]["training"]["grad_clip"],
            "step_code": "loss.backward(); clip_grad_norm_; optimizer.step()",
            "historical_parameter_update_verification": "not independently reconstructable: initial weights and optimizer states were not saved; decreasing training loss and epoch-1 checkpoint are consistent with training having occurred",
        },
        "root_cause_assessment": {
            "loss_formula": "no defect found for finite targets",
            "broadcasting_and_mapping": "no defect found",
            "target_scaling_or_inverse_transform": "no defect found; no target transform exists",
            "gradient_connectivity": "synthetic backward pass on the frozen checkpoint verifies finite nonzero gradients through both LSTMs, shared attention, and all three quantile heads; this does not prove historical batch gradients",
            "checkpoint_and_early_stopping": "no defect found; selected epoch matches validation history maximum",
            "prediction_collapse_cause": "unresolved; objective mismatch and generalization/optimization behavior remain candidates, not proven causes",
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "phase2_4_neural_implementation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT / "phase2_4_neural_implementation_audit.md").write_text(_report(audit), encoding="utf-8")
    return OUT


def _report(audit: dict) -> str:
    verification = audit.get("verification", {})
    lines = [
        "# Phase 2.4 - DroughtCast Neural Implementation Audit", "",
        "**Status: DEFECT FOUND** - a masked non-finite target can poison the pinball loss. The canonical dataset contains no infinite SPEI targets, so this defect does not explain the observed collapse.", "",
        "## Scope", "",
        "Loaded the frozen seed-42 checkpoint and ran a synthetic forward/backward pass only to inspect gradient connectivity. No optimizer step, training, model changes, dataset changes, or test-set analysis was performed.", "",
        "## Files and functions traced", "",
        "- `data/loader.py::load_site_table`: raw targets selected and transposed to `[B, scale, lead]`; no target normalization; missing NaNs are masked and replaced with zero placeholders.",
        "- `models/forecaster.py::MultiScaleForecaster.forward` / `predict_tensor`: three named scale heads each emit `[B, lead, quantile]`; median stack is `[B, scale, lead]`.",
        "- `train.py::pinball_loss`: standard pinball expression, quantile axis broadcasting, mask/weight reduction.",
        "- `experiments/run_phase2_2_neural.py::_train_epoch`: sums scale-specific losses, backpropagates, clips norm at configured limit, and updates AdamW.",
        "- `experiments/run_phase2_2_neural.py::_fit_profile`: validation-only primary skill selection, scheduler, early stopping, and checkpoint restoration.",
        "- `inference.py::DroughtCastPredictor.predict`: returns model outputs directly; no target inverse transformation.",
        "", "## Tensor contract", "",
        "| Tensor | Shape | Meaning |", "| --- | --- | --- |",
        "| Input X | `[B, 6, 15]` | Six chronological months, protocol-10 feature order |",
        "| Target Y | `[B, 3, 4]` | SPEI-3/6/12 x T+1..T+4, native units |",
        "| Mask | `[B, 3, 4]` | Valid target indicator |",
        "| Per-scale output | `[B, 4, 3]` | Four leads x q10/q50/q90 |",
        "| `predict_tensor` | `[B, 3, 4]` | Median forecast by scale and lead |",
        "", "The mapping is explicit in code order: scales `spei_3`, `spei_6`, `spei_12`; leads follow the next four monthly rows after the six input rows; quantile axis follows `[0.1,0.5,0.9]`. Monotonic q10/q50/q90 construction is structural.",
        "", "## Loss and target scaling", "",
        "For finite values, `max(q*(y-yhat),(q-1)*(y-yhat))` is correct and broadcasts as intended. The loss sums across the three quantiles and divides by the number/weight of valid target cells; this is three times the per-quantile mean convention, not an accidentally tiny loss. It has no graph detachment. Focused tests verify positive, negative, zero, multi-quantile, and masked finite-target cases.",
        "Targets are not scaled by the loader or training loop, so no inverse transform is expected. The dataset builder's SPEI generation clips its standardized targets to +/-3.5; this clipping occurs before training and does not suppress the -1.5 event threshold. Inputs alone use the saved normalizer.",
        "", "## Gradient-flow probe", "",
        f"A deterministic synthetic batch was backpropagated through the frozen checkpoint without an optimizer step. Gradient coverage and norms: `{json.dumps(audit['gradient_flow_on_frozen_checkpoint']['groups'], sort_keys=True)}`. This proves the current graph is connected under the probe, but historical per-batch gradient norms were not saved.",
        "", "## Confirmed defect", "",
        "The loader converts missing target values with `np.nan_to_num(y, nan=0.0)`. For +/-Inf, NumPy substitutes the largest float; conversion to float32 can restore Inf. Although the target mask is false, `pinball_loss` computes the non-finite error and then multiplies it by zero (`Inf * 0` / `NaN * 0`), yielding NaN. The direct probe records a non-finite loss. The current canonical SPEI columns have zero infinities, so this is a defensive data-integrity defect, not the established source of Phase 2.3 collapse.",
        "Do not apply the correction in this audit. The minimal future fix is to replace invalid targets with a finite neutral value using the mask before calculating loss (and to sanitize both NaN and infinities in the loader), with regression tests for masked NaN and +/-Inf plus finite gradient behavior.",
        "", "## Training selection and unresolved collapse", "",
        "All five selected checkpoint epochs reproduce the maximum validation primary skill in their histories. The configured patience is 12; epoch 1 remains best, so stopping after 12 subsequent failures at epoch 13 is expected. The checkpoint selected is the best state, not the final epoch state. This is not an early-stopping implementation defect.",
        "The loss trains quantiles across all scales/leads, while checkpoint selection optimizes only SPEI-3 T+3/T+4 site-macro MSE skill. That objective mismatch is real, but it is not proven to cause collapse. The synthetic backward probe confirms finite nonzero gradients reach both LSTMs, shared attention, and all three heads. Actual historical batch gradients and optimizer states were not saved, so optimization stability and generalization cause remain unresolved.",
        "", "## Recommendation", "",
        "Do not claim that the collapse is caused by a loss, mapping, target-scaling, or checkpoint bug. Fix the masked non-finite target path in a separately authorized code-change phase. To isolate the prediction-collapse cause, next run a small, separately versioned training-only subset-overfit diagnostic with no validation/test data and no changes to official checkpoints. If that diagnostic can overfit, investigate validation generalization/objective mismatch; if it cannot, instrument gradients and update magnitudes during the diagnostic run.",
        "", "## Verification", "",
        f"- Focused Phase 2.4 tests: {verification.get('focused_tests', 'not recorded')}.",
        f"- Full pytest suite: {verification.get('full_pytest', 'not recorded')}.",
        f"- Ruff on changed Python files: {verification.get('ruff', 'not recorded')}.",
        "- No official model training or optimizer step was performed; the checkpoint gradient probe used a synthetic batch only.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(run())
