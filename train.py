"""Train on train windows, select on validation loss, never inspect test for selection."""
import random
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.loader import load_site_table, validate_dataset_protocol
from models.forecaster import MultiScaleForecaster
from utils.artifacts import artifact_metadata


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_loss(prediction, target, mask, extreme_weight=1.0, extreme_threshold=-1.5, loss_kind="mse"):
    weights = torch.where(target <= extreme_threshold, extreme_weight, 1.0)
    valid = mask.to(prediction.dtype) * weights
    if loss_kind == "huber":
        point_loss = torch.nn.functional.huber_loss(prediction, target, reduction="none")
    elif loss_kind in ("mse", "squared_error"):
        point_loss = (prediction - target) ** 2
    else:
        raise ValueError(f"Unknown point loss: {loss_kind}")
    return (point_loss * valid).sum() / valid.sum().clamp_min(1.0)


def pinball_loss(prediction, target, mask, quantiles, extreme_weight=1.0, extreme_threshold=-1.5):
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device).view(1, 1, -1)
    error = target.unsqueeze(-1) - prediction
    losses = torch.maximum(q * error, (q - 1.0) * error)
    weights = torch.where(target <= extreme_threshold, extreme_weight, 1.0).unsqueeze(-1)
    valid = mask.unsqueeze(-1).to(prediction.dtype) * weights
    return (losses * valid).sum() / valid.sum().clamp_min(1.0)


def unweighted_mse(prediction, target, mask):
    """Plain masked MSE used for checkpoint selection (ignores task/extreme weights)."""
    valid = mask.to(prediction.dtype)
    return (((prediction - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)


def consistency_loss(predictions):
    """Penalize disagreement between neighboring accumulation-scale forecasts."""
    return sum(torch.mean((predictions[a] - predictions[b]) ** 2)
               for a, b in (("spei_3", "spei_6"), ("spei_6", "spei_12")))


def crps_loss(prediction, target, mask, quantiles):
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device).view(1, 1, -1)
    error = target.unsqueeze(-1) - prediction
    losses = torch.maximum(q * error, (q - 1.0) * error)
    valid = mask.unsqueeze(-1).to(prediction.dtype)
    q_widths = torch.diff(torch.as_tensor(quantiles, dtype=prediction.dtype,
                                           device=prediction.device))
    weights = torch.cat([q_widths[:1] / 2, (q_widths[:-1] + q_widths[1:]) / 2,
                         q_widths[-1:] / 2]).view(1, 1, -1)
    return (losses * valid * weights).sum() / valid.sum().clamp_min(1.0)


def epoch_loss(model, loader, optimizer=None, extreme_weight=3.0, extreme_threshold=-1.5,
               quantiles=None, task_weighting="uncertainty", task_log_vars=None, loss_kind="mse",
               consistency_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total, total_mse, count = 0.0, 0.0, 0
    for batch in loader:
        x, y, mask, _ = batch[:4]
        residual = model.head_mode == "residual"
        last_spei = batch[4] if residual and len(batch) > 4 else None
        static_index = 5 if residual else 4
        static = batch[static_index] if model.static_context and len(batch) > static_index else None
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions, _ = model(x, last_spei=last_spei, static=static)
            task_losses = []
            mse_losses = []
            for i, k in enumerate(("spei_3", "spei_6", "spei_12")):
                if quantiles:
                    base_loss = crps_loss if loss_kind == "crps" else pinball_loss
                    task_losses.append(base_loss(predictions[k], y[:, i], mask[:, i], quantiles)
                                       if loss_kind == "crps" else
                                       base_loss(predictions[k], y[:, i], mask[:, i], quantiles,
                                                 extreme_weight, extreme_threshold))
                    median = predictions[k][..., quantiles.index(0.5)] if 0.5 in quantiles else predictions[k][..., len(quantiles) // 2]
                    mse_losses.append(unweighted_mse(median, y[:, i], mask[:, i]))
                else:
                    task_losses.append(masked_loss(predictions[k], y[:, i], mask[:, i], extreme_weight, extreme_threshold, loss_kind))
                    mse_losses.append(unweighted_mse(predictions[k], y[:, i], mask[:, i]))
            if task_weighting == "uncertainty" and task_log_vars is not None:
                # Kendall et al.: exp(-s_i) * L_i + s_i
                loss = sum(torch.exp(-task_log_vars[i]) * value + task_log_vars[i]
                           for i, value in enumerate(task_losses))
            else:
                loss = sum(task_losses)
            selection_mse = sum(mse_losses)
            if consistency_weight:
                loss = loss + consistency_weight * consistency_loss({
                    k: (value[..., quantiles.index(0.5)] if quantiles and 0.5 in quantiles
                        else value[..., len(quantiles) // 2] if quantiles else value)
                    for k, value in predictions.items()
                })
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * len(x)
        total_mse += selection_mse.item() * len(x)
        count += len(x)
    return total / max(count, 1), total_mse / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--holdout-region", action="append", default=[])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else cfg["experiment"].get("seed", 42)
    seed_everything(seed)
    if not Path(cfg["data"]["path"]).exists():
        raise FileNotFoundError(f"Real dataset not found at {cfg['data']['path']}. Run `python -m data.build_dataset` first.")
    validate_dataset_protocol()
    data = load_site_table(cfg["data"]["path"], cfg["data"].get("features"),
        cfg["data"]["seq_len"], cfg["data"]["forecast_steps"], cfg["data"].get("metadata_path", "data/meta.json"),
        args.holdout_region, cfg["data"].get("site_filter"),
        cfg["model"].get("input_extras"), cfg["model"].get("static_context", False),
        cfg["model"].get("heads", cfg["model"].get("head_mode", "direct")) == "residual",
        cfg.get("sites"))
    train, val = data["train"], data["val"]
    if not len(train) or not len(val):
        raise RuntimeError("Training and validation splits need real monthly observations through 2015")
    cfg["data"]["input_dim"] = train.X.shape[-1]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=cfg["training"]["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=cfg["training"]["batch_size"], shuffle=False)
    model = MultiScaleForecaster(cfg)
    quantiles = cfg["model"].get("quantiles")
    task_weighting = cfg["training"].get("task_weighting", "uncertainty")
    task_log_vars = torch.nn.Parameter(torch.zeros(3)) if task_weighting == "uncertainty" else None

    params = list(model.parameters())
    if task_log_vars is not None:
        params.append(task_log_vars)
    opt = torch.optim.AdamW(params, lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=cfg["training"].get("lr_patience", 5), factor=.5)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg["training"]["checkpoint_dir"]) / "current_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale = float("inf"), 0
    history = []
    ext_weight = cfg["training"].get("extreme_weight", 3.0)
    ext_thresh = cfg["training"].get("extreme_threshold", -1.5)
    loss_kind = cfg["training"].get("loss", "mse")

    for epoch in range(cfg["training"]["epochs"]):
        tr_loss, tr_mse = epoch_loss(model, train_loader, opt, extreme_weight=ext_weight, extreme_threshold=ext_thresh,
                                     quantiles=quantiles, task_weighting=task_weighting, task_log_vars=task_log_vars,
                                     loss_kind=loss_kind, consistency_weight=model.consistency_weight)
        va_loss, va_mse = epoch_loss(model, val_loader, extreme_weight=ext_weight, extreme_threshold=ext_thresh,
                                     quantiles=quantiles, task_weighting=task_weighting, task_log_vars=task_log_vars,
                                     loss_kind=loss_kind, consistency_weight=model.consistency_weight)
        history.append({"epoch": epoch + 1, "train_loss_weighted": tr_loss,
                        "train_mse_unweighted": tr_mse, "validation_loss_weighted": va_loss,
                        "validation_mse_unweighted": va_mse})
        # Checkpoint on unweighted validation MSE; learned uncertainty weights change over epochs.
        scheduler.step(va_mse)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch={epoch+1:03d} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
                  f"train_mse={tr_mse:.6f} val_mse={va_mse:.6f}")
        if va_mse < best:
            best, stale = va_mse, 0
            state = {"model": model.state_dict(),
                     "task_log_vars": task_log_vars.detach().cpu() if task_log_vars is not None else None,
                     "config": cfg, "seed": seed, "best_val_mse": best, "best_val_loss": va_loss,
                     "artifact_metadata": artifact_metadata(cfg, cfg["data"]["path"], seed)}
            torch.save(state, checkpoint)
        else:
            stale += 1
        if stale >= cfg["training"].get("early_stopping_patience", 12):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    run_dir = checkpoint.parent
    (run_dir / f"history_seed_{seed}.json").write_text(json.dumps({
        "seed": seed, "config": str(args.config), "selected_epoch": int(min(history, key=lambda row: row["validation_mse_unweighted"])["epoch"]),
        "epochs": history,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
