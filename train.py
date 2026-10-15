"""Train on train windows, select on validation loss, never inspect test for selection."""
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.loader import load_site_table
from models.forecaster import MultiScaleForecaster


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_loss(prediction, target, mask, extreme_weight=1.0, loss_kind="mse"):
    weights = torch.where(target <= -1.0, extreme_weight, 1.0)
    valid = mask.to(prediction.dtype) * weights
    if loss_kind == "huber":
        point_loss = torch.nn.functional.huber_loss(prediction, target, reduction="none")
    elif loss_kind in ("mse", "squared_error"):
        point_loss = (prediction - target) ** 2
    else:
        raise ValueError(f"Unknown point loss: {loss_kind}")
    return (point_loss * valid).sum() / valid.sum().clamp_min(1)


def pinball_loss(prediction, target, mask, quantiles, extreme_weight=1.0):
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device).view(1, 1, -1)
    error = target.unsqueeze(-1) - prediction
    losses = torch.maximum(q * error, (q - 1.0) * error)
    weights = torch.where(target <= -1.0, extreme_weight, 1.0).unsqueeze(-1)
    valid = mask.unsqueeze(-1).to(prediction.dtype) * weights
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def epoch_loss(model, loader, optimizer=None, extreme_weight=1.0, quantiles=None,
               task_weighting="fixed", task_log_vars=None, loss_kind="mse"):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for x, y, mask, _ in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions, _ = model(x)
            task_losses = []
            for i, k in enumerate(("spei_3", "spei_6", "spei_12")):
                if quantiles:
                    task_losses.append(pinball_loss(predictions[k], y[:, i], mask[:, i], quantiles, extreme_weight))
                else:
                    task_losses.append(masked_loss(predictions[k], y[:, i], mask[:, i], extreme_weight, loss_kind))
            if task_weighting == "uncertainty":
                loss = sum(torch.exp(-task_log_vars[i]) * value + task_log_vars[i]
                           for i, value in enumerate(task_losses))
            else:
                loss = sum(task_losses)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / max(count, 1)


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
        raise FileNotFoundError(f"Real dataset not found at {cfg['data']['path']}. Run `python -m data.build_dataset` first; synthetic fallback is disabled.")
    data = load_site_table(cfg["data"]["path"], cfg["data"].get("features"),
        cfg["data"]["seq_len"], cfg["data"]["forecast_steps"], cfg["data"].get("metadata_path", "data/meta.json"),
        args.holdout_region, cfg["data"].get("site_filter"))
    train, val = data["train"], data["val"]
    if not len(train) or not len(val):
        raise RuntimeError("Training and validation splits need real monthly observations through 2015")
    cfg["data"]["input_dim"] = train.X.shape[-1]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=cfg["training"]["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=cfg["training"]["batch_size"], shuffle=False)
    model = MultiScaleForecaster(cfg)
    quantiles = cfg["model"].get("quantiles")
    task_weighting = cfg["training"].get("task_weighting", "fixed")
    task_log_vars = torch.nn.Parameter(torch.zeros(3)) if task_weighting == "uncertainty" else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 1e-4))
    if task_log_vars is not None:
        opt.add_param_group({"params": [task_log_vars]})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=cfg["training"].get("lr_patience", 5), factor=.5)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg["training"]["checkpoint_dir"]) / "best_research_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale = float("inf"), 0
    for epoch in range(cfg["training"]["epochs"]):
        tr = epoch_loss(model, train_loader, opt, cfg["training"].get("extreme_weight", 1.0),
                        quantiles, task_weighting, task_log_vars, cfg["training"].get("loss", "mse"))
        va = epoch_loss(model, val_loader, extreme_weight=cfg["training"].get("extreme_weight", 1.0),
                        quantiles=quantiles, task_weighting=task_weighting, task_log_vars=task_log_vars,
                        loss_kind=cfg["training"].get("loss", "mse"))
        scheduler.step(va)
        print(f"epoch={epoch+1} train_loss={tr:.6f} val_loss={va:.6f}")
        if va < best:
            best, stale = va, 0
            state = {"model": model.state_dict(),
                     "task_log_vars": task_log_vars.detach().cpu() if task_log_vars is not None else None,
                     "config": cfg, "seed": seed, "best_val_loss": best}
            torch.save(state, checkpoint)
        else:
            stale += 1
        if stale >= cfg["training"].get("early_stopping_patience", 12):
            break


if __name__ == "__main__":
    main()
