"""Inference interface for DroughtCast.

Loads a trained model checkpoint and config, accepts an input sequence of
historical meteorological and teleconnection observations, and produces
multi-scale SPEI forecasts (SPEI-3, SPEI-6, SPEI-12) over lead times T+1..T+4.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from models.forecaster import MultiScaleForecaster
from utils.artifacts import architecture_fingerprint, validate_checkpoint_metadata


class DroughtCastPredictor:
    """Production and research inference wrapper for DroughtCast."""

    def __init__(self, checkpoint_path: Union[str, Path], config_path: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.device = torch.device(device)
        chk = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_cfg = None
        if isinstance(chk, dict) and "config" in chk and chk["config"]:
            checkpoint_cfg = chk["config"]
        elif config_path is not None:
            checkpoint_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        else:
            raise ValueError("Configuration must be provided or present in the checkpoint.")

        if isinstance(chk, dict):
            validate_checkpoint_metadata(chk, checkpoint_cfg)
        if config_path is not None and isinstance(chk, dict) and chk.get("config"):
            supplied_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            supplied_cfg.setdefault("data", {})
            if not supplied_cfg["data"].get("input_dim") and supplied_cfg["data"].get("features"):
                supplied_cfg["data"]["input_dim"] = len(supplied_cfg["data"]["features"])
            checkpoint_arch = chk["artifact_metadata"]["architecture_fingerprint"]
            if architecture_fingerprint(supplied_cfg) != checkpoint_arch:
                raise ValueError(
                    "The supplied config does not match the checkpoint architecture. "
                    "Use the config saved in the checkpoint or retrain."
                )
        self.cfg = checkpoint_cfg
        self.model = MultiScaleForecaster(self.cfg)
        state_dict = chk.get("model", chk)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Produce multi-scale forecasts for input tensor.

        Parameters
        ----------
        x : array-like of shape [Batch, Seq_len=6, Features=12]

        Returns
        -------
        predictions : dict
            Keys 'spei_3', 'spei_6', 'spei_12', each ndarray of shape [Batch, 4].
        aux : dict
            Auxiliary outputs including attention weights of shape [Batch, ...].
        """
        if isinstance(x, np.ndarray):
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_t = x.to(device=self.device, dtype=torch.float32)

        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)

        with torch.no_grad():
            preds_t, aux_t = self.model(x_t)

        preds = {k: v.cpu().numpy() for k, v in preds_t.items()}
        def to_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy()
            if isinstance(value, list):
                return [to_numpy(item) for item in value]
            if isinstance(value, tuple):
                return tuple(to_numpy(item) for item in value)
            return value

        aux = {k: to_numpy(v) for k, v in aux_t.items()}
        return preds, aux


def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained DroughtCast checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/current_model.pt", help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu, cuda)")
    args = parser.parse_args()

    chk_path = Path(args.checkpoint)
    if not chk_path.exists():
        print(f"Checkpoint not found at {chk_path}. Please train a model first using `python train.py`.")
        return

    predictor = DroughtCastPredictor(chk_path, config_path=args.config, device=args.device)
    print(f"Loaded DroughtCast checkpoint successfully from {chk_path}")

    # Generate synthetic inference sample [1, 6, 12] for functional verification
    sample_input = np.random.randn(1, 6, predictor.cfg["data"].get("input_dim", 12)).astype(np.float32)
    preds, aux = predictor.predict(sample_input)
    print("Functional test output:")
    for scale, pred in preds.items():
        print(f"  {scale}: {np.round(pred[0], 3)}")
    if aux.get("attention") is not None:
        print(f"  Attention tensor shape: {aux['attention'].shape}")


if __name__ == "__main__":
    main()
