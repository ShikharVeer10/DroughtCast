"""Export model attention and Integrated Gradients for test examples."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.loader import load_site_table
from models.forecaster import MultiScaleForecaster
from utils.interpretability import batch_interpretability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_research_model.pt")
    parser.add_argument("--output", default="results/interpretability.json")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lead", type=int, default=0, help="zero-based forecast lead")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    datasets = load_site_table(cfg["data"]["path"], cfg["data"].get("features"),
        cfg["data"]["seq_len"], cfg["data"]["forecast_steps"], cfg["data"].get("metadata_path"))
    test = datasets["test"]
    count = min(len(test), args.samples)
    cfg["data"]["input_dim"] = test.X.shape[-1]
    model = MultiScaleForecaster(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    # Select a deterministic spread of test windows instead of only adjacent months.
    indices = np.linspace(0, len(test) - 1, count, dtype=int)
    selected = test.X[indices]
    metadata = [test.metadata[i] for i in indices]
    result = batch_interpretability(model, selected, cfg["data"].get("features", []), metadata,
                                   lead=args.lead, steps=args.steps)
    serializable = {"feature_names": result["feature_names"], "metadata": result["metadata"],
                    "attention": result["attention"].tolist(), "warning": result["warning"],
                    "integrated_gradients": {k: v.tolist() for k, v in result["integrated_gradients"].items()},
                    "aggregation": "absolute attribution averaged over sample and input time"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable), encoding="utf-8")
    print(f"Saved {count} test-window attribution records to {output}")


if __name__ == "__main__":
    main()
