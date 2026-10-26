"""Plot test reliability diagrams from saved calibrated prediction CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/probability_calibration")
    parser.add_argument("--scale", default="spei_6")
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--output-dir", default="results/probability_calibration")
    args = parser.parse_args()
    frames = []
    for path in sorted(Path(args.input_dir).glob("predictions_seed_*.csv")):
        frame = pd.read_csv(path)
        frames.append(frame[(frame["scale"] == args.scale) & (frame["lead"] == args.lead)])
    if not frames:
        raise FileNotFoundError("No saved calibrated prediction CSVs found")
    frame = pd.concat(frames, ignore_index=True)
    bins = pd.cut(frame["calibrated_probability"], bins=10, labels=False,
                   include_lowest=True)
    grouped = frame.assign(bin=bins).groupby("bin", observed=True).agg(
        predicted=("calibrated_probability", "mean"),
        observed=("observed_event", "mean"),
        n=("observed_event", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(grouped["predicted"], grouped["observed"], "o-",
            color="#0072B2", label=f"{args.scale.upper()} T+{args.lead}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed event frequency",
           title="Held-out drought-event reliability")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(out / f"reliability_{args.scale}_T{args.lead}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
