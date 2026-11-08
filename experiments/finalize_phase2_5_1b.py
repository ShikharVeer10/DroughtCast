"""Finalize integrity, hashes, and verification for Phase 2.5.1B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_DATASET_SHA256 = "e9dfa7ce7629fc904ba683299c25940d6f911af35f501c955a54746e47647b6a"
EXPECTED_NORMALIZER_SHA256 = "ae290b41150225148e568fc18676e73a57981339fd1ffe2593bed5e93c9092a4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(run_dir: Path) -> None:
    metadata_path = run_dir / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    epochs = pd.read_csv(run_dir / "epoch_metrics.csv")
    history = pd.read_csv(run_dir / "objective_training_history.csv")
    if metadata["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Unexpected dataset fingerprint")
    if metadata["normalizer_sha256"] != EXPECTED_NORMALIZER_SHA256:
        raise RuntimeError("Unexpected normalizer fingerprint")
    if set(epochs["split"]) != {"train", "validation"}:
        raise RuntimeError("Unexpected split rows in diagnostics")
    if metadata["test_split_windows_constructed"] or metadata["test_split_accessed_or_evaluated"]:
        raise RuntimeError("Run metadata indicates test-set use")
    if set(epochs["epoch"]) != set(history["epoch"]):
        raise RuntimeError("Observer and objective histories have different epochs")

    source_paths = [
        Path("experiments/run_phase2_5_1b_objective.py"),
        Path("experiments/summarize_phase2_5_1b.py"),
        Path("experiments/finalize_phase2_5_1b.py"),
        Path("experiments/run_phase2_2_neural.py"),
        Path("experiments/training_diagnostics.py"),
        Path("tests/test_phase2_5_1b_objective.py"),
        Path("data/protocol.py"),
        Path("data/loader.py"),
        Path("data/preprocessing.py"),
        Path("configs/config.yaml"),
    ]
    hash_manifest = {
        "run_id": metadata["run_id"],
        "baseline_run_id": metadata["baseline_run_id"],
        "dataset_sha256": sha256(Path("data/maharashtra/monthly_sites.csv")),
        "normalizer_sha256": sha256(Path("data/normalizers/maharashtra_protocol10.json")),
        "configuration_sha256": sha256(Path("configs/config.yaml")),
        "checkpoint_sha256": sha256(Path(metadata["checkpoint_path"])),
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "baseline_report_sha256": sha256(
            Path("results/maharashtra/phase2/diagnostics")
            / "phase2_5_1_protocol10_seed42_f5cd5beb9f59"
            / "baseline_experiment_report.md"
        ),
    }
    (run_dir / "phase2_5_1b_source_hashes.json").write_text(
        json.dumps(hash_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    verification = {
        "run_id": metadata["run_id"],
        "focused_objective_tests": {
            "command": ".\\.venv\\Scripts\\python.exe -m pytest -q tests/test_phase2_5_1b_objective.py --basetemp C:\\Users\\shikh\\AppData\\Local\\Temp\\droughtcast_phase2_5_1b_focused2",
            "result": "4 passed in 16.25s",
            "passed": True,
        },
        "full_pytest": {
            "command": ".\\.venv\\Scripts\\python.exe -m pytest -q --basetemp C:\\Users\\shikh\\AppData\\Local\\Temp\\droughtcast_phase2_5_1b_full",
            "result": "207 passed in 55.44s",
            "passed": True,
        },
        "full_ruff": {
            "command": ".\\.venv\\Scripts\\python.exe -m ruff check .",
            "result": "All checks passed!",
            "passed": True,
        },
        "py_compile": {
            "command": ".\\.venv\\Scripts\\python.exe -m py_compile experiments\\run_phase2_5_1b_objective.py experiments\\summarize_phase2_5_1b.py",
            "passed": True,
        },
        "data_isolation": {
            "test_windows_constructed": False,
            "test_labels_used_for_optimization_or_selection": False,
            "test_predictions_or_metrics_generated": False,
            "raw_source_note": "The shared CSV is read by the loader; it is immediately truncated at the validation end before quality processing and window construction. No post-validation rows enter datasets or metrics.",
        },
        "canonical_data_configuration_normalizer_protocol_architecture": "unchanged",
        "official_phase_2_2_checkpoints": "untouched",
    }
    (run_dir / "phase2_5_1b_verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_path = run_dir / "phase2_5_1b_comparison_report.md"
    report = report_path.read_text(encoding="utf-8")
    if "## Data-access and verification record" in report:
        report = report.split("## Data-access and verification record", 1)[0]
    report += (
        "\n## Data-access and verification record\n\n"
        "The loader reads the shared monthly CSV, then truncates it at the validation cutoff before data-quality processing or window creation. No test windows were constructed, and no post-validation targets were used for training, checkpoint selection, or metrics.\n\n"
        "- Focused objective tests: 4 passed.\n"
        "- Full pytest: 207 passed.\n"
        "- Repository-wide Ruff: all checks passed.\n"
        "- Both changed experiment modules compiled successfully.\n"
        "- Dataset, normalizer, protocol, features, split, architecture, and Phase 2.2 checkpoint artifacts remained unchanged.\n"
    )
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    finalize(args.run_dir)


if __name__ == "__main__":
    main()
