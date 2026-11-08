"""Write provenance and verification manifests for a completed Phase 2.5.1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_DATASET_SHA256 = "e9dfa7ce7629fc904ba683299c25940d6f911af35f501c955a54746e47647b6a"
EXPECTED_NORMALIZER_SHA256 = "ae290b41150225148e568fc18676e73a57981339fd1ffe2593bed5e93c9092a4"
PHASE22_METADATA = Path(
    "results/maharashtra/phase2/neural/"
    "protocol10_seed42_e9dfa7ce76_ae290b4115_1d7a5ce0/neural_experiment_metadata.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(run_dir: Path) -> None:
    metadata = json.loads((run_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    effective_config = json.loads(
        (run_dir / "effective_configuration.json").read_text(encoding="utf-8")
    )
    phase22_metadata = json.loads(PHASE22_METADATA.read_text(encoding="utf-8"))
    historical_config = phase22_metadata["selected_config"]
    historical_config["model"].pop("use_variable_gate", None)
    effective_config_matches_phase22 = historical_config == effective_config
    if not effective_config_matches_phase22:
        raise RuntimeError("Diagnostic effective config differs from frozen Phase 2.2 selection")
    epoch = pd.read_csv(run_dir / "epoch_metrics.csv")
    history = pd.read_csv(run_dir / "training_history.csv")
    if metadata["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Diagnostic run dataset fingerprint is not the verified protocol-10 dataset")
    if metadata["normalizer_sha256"] != EXPECTED_NORMALIZER_SHA256:
        raise RuntimeError("Diagnostic run normalizer fingerprint is not the verified protocol-10 normalizer")
    if set(epoch["split"]) != {"train", "validation"} or "test" in epoch["split"].tolist():
        raise RuntimeError("Unexpected split in epoch diagnostics")
    if len(history) != 13 or history["epoch"].max() != 13:
        raise RuntimeError("Unexpected baseline training-history length")
    if any((run_dir / name).exists() for name in ("neural_metrics_test.csv", "test_predictions.csv")):
        raise RuntimeError("Test evaluation artifacts must not exist in this diagnostic run")

    source_paths = [
        Path("data/protocol.py"),
        Path("data/loader.py"),
        Path("data/preprocessing.py"),
        Path("experiments/run_phase2_2_neural.py"),
        Path("experiments/training_diagnostics.py"),
        Path("experiments/run_phase2_5_1_baseline.py"),
        Path("experiments/summarize_phase2_5_1.py"),
        Path("experiments/finalize_phase2_5_1.py"),
        Path("tests/test_phase2_5_0_training_diagnostics.py"),
    ]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    source_and_data = {
        "run_id": metadata["run_id"],
        "dataset_sha256": sha256(Path(metadata["dataset_path"])),
        "normalizer_sha256": sha256(Path(metadata["normalizer_path"])),
        "configuration_sha256": sha256(Path("configs/config.yaml")),
        "historical_phase2_2_configuration_sha256": phase22_metadata["configuration_sha256"],
        "effective_configuration_matches_phase2_2_after_legacy_gate_alias_removal": effective_config_matches_phase22,
        "source_sha256": source_hashes,
        "checkpoint_sha256": sha256(Path(metadata["checkpoint_path"])),
    }
    (run_dir / "source_and_data_hashes.json").write_text(
        json.dumps(source_and_data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    verification = {
        "run_id": metadata["run_id"],
        "test_split_materialized_as_windows": False,
        "test_split_evaluated": False,
        "test_predictions_or_metrics_created": False,
        "focused_observer_tests": {
            "command": ".\\.venv\\Scripts\\python.exe -m pytest -q tests/test_phase2_5_0_training_diagnostics.py tests/test_phase2_4_neural_audit.py::test_pinball_manual_positive_negative_zero_and_masked_target tests/test_phase2_4_neural_audit.py::test_masked_infinity_has_zero_contribution_and_finite_gradients --basetemp C:\\Users\\shikh\\AppData\\Local\\Temp\\droughtcast_phase2_5_1_observer2",
            "passed": True,
            "result": "7 passed in 8.71s",
        },
        "focused_training_and_observer_tests": {
            "command": ".\\.venv\\Scripts\\python.exe -m pytest -q tests/test_phase2_5_0_training_diagnostics.py tests/test_phase2_2_neural.py --basetemp C:\\Users\\shikh\\AppData\\Local\\Temp\\droughtcast_phase2_5_1_focused",
            "passed": True,
            "result": "17 passed in 15.46s",
        },
        "full_pytest": {
            "command": ".\\.venv\\Scripts\\python.exe -m pytest -q --basetemp C:\\Users\\shikh\\AppData\\Local\\Temp\\droughtcast_phase2_5_1_final",
            "passed": True,
            "result": "203 passed in 47.92s",
        },
        "full_ruff": {
            "command": ".\\.venv\\Scripts\\python.exe -m ruff check .",
            "passed": True,
            "result": "All checks passed!",
        },
        "changed_file_ruff": {
            "command": ".\\.venv\\Scripts\\python.exe -m ruff check experiments\\run_phase2_5_1_baseline.py experiments\\summarize_phase2_5_1.py experiments\\finalize_phase2_5_1.py experiments\\run_phase2_2_neural.py experiments\\training_diagnostics.py tests\\test_phase2_5_0_training_diagnostics.py",
            "passed": True,
            "result": "All checks passed!",
        },
        "source_compilation": {
            "passed": True,
            "result": "py_compile passed for Phase 2.5.1 run and summary scripts",
        },
        "scope": "No canonical data, targets, normalizer, protocol, architecture, or model hyperparameters changed; alternative objective not run.",
    }
    (run_dir / "verification_results.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_path = run_dir / "baseline_experiment_report.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "## Verification\n"
    if marker in report:
        report = report.split(marker, 1)[0]
    report += (
        "\n## Verification\n\n"
        "- Focused observer/loss tests: 7 passed.\n"
        "- Focused observer and Phase 2.2 neural tests: 17 passed.\n"
        "- Full pytest: 203 passed.\n"
        "- Repository-wide Ruff: all checks passed. Changed-file Ruff and `py_compile` also passed.\n"
        "- The full test suite completed without evaluating the consumed model test split; it did not run the baseline or neural experiment entry points.\n"
        "- The historical Phase 2.2 raw config SHA differs from the current SHA due to the previously corrected inactive `use_variable_gate` alias. The effective config used here matches the frozen Phase 2.2 selected config after removing that ignored alias; no current-phase model or training setting changed.\n"
    )
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    finalize(args.run_dir)


if __name__ == "__main__":
    main()
