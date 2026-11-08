import json
from pathlib import Path


def test_phase316_hash_audit_reports_no_failures():
    root = Path("results/maharashtra/phase3/phase3_16_reproducibility_audit")
    audit = json.loads((root / "artifact_hash_verification.json").read_text())
    assert audit["phase3_14_hash_failures"] == 0
    assert audit["phase3_15_hash_failures"] == 0
