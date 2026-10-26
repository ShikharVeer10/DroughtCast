import json

import pytest

from data.loader import INDEX_ALIGNMENT, validate_dataset_protocol


def test_current_dataset_protocol_is_accepted(tmp_path):
    path = tmp_path / "dataset_metadata.json"
    path.write_text(json.dumps({"schema_version": 2, "index_alignment": INDEX_ALIGNMENT}), encoding="utf-8")
    assert validate_dataset_protocol(path)["schema_version"] == 2


def test_stale_dataset_protocol_is_rejected(tmp_path):
    path = tmp_path / "dataset_metadata.json"
    path.write_text(json.dumps({"model_features": ["oni", "dmi", "nao", "pdo"]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="predates the current teleconnection alignment"):
        validate_dataset_protocol(path)


def test_missing_dataset_provenance_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset provenance is missing"):
        validate_dataset_protocol(tmp_path / "absent.json")
