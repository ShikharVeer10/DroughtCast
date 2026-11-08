import numpy as np
import pytest

from evaluation.independent import OUTPUT_COLUMNS, validate_output_schema


def test_synthetic_output_schema_accepts_all_required_fields():
    row = {column: np.nan for column in OUTPUT_COLUMNS}
    row.update({"scale": "spei_3", "lead": "T+1"})
    validate_output_schema([row])


def test_synthetic_output_schema_rejects_invalid_lead():
    row = {column: np.nan for column in OUTPUT_COLUMNS}
    row.update({"scale": "spei_3", "lead": "T+5"})
    with pytest.raises(ValueError):
        validate_output_schema([row])
