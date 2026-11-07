import pytest

from experiments.audit_power_october_reconciliation import classify_october_value


@pytest.mark.parametrize(
    ("existing", "monthly", "daily", "complete", "expected"),
    [
        (1.0, 1.0, None, False, "verified"),
        (1.0, None, 1.0, True, "provisional"),
        (1.0, None, None, False, "unverified"),
        (None, None, None, False, "unavailable"),
        (None, None, 2.0, True, "provisional"),
    ],
)
def test_october_source_classification(existing, monthly, daily, complete, expected):
    classification, _ = classify_october_value(existing, monthly, daily, complete)
    assert classification == expected


def test_october_value_without_original_source_is_unverified_if_no_match():
    classification, reason = classify_october_value(
        27.4,
        monthly_value=None,
        fresh_daily_value=None,
        daily_complete=False,
    )
    assert classification == "unverified"
    assert "original source cannot be established" in reason


def test_october_fallback_requires_complete_fresh_daily_month():
    classification, _ = classify_october_value(
        27.4,
        monthly_value=None,
        fresh_daily_value=27.4,
        daily_complete=False,
    )
    assert classification == "unverified"
