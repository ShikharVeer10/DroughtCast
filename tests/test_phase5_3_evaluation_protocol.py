from evaluation.protocol import validate_role_assignments


def sample(role, end, valid=True):
    return {
        "role": role,
        "site": "a",
        "input_dates": [f"{end}-01"],
        "target_dates": [f"{end}-02"],
        "spei_target_valid": valid,
    }


def test_roles_reject_overlap_and_invalid_targets():
    result = validate_role_assignments(
        [sample("training", "2010-01"), sample("validation", "2010-01", False)],
        embargo_months=0,
    )
    assert not result["pass"]
    assert any("multiple roles" in item for item in result["violations"])
    assert any("invalid SPEI" in item for item in result["violations"])


def test_roles_reject_insufficient_embargo():
    result = validate_role_assignments(
        [sample("training", "2010-01"), sample("validation", "2011-01")],
        embargo_months=22,
    )
    assert not result["pass"]
    assert any("embargo" in item for item in result["violations"])
