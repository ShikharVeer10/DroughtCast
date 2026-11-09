from scripts.phase5_9_final_research_audit import PHASE_DIRS


def test_audit_covers_all_required_phases():
    assert {"4.7", "5.0", "5.1", "5.1.1", "5.2", "5.3", "5.4", "5.5A", "5.6", "5.7", "5.8"} == set(PHASE_DIRS)
