import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"tests", ".venv", ".git", ".pytest_tmp", "__pycache__", "node_modules"}
BANNED = re.compile(r'"(2010-12-01|2012-01-01|2015-12-01|2017-01-01)"')


def test_no_hardcoded_split_dates():
    bad = []
    for path in ROOT.rglob("*.py"):
        if SKIP_DIRS & set(path.relative_to(ROOT).parts) or path.name == "protocol.py":
            continue
        if BANNED.search(path.read_text(encoding="utf-8")):
            bad.append(str(path.relative_to(ROOT)))
    assert not bad, f"Hard-coded split dates found; import SPLITS from data.protocol instead: {bad}"