"""Relative links in the markdown files must point at files that exist (after the README was split into docs/ this is easy to break)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "README.md", ROOT / "PROJECT.md", ROOT / "results" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
FILES = [f for f in FILES if f.exists() and not f.name.startswith(("导学", "面经"))]       # those two are local-only (gitignored)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


@pytest.mark.parametrize("md", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(md):
    text = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)         # ignore code blocks
    broken = []
    for target in LINK.findall(text):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        path = target.split("#")[0]
        if path and not (md.parent / path).exists():
            broken.append(target)
    assert not broken, f"{md.relative_to(ROOT)}: broken links {broken}"
