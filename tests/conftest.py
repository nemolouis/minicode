"""Every trajectory a test run produces with the current agent must satisfy the documented schema (minicode.trajectory)."""
import json

import pytest

from minicode.trajectory import validate_events


@pytest.fixture(autouse=True)
def _trajectories_follow_the_schema(request):
    yield
    tmp = request.node.funcargs.get("tmp_path") if hasattr(request.node, "funcargs") else None
    if tmp is None or not tmp.exists():
        return
    problems = []
    for path in tmp.rglob("trajectory.jsonl"):
        if "expect_invalid" in path.parts:                           # a test that checks the validator on a broken file
            continue
        try:
            events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        except ValueError:
            continue                                                  # a test that writes garbage on purpose
        if events and "schema_version" in events[0]:                 # written by today's Agent; hand-made fixtures are skipped
            problems += [f"{path.relative_to(tmp)}: {p}" for p in validate_events(events)]
    assert not problems, "trajectory schema violations:\n" + "\n".join(problems[:10])
