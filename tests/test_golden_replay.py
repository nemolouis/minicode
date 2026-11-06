"""Regression test for the agent loop: scripted trajectories recorded on real tasks (tests/golden, made by scripts/make_golden.py)
are replayed through the current loop. The recorded model answers are fed back in order; the replay must make the same tool calls,
hit the same errors, compact at the same moments and end the same way. If this fails after a change, either the change altered the
behaviour on purpose (regenerate the golden files and review the diff) or it is a bug."""
import json
from pathlib import Path

import pytest

from minicode.eval.replay import replay_run
from minicode.trajectory import validate_events

GOLDEN = Path(__file__).parent / "golden"
NAMES = sorted(p.name for p in GOLDEN.iterdir() if p.is_dir())
ROOT = Path(__file__).resolve().parent.parent


def events(name):
    return [json.loads(l) for l in (GOLDEN / name / "trajectory.jsonl").read_text().splitlines()]


def test_golden_set_is_present():
    assert NAMES == ["budget-01", "compaction-03", "errors-05", "max-steps-01", "patch-01", "solve-01", "stuck-01"]


@pytest.mark.parametrize("name", NAMES)
def test_golden_trajectory_replays_identically(name):
    r = replay_run(GOLDEN / name / "trajectory.jsonl", ROOT / "tasks")
    assert r["status"] == "ok", r
    assert r["output_mismatches"] == 0, r                   # the tool outputs must match too, not only the calls


@pytest.mark.parametrize("name", NAMES)
def test_golden_trajectories_are_schema_valid(name):
    assert validate_events(events(name)) == []


def test_the_golden_set_covers_what_it_is_meant_to_cover():
    ev = {n: events(n) for n in NAMES}
    kinds = lambda n: [e["event"] for e in ev[n]]
    assert any(e.get("parallel") for e in ev["solve-01"] if e["event"] == "tool_result")             # concurrent reads
    errs = [e for e in ev["errors-05"] if e["event"] == "tool_result" and e["is_error"]]
    texts = " ".join(e["output"] for e in errs)
    for needle in ("unknown tool", "invalid arguments", "not valid JSON", "old_string not found", "blocked by safety policy"):
        assert needle in texts, needle
    assert [e["kind"] for e in ev["compaction-03"] if e["event"] == "compaction"] == ["elide", "elide", "summary"]
    assert ev["max-steps-01"][-1]["status"] == "max_steps"
    assert [(e["count"], e["action"]) for e in ev["stuck-01"] if e["event"] == "stuck"] == [(3, "warn"), (4, "warn"), (5, "stop")]
    assert ev["stuck-01"][-1]["status"] == "stuck"
    assert any(e["event"] == "budget" for e in ev["budget-01"]) and ev["budget-01"][-1]["status"] == "budget"
    assert ev["budget-01"][-1]["cost_usd"] > 0.001
    patch_calls = [e for e in ev["patch-01"] if e["event"] == "tool_result"]
    assert [e["name"] for e in patch_calls] == ["symbols", "read_file", "apply_patch", "bash"] and not any(e["is_error"] for e in patch_calls)
    assert "applied patch: updated stats.py" in patch_calls[2]["output"] and "def moving_average" in patch_calls[0]["output"]
    assert kinds("solve-01")[-1] == "run_end"
