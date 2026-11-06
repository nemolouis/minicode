"""The trajectory schema: what valid looks like, what each rule catches, and the check-trajectories command."""
import copy
import json

import pytest

from miniagent import session as sess
from miniagent.agent import Agent
from miniagent.client import ModelResponse, ToolCall
from miniagent.features import EXPLORE_PROMPT
from miniagent.permissions import Permissions
from miniagent.provenance import provenance
from miniagent.tools import Workspace
from miniagent.trajectory import SCHEMA_VERSION, Trajectory, read_events, validate_events


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Client:
    def __init__(self, parent, child=()):
        self.p, self.c = list(parent), list(child)

    def chat(self, **kw):
        content, tok = (self.c if kw.get("system") == EXPLORE_PROMPT else self.p).pop(0)
        calls = [x for x in content if isinstance(x, ToolCall)]
        return ModelResponse("".join(x for x in content if isinstance(x, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 5, "cache_hit_tokens": 0})


@pytest.fixture
def rich(tmp_path):
    """A trajectory containing every event type the current agent writes."""
    w = tmp_path / "w"
    w.mkdir()
    (w / "a.txt").write_text("A")
    big = [(["SUM"], 5)]
    parent = [([call("1", "read_file", path="a.txt"), call("2", "grep", pattern="A"), call("3", "explore", question="q?")], 10),
              ([call("4", "write_file", path="b.txt", content="B"), call("5", "bash", command="echo hi")], 90_000),
              ([call("6", "write_file", path=".env", content="x")], 10), (["done"], 10)]
    a = Agent(Client(parent, [(["report"], 5)] + big), "fake", Workspace(w), trajectory=Trajectory(tmp_path / "run"), session=True,
              features=("explore",), permissions=Permissions(deny=["write_file(.env*)"], allow=["bash(echo*)"]),
              context_limit=1000, keep_recent=2)
    a.run("task")
    ev = read_events(tmp_path / "run" / "trajectory.jsonl")
    kinds = {e["event"] for e in ev}
    assert {"run_start", "model_call", "tool_result", "subagent", "checkpoint", "run_end"} <= kinds
    assert any(e.get("parallel") for e in ev) and any(e.get("permission") == "deny" for e in ev)
    return ev, tmp_path


def test_the_current_agent_writes_valid_trajectories(rich):
    ev, _ = rich
    assert validate_events(ev) == []
    assert ev[0]["schema_version"] == SCHEMA_VERSION and ev[0]["provenance"]["python"]


def test_provenance_fields():
    p = provenance()
    assert set(p) == {"git_sha", "git_dirty", "dirty_files", "python", "platform"}
    assert p["git_sha"] is None or len(p["git_sha"]) == 40
    assert p["git_dirty"] in (None, True, False)


def mutate(ev, fn):
    ev = copy.deepcopy(ev)
    fn(ev)
    return validate_events(ev)


def first(ev, kind): return next(e for e in ev if e["event"] == kind)


@pytest.mark.parametrize("name,fn,expect", [
    ("missing field", lambda ev: first(ev, "model_call").pop("usage"), "missing `usage`"),
    ("wrong type", lambda ev: first(ev, "tool_result").update(is_error="no"), "`is_error` should be bool"),
    ("bool as int", lambda ev: first(ev, "model_call").update(step=True), "`step` should be int"),
    ("unknown event", lambda ev: ev.insert(2, {"ts": 1, "event": "mystery"}), "unknown event type"),
    ("no ts", lambda ev: first(ev, "model_call").pop("ts"), "numeric `ts`"),
    ("steps not increasing", lambda ev: [e for e in ev if e["event"] == "model_call"][1].update(step=1), "does not increase"),
    ("orphan tool result", lambda ev: first(ev, "tool_result").update(step=99), "no model_call for step 99"),
    ("result for a call that was not made", lambda ev: first(ev, "tool_result").update(tool_call_id="zzz"), "was not requested"),
    ("permission without rule", lambda ev: next(e for e in ev if e.get("permission")).pop("rule"), "without `rule`"),
    ("bad permission", lambda ev: next(e for e in ev if e.get("permission")).update(permission="maybe"), "allow, ask or deny"),
    ("parallel of one", lambda ev: next(e for e in ev if e.get("parallel")).update(parallel=1), "batches of 2"),
    ("bad compaction kind", lambda ev: first(ev, "compaction").update(kind="shrink"), "unknown compaction kind"),
    ("tool call without id", lambda ev: first(ev, "model_call")["tool_calls"].append({"name": "x"}), "id, name and an input"),
    ("wrong version", lambda ev: ev[0].update(schema_version=99), "not the current"),
    ("no provenance", lambda ev: ev[0].pop("provenance"), "missing `provenance`"),
    ("subagent tokens", lambda ev: first(ev, "subagent").pop("input_tokens"), "missing `input_tokens`"),
])
def test_each_rule_catches_its_violation(rich, name, fn, expect):
    errs = mutate(rich[0], fn)
    assert any(expect in e for e in errs), (name, errs)


def test_must_start_with_run_start_and_not_be_empty():
    assert validate_events([]) == ["empty trajectory"]
    assert "expected run_start" in validate_events([{"ts": 1, "event": "model_call"}])[0]


def test_legacy_trajectories_are_checked_loosely():
    legacy = [{"ts": 1, "event": "run_start", "task": "t"}, {"ts": 2, "event": "whatever", "x": 1}]
    assert validate_events(legacy) == []
    assert validate_events([{"event": "run_start"}])                          # still needs ts


def test_rewind_and_resume_branch_is_valid_and_overlap_without_rewind_is_not(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    (w / "f.txt").write_text("0")
    a = Agent(Client([([call("1", "write_file", path="f.txt", content="1")], 10), ([call("2", "write_file", path="f.txt", content="2")], 10),
                      (["done"], 10)]), "fake", Workspace(w), trajectory=Trajectory(tmp_path / "run"), session=True)
    a.run("t")
    sess.rewind(tmp_path / "run", 2, w)
    b = Agent(Client([(["again"], 10)]), "fake", Workspace(w), trajectory=Trajectory(tmp_path / "run"), session=True)
    b.run("go on", resume=sess.load_session(tmp_path / "run"))
    ev = read_events(tmp_path / "run" / "trajectory.jsonl")
    assert [e["event"] for e in ev].count("rewind") == 1 and validate_events(ev) == []
    assert [e["step"] for e in ev if e["event"] == "model_call"] == [1, 2, 3, 2]       # branch: step 2 again after the rewind
    without = [e for e in ev if e["event"] != "rewind"]
    assert any("does not increase" in e for e in validate_events(without))


def test_check_trajectories_command(rich, tmp_path, capsys):
    from miniagent.eval.cli import main
    ev, base = rich
    assert main(["check-trajectories", str(base / "run")]) == 0
    assert "2/2 valid" in capsys.readouterr().out      # the run and its explore-1 sub-agent trajectory
    bad = tmp_path / "expect_invalid"
    bad.mkdir()
    broken = copy.deepcopy(ev)
    first_call = next(e for e in broken if e["event"] == "model_call")
    first_call.pop("usage")
    (bad / "trajectory.jsonl").write_text("".join(json.dumps(e) + "\n" for e in broken))
    assert main(["check-trajectories", str(bad)]) == 1
    assert "missing `usage`" in capsys.readouterr().out
