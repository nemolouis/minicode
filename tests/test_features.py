"""Schema validation, concurrent read-only tools, AGENTS.md memory, todo list and the explore sub-agent (no network)."""
import copy
import json
import threading
import time

import pytest

from minicode.agent import Agent
from minicode.client import ModelResponse, ToolCall
from minicode.eval.grade import trajectory_stats
from minicode.features import EXPLORE_PROMPT, MEMORY_LIMIT
from minicode.sandbox import SandboxError
from minicode.schema import signature, validate
from minicode.tools import Tool, Workspace
from minicode.trajectory import Trajectory


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class RoleClient:
    """Scripted client with one script per role (parent / explore child), so threads cannot reorder the scripts."""
    def __init__(self, parent, child=()):
        self.scripts = {"parent": list(parent), "child": list(child)}
        self.calls = {"parent": [], "child": []}
        self.lock = threading.Lock()

    def chat(self, **kw):
        role = "child" if kw.get("system") == EXPLORE_PROMPT else "parent"
        with self.lock:
            self.calls[role].append(copy.deepcopy(kw))
            content, tok = self.scripts[role].pop(0)
        calls = [c for c in content if isinstance(c, ToolCall)]
        txt = "".join(c for c in content if isinstance(c, str))
        return ModelResponse(txt, calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 10, "cache_hit_tokens": 0})


def make(tmp_path, parent, child=(), **kw):
    return Agent(RoleClient(parent, child), "fake", Workspace(tmp_path), trajectory=Trajectory(tmp_path / "run"), **kw)


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "run" / "trajectory.jsonl").read_text().splitlines()]


def tool_msgs(agent, n=1):
    return [m for m in agent.client.calls["parent"][n]["messages"] if m["role"] == "tool"]


# ---------------------------------------------------------------- schema
SCHEMA = {"type": "object", "required": ["path"], "properties": {
    "path": {"type": "string"}, "offset": {"type": "integer"}, "mode": {"type": "string", "enum": ["a", "b"]},
    "items": {"type": "array", "items": {"type": "object", "required": ["k"], "properties": {"k": {"type": "string"}}}}}}


def test_validate_catches_the_usual_mistakes():
    assert validate({"path": "x"}, SCHEMA) == []
    assert validate({}, SCHEMA) == ["path: required"]
    assert "expected integer, got str '5'" in validate({"path": "x", "offset": "5"}, SCHEMA)[0]
    assert validate({"path": "x", "offset": True}, SCHEMA)               # bool is not an integer
    assert validate({"path": "x", "mode": "c"}, SCHEMA)
    assert validate({"path": "x", "items": [{"k": "a"}, {}]}, SCHEMA) == ["items[1].k: required"]
    assert validate({"path": "x", "unknown": 1, "anyOf": 3}, {**SCHEMA, "anyOf": []}) == []   # unknown keys are not errors
    assert signature(SCHEMA) == "{path: string, offset?: integer, mode?: string, items?: array}"


def test_bad_arguments_get_a_precise_error_and_the_tool_does_not_run(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\n")
    a = make(tmp_path, [([call("1", "read_file", path="f.txt", offset="2"), call("2", "read_file")], 10), (["ok"], 10)])
    a.run("t")
    m1, m2 = tool_msgs(a)
    assert "invalid arguments for read_file" in m1["content"] and "offset: expected integer" in m1["content"]
    assert "path: required" in m2["content"] and "Expected {path: string" in m2["content"]


def test_validation_can_be_switched_off(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\n")
    a = make(tmp_path, [([call("1", "read_file", path="f.txt", offset="2")], 10), (["ok"], 10)], validate_args=False)
    a.run("t")
    assert tool_msgs(a)[0]["content"].strip().endswith("b")           # the old lenient int("2") path


# ---------------------------------------------------------------- concurrency
def _slow_tool(name, log, barrier=None, read_only=True, confirm=False):
    def run(args):
        log.append(("start", name, args["i"]))
        if barrier is not None:
            barrier.wait(timeout=2)         # only passes when the other call is running at the same time
        time.sleep(0.01)
        log.append(("end", name, args["i"]))
        return f"{name}-{args['i']}"
    return Tool(name, "d", {"type": "object", "properties": {"i": {"type": "integer"}}, "required": ["i"]}, run,
                needs_confirm=confirm, read_only=read_only)


def test_read_only_calls_run_concurrently_but_results_keep_order(tmp_path):
    log, barrier = [], threading.Barrier(3)
    a = make(tmp_path, [([call(str(i), "slow", i=i) for i in range(3)], 10), (["ok"], 10)],
             extra_tools={"slow": _slow_tool("slow", log, barrier)})
    assert a.run("t").status == "done"
    assert [m["content"] for m in tool_msgs(a)] == ["slow-0", "slow-1", "slow-2"]
    results = [e for e in events(tmp_path) if e["event"] == "tool_result"]
    assert [e["tool_call_id"] for e in results] == ["0", "1", "2"] and all(e["parallel"] == 3 for e in results)
    assert not any(e["is_error"] for e in results)                    # the barrier only opens if all three overlapped


def test_parallel_off_runs_one_at_a_time(tmp_path):
    log, barrier = [], threading.Barrier(2)
    a = make(tmp_path, [([call("0", "slow", i=0), call("1", "slow", i=1)], 10), (["ok"], 10)], parallel_tools=False,
             extra_tools={"slow": _slow_tool("slow", log, barrier)})
    a.run("t")
    assert all(e["is_error"] for e in events(tmp_path) if e["event"] == "tool_result")   # barrier timed out: never overlapped
    assert not any("parallel" in e for e in events(tmp_path))


def test_a_write_between_reads_is_a_barrier_and_confirm_tools_stay_sequential(tmp_path):
    log = []
    tools = {"r": _slow_tool("r", log), "w": _slow_tool("w", log, read_only=False), "c": _slow_tool("c", log, confirm=True)}
    script = [call("0", "r", i=0), call("1", "r", i=1), call("2", "w", i=2), call("3", "r", i=3), call("4", "c", i=4), call("5", "c", i=5)]
    a = make(tmp_path, [(script, 10), (["ok"], 10)], extra_tools=tools)
    a.run("t")
    starts = [x for x in log if x[0] == "start"]
    assert [x[2] for x in starts][:2] in ([0, 1], [1, 0])             # the first two reads may interleave ...
    ends = {x[2]: n for n, x in enumerate(log) if x[0] == "end"}
    first_w_start = next(n for n, x in enumerate(log) if x[:2] == ("start", "w"))
    assert ends[0] < first_w_start and ends[1] < first_w_start        # ... but both finish before the write starts
    last_r = next(n for n, x in enumerate(log) if x[:3] == ("start", "r", 3))
    assert last_r > ends[2]                                           # and the read after the write starts after it ended
    for n, x in enumerate(log):                                       # confirm-needing calls never overlap each other
        if x[:2] == ("start", "c"):
            assert log[n + 1][:2] == ("end", "c")
    assert [m["content"] for m in tool_msgs(a)] == ["r-0", "r-1", "w-2", "r-3", "c-4", "c-5"]


def test_invalid_call_is_not_run_in_the_concurrent_batch(tmp_path):
    log, barrier = [], threading.Barrier(2)
    a = make(tmp_path, [([call("0", "slow", i=0), call("1", "slow", i="x"), call("2", "slow", i=2)], 10), (["ok"], 10)],
             extra_tools={"slow": _slow_tool("slow", log, barrier)})
    a.run("t")
    msgs = [m["content"] for m in tool_msgs(a)]
    assert "invalid arguments" in msgs[1]
    assert [x[2] for x in log if x[0] == "start"] == [0, 2] or [x[2] for x in log if x[0] == "start"] == [2, 0]


def test_sandbox_error_in_a_worker_ends_the_run_as_error(tmp_path):
    def boom(args): raise SandboxError("docker died")
    t = Tool("boom", "d", {"type": "object", "properties": {}}, boom, read_only=True)
    a = make(tmp_path, [([call("0", "boom"), call("1", "boom")], 10)], extra_tools={"boom": t})
    r = a.run("t")
    assert r.status == "error" and "docker died" in r.error


# ---------------------------------------------------------------- memory
def test_memory_is_opt_in_and_loaded_from_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Run tests with `make check`.\n")
    assert "make check" not in make(tmp_path, []).system
    a = make(tmp_path, [(["ok"], 10)], features=("memory",))
    assert "make check" in a.system and "# Project instructions (from AGENTS.md" in a.system
    a.run("t")
    assert "make check" in a.client.calls["parent"][0]["system"]
    assert events(tmp_path)[0]["config"]["features"] == ["memory"]


def test_memory_fallback_truncation_idempotence_and_symlink(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("x" * (MEMORY_LIMIT + 500))
    s = make(tmp_path, [], features=("memory",)).system
    assert "[truncated]" in s and "x" * MEMORY_LIMIT in s and "x" * (MEMORY_LIMIT + 1) not in s
    again = make(tmp_path, [], features=("memory",), system=s).system
    assert again == s                                                  # a recorded system prompt is not extended twice
    (tmp_path / "CLAUDE.md").unlink()
    (tmp_path / "secret.txt").write_text("TOPSECRET")
    (tmp_path / "AGENTS.md").symlink_to(tmp_path / "secret.txt")
    assert "TOPSECRET" not in make(tmp_path, [], features=("memory",)).system


def test_unknown_feature_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, [], features=("telepathy",))


# ---------------------------------------------------------------- todo
def test_todo_tool_keeps_state_and_enforces_one_in_progress(tmp_path):
    todos = [{"content": "read", "status": "completed"}, {"content": "fix", "status": "in_progress"}, {"content": "test", "status": "pending"}]
    bad = [{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}]
    a = make(tmp_path, [([call("1", "todo_write", todos=todos), call("2", "todo_write", todos=bad),
                          call("3", "todo_write", todos=[{"content": "a", "status": "done"}])], 10), (["ok"], 10)], features=("todo",))
    a.run("t")
    m = [x["content"] for x in tool_msgs(a)]
    assert m[0].endswith("[x] read\n[~] fix\n[ ] test") and a.todos == todos
    assert "at most one" in m[1] and "must be one of" in m[2]          # a rejected update leaves the list unchanged
    assert "todo_write" not in make(tmp_path, []).tools


def test_todo_list_survives_summary_compaction(tmp_path):
    a = make(tmp_path, [(["SUMMARY"], 10)], features=("todo",), keep_recent=2)
    a.todos = [{"content": "fix", "status": "in_progress"}]
    msgs = [{"role": "user", "content": "task"}]
    for i in range(6):
        msgs += [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": "u" * 3000}]
    a.context_limit = 1000
    out = a._compact(msgs)
    assert "SUMMARY" in out[0]["content"] and "[~] fix" in out[0]["content"]


# ---------------------------------------------------------------- glob
def test_glob_tool(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.py").write_text("")
    outside = tmp_path.parent / (tmp_path.name + "-out")
    outside.mkdir()
    (outside / "leak.py").write_text("")
    (tmp_path / "link.py").symlink_to(outside / "leak.py")
    ws = Workspace(tmp_path)
    from minicode.tools import ToolError, glob_tool
    g = glob_tool(ws).func
    assert g({"pattern": "**/*.py"}) == "pkg/a.py"
    for bad in ("../*", "/etc/*"):
        with pytest.raises(ToolError):
            g({"pattern": bad})
    assert "glob" not in make(tmp_path, []).tools and "glob" in make(tmp_path, [], features=("glob",)).tools


# ---------------------------------------------------------------- explore
def test_explore_subagent_is_read_only_isolated_and_accounted(tmp_path):
    (tmp_path / "big.py").write_text("def f():\n    return 1\n" * 200)
    child = [([call("c1", "read_file", path="big.py"), call("c2", "write_file", path="evil.txt", content="x"),
               call("c3", "bash", command="touch evil2")], 500),
             (["f is defined at big.py:1 and returns 1."], 700)]
    parent = [([call("1", "explore", question="where is f?")], 100), (["done"], 120)]
    a = make(tmp_path, parent, child, features=("explore",))
    r = a.run("t")
    assert r.status == "done"
    # the child saw only read-only tools, and the write/bash attempts were refused, not executed
    child_tools = {t["function"]["name"] for t in a.client.calls["child"][0]["tools"]}
    assert child_tools == {"read_file", "grep", "glob"}
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "evil2").exists()
    assert any("unknown tool: write_file" in m["content"] for m in a.client.calls["child"][1]["messages"] if m["role"] == "tool")
    # the parent got only the short report, not the file contents
    assert tool_msgs(a)[0]["content"] == "f is defined at big.py:1 and returns 1."
    assert "def f()" not in json.dumps(a.client.calls["parent"][1]["messages"])
    # tokens: parent (100 + 120) + child (500 + 700) input, 10 output per call
    assert r.input_tokens == 1420 and r.output_tokens == 40
    ev = events(tmp_path)
    sub = next(e for e in ev if e["event"] == "subagent")
    assert sub["steps"] == 2 and sub["input_tokens"] == 1200
    assert (tmp_path / "run" / "explore-1" / "trajectory.jsonl").exists()
    st = trajectory_stats(ev)
    assert st["subagent_runs"] == 1 and st["uncached_input_tokens"] == 1420 and st["output_tokens"] == 40


def test_explore_cannot_recurse_and_parallel_explores_run_together(tmp_path):
    child = [(["report A"], 50), (["report B"], 60)]
    parent = [([call("1", "explore", question="A?"), call("2", "explore", question="B?")], 100), (["done"], 100)]
    a = make(tmp_path, parent, child, features=("explore", "todo"))
    a.run("t")
    assert {t["function"]["name"] for t in a.client.calls["child"][0]["tools"]} == {"read_file", "grep", "glob"}   # no explore, no todo
    res = [e for e in events(tmp_path) if e["event"] == "tool_result"]
    assert [e["parallel"] for e in res] == [2, 2]
    assert sorted(m["content"] for m in tool_msgs(a)) == ["report A", "report B"]
    assert sorted(p.name for p in (tmp_path / "run").iterdir() if p.is_dir()) == ["explore-1", "explore-2"]


def test_explore_reports_an_unfinished_child(tmp_path):
    loop = ([call("c", "grep", pattern="a")], 10)
    a = make(tmp_path, [([call("1", "explore", question="q")], 10), (["ok"], 10)], [loop] * 3, features=("explore",), explore_max_steps=3)
    a.run("t")
    assert "without a full answer" in tool_msgs(a)[0]["content"] and "max_steps" in tool_msgs(a)[0]["content"]
