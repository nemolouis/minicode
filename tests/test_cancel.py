"""Cooperative cancellation: Agent.cancel() from another thread stops a run at a safe point and leaves a resumable session."""
import json
import threading
import time

import pytest

from minicode.agent import Agent
from minicode.client import ModelResponse, ToolCall
from minicode.proc import run_shell
from minicode.session import load_session
from minicode.tools import Workspace
from minicode.trajectory import Trajectory


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    def __init__(self, script, hook=None):
        self.script, self.calls, self.hook = list(script), [], hook

    def chat(self, on_text=None, **kw):
        import copy
        self.calls.append(copy.deepcopy(kw))
        if self.hook:
            self.hook(len(self.calls))
        content, tok = self.script.pop(0)
        text = "".join(c for c in content if isinstance(c, str))
        if on_text and text:
            for ch in text:
                on_text(ch)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse(text, calls, "tool_calls" if calls else "stop", {"input_tokens": tok, "output_tokens": 5})


def valid_history(messages):
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            following = [x.get("tool_call_id") for x in messages[i + 1:i + 1 + len(ids)] if x["role"] == "tool"]
            assert sorted(following) == sorted(ids)
    return True


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    return w


def agent(tmp_path, ws, script, hook=None, **kw):
    return Agent(Script(script, hook), "m", Workspace(ws), trajectory=Trajectory(tmp_path / "run"), session=True, **kw)


def test_cancel_set_before_the_next_request_ends_the_run_cleanly(tmp_path, ws):
    (ws / "f.txt").write_text("x")
    a = agent(tmp_path, ws, [([call("1", "read_file", path="f.txt")], 1), (["never"], 1)])
    t = threading.Timer(0.0, a.cancel)
    a.client.hook = lambda n: a.cancel() if n == 1 else None            # cancelled while the first request is in flight
    r = a.run("t")
    t.cancel()
    assert r.status == "interrupted" and len(a.client.calls) == 1
    st = load_session(tmp_path / "run")
    assert valid_history(st["messages"]) and st["messages"] == [{"role": "user", "content": "t"}]       # the unfinished request left no trace


def test_cancel_between_tool_turns(tmp_path, ws):
    (ws / "f.txt").write_text("x")
    a = agent(tmp_path, ws, [([call("1", "read_file", path="f.txt")], 1), ([call("2", "read_file", path="f.txt")], 1), (["no"], 1)],
              hook=lambda n: a.cancel() if n == 2 else None)
    r = a.run("t")
    assert r.status == "interrupted" and r.steps == 2                          # step 2's request was answered (and logged) before the cancel took effect
    assert valid_history(load_session(tmp_path / "run")["messages"])


def test_cancel_stops_a_streaming_reply_at_the_next_piece(tmp_path, ws):
    pieces = []

    def on_text(p):
        pieces.append(p)
        if len(pieces) == 3:
            a.cancel()

    a = agent(tmp_path, ws, [(["abcdefghij"], 1)], on_text=on_text)
    r = a.run("t")
    assert r.status == "interrupted" and pieces == ["a", "b", "c"]


def test_cancel_kills_a_running_command_promptly(tmp_path, ws):
    marker = ws / "alive"
    a = agent(tmp_path, ws, [([call("1", "bash", command=f"(while true; do touch {marker}; sleep 0.1; done) & sleep 30")], 1), (["no"], 1)])
    threading.Timer(0.6, a.cancel).start()
    t0 = time.time()
    r = a.run("t")
    assert time.time() - t0 < 5 and r.status == "interrupted"
    marker.unlink(missing_ok=True)
    time.sleep(0.4)
    assert not marker.exists()                                                  # the whole process group died, not just the shell
    st = load_session(tmp_path / "run")
    assert valid_history(st["messages"]) and "stopped by the user" in st["messages"][-1]["content"]


def test_run_can_continue_after_a_cancel_and_the_flag_is_reset(tmp_path, ws):
    a = agent(tmp_path, ws, [(["first"], 1), (["second"], 1)], hook=lambda n: a.cancel() if n == 1 else None)
    assert a.run("one").status == "interrupted"
    a.client.hook = None
    r = a.run("two", resume=load_session(tmp_path / "run"))
    assert r.status == "done" and r.final_text == "second"


def test_run_shell_cancel_and_plain_timeout():
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t0 = time.time()
    r = run_shell("sleep 30", shell=True, timeout=20, cancel=ev)
    assert r.cancelled and not r.timed_out and time.time() - t0 < 3
    r = run_shell("sleep 30", shell=True, timeout=0.5, cancel=threading.Event())
    assert r.timed_out and not r.cancelled
    r = run_shell("echo hi", shell=True, timeout=5, cancel=threading.Event())
    assert r.returncode == 0 and r.stdout == "hi\n" and not r.cancelled and not r.timed_out
    assert run_shell("echo plain", shell=True, timeout=5).stdout == "plain\n"          # no cancel event: the old path


def test_cancel_trajectory_is_schema_valid(tmp_path, ws):
    a = agent(tmp_path, ws, [(["x"], 1)], hook=lambda n: a.cancel())
    a.run("t")
    ev = [json.loads(l) for l in (tmp_path / "run" / "trajectory.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "run_end" and ev[-1]["status"] == "interrupted"
