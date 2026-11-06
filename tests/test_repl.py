"""The interactive terminal client, driven by scripted input (no network, no model)."""
import io
import json

import pytest

from miniagent import repl as R
from miniagent.agent import Agent
from miniagent.client import ModelResponse, ToolCall
from miniagent.repl import Repl, Terminal, describe_action, workspace_diff
from miniagent.tools import Workspace


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    """Scripted model: items are (content, tokens), the string "INTERRUPT", or a callable(kw) -> item."""
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, on_text=None, **kw):
        import copy
        self.calls.append(copy.deepcopy(kw))
        item = self.script.pop(0)
        if item == "INTERRUPT":
            raise KeyboardInterrupt
        content, tok = item
        text = "".join(c for c in content if isinstance(c, str))
        if on_text and text:
            for ch in (text[: len(text) // 2], text[len(text) // 2:]):
                on_text(ch)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse(text, calls, "tool_calls" if calls else "stop", {"input_tokens": tok, "output_tokens": 7, "cache_hit_tokens": 0})


@pytest.fixture
def env(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    (w / "a.txt").write_text("one\ntwo\n")
    return tmp_path, w


def free_dir(tmp, claim=None):
    """The next unused runs/rN (a test may start several clients in one workspace; each needs its own session directory)."""
    n = 1
    while (tmp / "runs" / f"r{n}").exists() or (claim is not None and (tmp / "runs" / f"r{n}") == claim):
        n += 1
    return tmp / "runs" / f"r{n}"


def start(env, script, lines, *, model="fake", price=None, auto=False, stream=False, features=(), resume=None, run_dir=None, **kw):
    tmp, w = env
    run_dir = run_dir or free_dir(tmp)
    client = Script(script)
    out = io.StringIO()
    term = Terminal(inp=io.StringIO("\n".join(lines) + "\n"), out=out, color=False)

    def make_agent(trajectory=None, on_text=None, confirm=None):
        return Agent(client, model, Workspace(w), trajectory=trajectory, on_text=on_text, confirm=confirm, session=True,
                     features=features, keep_background=True, **kw)

    r = Repl(make_agent=make_agent, run_dir=run_dir, workspace=w, term=term, new_run_dir=lambda: free_dir(tmp, claim=run_dir), price=price,
             auto_approve=auto,
             stream=stream, resume=resume)
    return r, client, out


def go(*a, **kw):
    r, client, out = start(*a, **kw)
    assert r.run() == 0
    return r, client, out.getvalue()


# ---------------------------------------------------------------- the loop
def test_a_turn_prints_the_answer_and_a_summary_and_exit_leaves(env):
    r, client, out = go(env, [(["Hello there."], 100)], ["hi", "/exit"])
    assert "Hello there." in out and "[done] 1 steps total, 100 in / 7 out tokens" in out
    assert "continue later with: miniagent" in out and str(r.run_dir) in out
    assert len(client.calls) == 1


def test_eof_leaves_and_blank_lines_are_ignored(env):
    r, client, out = go(env, [(["a"], 10)], ["", "   ", "go"])                                 # no /exit: end of input
    assert len(client.calls) == 1 and "[done]" in out


def test_turns_continue_one_conversation(env):
    script = [([call("1", "read_file", path="a.txt")], 100), (["first answer"], 120), (["second answer"], 150)]
    r, client, out = go(env, script, ["look at a.txt", "and again?"])
    assert r.turns == 2 and r.state["step"] == 3 and r.tokens == {"in": 370, "out": 21}
    second = client.calls[2]["messages"]
    assert [m["content"] for m in second if m["role"] == "user"] == ["look at a.txt", "and again?"]
    assert any(m["role"] == "tool" for m in second)                                           # the first turn's tool exchange is still there
    events = [json.loads(l) for l in (r.run_dir / "trajectory.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events if e["event"] == "model_call"] == [1, 2, 3] and [e["event"] for e in events].count("run_resume") == 1


def test_tool_calls_are_listed_with_a_mark(env):
    script = [([call("1", "read_file", path="a.txt"), call("2", "grep", pattern="two"), call("3", "bash", command="false"),
                call("4", "read_file", path="missing.txt")], 100), (["done"], 120)]
    _, _, out = go(env, script, ["go"], auto=True)
    assert "✓ read_file a.txt  (parallel×2)" in out and "✓ grep two  (parallel×2)" in out
    assert "✓ bash false" in out                                                                # exit code 1 is the command's result, not a tool error
    assert "✗ read_file missing.txt" in out and "not a file: missing.txt" in out                # a real tool error is marked and explained


def test_streamed_text_is_shown_once(env):
    _, _, out = go(env, [(["streamed reply"], 10)], ["go"], stream=True)
    assert out.count("streamed reply") == 1


def test_multi_line_input(env):
    _, client, _ = go(env, [(["a"], 1), (["b"], 1)], ["first line \\", "second line \\", "third", '"""', "block one", "block two", '"""'])
    assert client.calls[0]["messages"][-1]["content"] == "first line \nsecond line \nthird"
    last = [m["content"] for m in client.calls[1]["messages"] if m["role"] == "user"][-1]
    assert last == "\nblock one\nblock two\n"
    _, client, _ = go(env, [(["a"], 1)], ['"""one-liner"""'])
    assert client.calls[0]["messages"][-1]["content"] == "one-liner"


def test_unknown_commands_are_reported_not_sent_to_the_model(env):
    _, client, out = go(env, [], ["/nope", "/help"])
    assert "unknown command /nope" in out and "/rewind N" in out and client.calls == []


def test_ctrl_c_during_a_turn_keeps_the_conversation_and_the_prompt(env):
    script = [([call("1", "read_file", path="a.txt")], 100), "INTERRUPT", (["recovered"], 120)]
    r, client, out = go(env, script, ["start", "second try"])
    assert "[interrupted]" in out and "[done]" in out
    assert r.state["step"] >= 1 and any(m["role"] == "tool" for m in client.calls[-1]["messages"])


def test_ctrl_c_at_the_prompt_exits_cleanly(env):
    r, client, out = start(env, [], [])
    r.term.read = lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt)
    assert r.run() == 0


# ---------------------------------------------------------------- approvals
def test_approval_shows_a_diff_and_n_refuses(env):
    tmp, w = env
    script = [([call("1", "edit_file", path="a.txt", old_string="two", new_string="TWO")], 100), (["ok"], 100)]
    r, client, out = go(env, script, ["edit it", "n"])
    assert "needs your approval" in out and "edit a.txt" in out and "-two" in out and "+TWO" in out
    assert (w / "a.txt").read_text() == "one\ntwo\n"
    tool_msg = [m["content"] for m in client.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "user denied" in tool_msg


def test_y_approves_and_a_approves_the_rest_of_the_session(env):
    tmp, w = env
    script = [([call("1", "edit_file", path="a.txt", old_string="two", new_string="TWO")], 100),
              ([call("2", "write_file", path="b.txt", content="x")], 100), (["ok"], 100)]
    r, _, out = go(env, script, ["edit", "y", "write", "n"])
    assert (w / "a.txt").read_text() == "one\nTWO\n" and not (w / "b.txt").exists()                  # y applied the edit, n refused the write
    r2, _, out2 = go((tmp, w), [([call("1", "write_file", path="c.txt", content="x")], 1), ([call("2", "write_file", path="d.txt", content="y")], 1), (["ok"], 1)],
                     ["go", "a"])
    assert (w / "c.txt").exists() and (w / "d.txt").exists() and out2.count("needs your approval") == 1 and r2.auto_approve


def test_unrecognised_answers_ask_again_and_eof_means_no(env):
    tmp, w = env
    script = [([call("1", "write_file", path="e.txt", content="x")], 1), (["ok"], 1)]
    _, _, out = go(env, script, ["go", "maybe", "perhaps"])                                   # two unclear answers, then end of input
    assert out.count("allow?") == 3 and not (w / "e.txt").exists()


def test_describe_action_variants(env):
    tmp, w = env
    assert describe_action("bash", {"command": "pytest -q"}, w) == ["run: pytest -q"]
    assert describe_action("bash_background", {"command": "npm start"}, w) == ["run in background: npm start"]
    d = describe_action("write_file", {"path": "a.txt", "content": "one\nTHREE\n"}, w)
    assert d[0] == "overwrite a.txt" and "-two" in d and "+THREE" in d
    assert describe_action("write_file", {"path": "new.txt", "content": "x\ny\n"}, w)[:2] == ["create new.txt (2 lines)", "+x"]
    assert describe_action("apply_patch", {"patch": "*** Begin Patch\n*** End Patch"}, w)[0] == "apply patch"
    assert describe_action("mcp__n__x", {"q": 1}, w) == ['mcp__n__x {"q": 1}']
    long = describe_action("write_file", {"path": "big.txt", "content": "\n".join(str(i) for i in range(100))}, w, max_lines=10)
    assert len(long) == 12 and long[-1].startswith("... (90 more lines)")


def test_yes_toggle_skips_prompts(env):
    tmp, w = env
    script = [([call("1", "write_file", path="f.txt", content="x")], 1), (["ok"], 1)]
    r, _, out = go(env, script, ["/yes on", "go"])
    assert (w / "f.txt").exists() and "needs your approval" not in out


# ---------------------------------------------------------------- /diff, /steps, /rewind, /undo
def test_diff_shows_everything_since_the_session_started(env):
    tmp, w = env
    script = [([call("1", "write_file", path="a.txt", content="one\nTWO\n"), call("2", "write_file", path="new.txt", content="n\n")], 1),
              ([call("3", "bash", command="rm -f nothing.txt")], 1), (["ok"], 1)]
    _, _, out = go(env, script, ["/diff", "go", "/diff"], auto=True)
    assert out.count("no changes since the session started") == 1
    assert "--- a/a.txt" in out and "-two" in out and "+TWO" in out and "--- /dev/null" in out and "+++ b/new.txt" in out


def test_diff_marks_deleted_files(env):
    tmp, w = env
    script = [([call("1", "bash", command="rm a.txt")], 1), (["ok"], 1)]
    _, _, out = go(env, script, ["go", "/diff"], auto=True)
    assert "+++ /dev/null" in out and "-one" in out


def test_workspace_diff_ignores_the_run_directory_and_binary_files(env):
    tmp, w = env
    from miniagent import session as sess
    (w / "bin.dat").write_bytes(b"\x00\x01\xff")
    base = tmp / "base"
    sess.snapshot(w, base, exclude=["runs"])
    (w / "bin.dat").write_bytes(b"\x00\x02\xff")
    (w / "runs").mkdir()
    (w / "runs" / "t.jsonl").write_text("x")
    assert workspace_diff(base, w, ["runs"]) == ["Binary file bin.dat differs"]


def test_steps_lists_rewind_points_with_what_happened(env):
    script = [([call("1", "read_file", path="a.txt")], 1), ([call("2", "write_file", path="x.txt", content="x")], 1), (["done"], 1)]
    _, _, out = go(env, script, ["go", "/steps"], auto=True)
    assert "step   1  read_file" in out and "step   2  write_file" in out and "(files changed here)" in out and "step   3  final answer" in out


def test_rewind_restores_files_and_conversation(env):
    tmp, w = env
    script = [([call("1", "write_file", path="a.txt", content="CHANGED\n")], 1), (["done"], 1), (["after rewind"], 1)]
    r, client, out = go(env, script, ["edit", "/rewind 1", "again"], auto=True)
    assert (w / "a.txt").read_text() == "one\ntwo\n" and "rewound to the start of step 1" in out
    last = client.calls[-1]["messages"]
    assert [m["content"] for m in last if m["role"] == "user"] == ["edit", "again"] and not any(m["role"] == "tool" for m in last)
    events = [json.loads(l)["event"] for l in (r.run_dir / "trajectory.jsonl").read_text().splitlines()]
    assert "rewind" in events


def test_rewind_errors_do_not_crash(env):
    _, _, out = go(env, [(["a"], 1)], ["/rewind", "/rewind x", "/rewind 99", "go", "/rewind 99"])
    assert out.count("usage: /rewind N") == 2 and "cannot rewind" in out


def test_undo_takes_back_the_last_turn_only(env):
    tmp, w = env
    script = [([call("1", "write_file", path="a.txt", content="TURN ONE\n")], 1), (["t1 done"], 1),
              ([call("2", "write_file", path="a.txt", content="TURN TWO\n")], 1), (["t2 done"], 1), (["t3"], 1)]
    r, client, out = go(env, script, ["first", "second", "/undo", "third"], auto=True)
    assert (w / "a.txt").read_text() == "TURN ONE\n" and "rewound to the start of step 3" in out
    users = [m["content"] for m in client.calls[-1]["messages"] if m["role"] == "user"]
    assert users == ["first", "third"]
    _, _, out2 = go(env, [], ["/undo"])
    assert "nothing to undo" in out2


# ---------------------------------------------------------------- other commands
def test_cost_status_model_stream_todo(env):
    price = {"input": 1.0, "cache_read": 0.1, "output": 2.0}
    r, client, out = go(env, [(["a"], 1_000_000)], ["go", "/cost", "/status", "/model other-model", "/model", "/stream on", "/stream maybe", "/todo"], price=price,
                        model="m1")
    assert "1000000 input + 7 output tokens in 1 turn(s) ≈ $1.0000" in out
    assert "model m1" in out and "1 turn(s), step 1" in out and "model for the next turns: other-model" in out and "model: other-model" in out
    assert "stream: on" in out and "usage: /stream on|off" in out and "no todo list; start with --features todo" in out
    assert r.agent.model == "other-model" and r.agent.on_text is not None
    _, _, out2 = go(env, [(["a"], 5)], ["go", "/cost"])
    assert "no price known" in out2


def test_todo_command_shows_the_list(env):
    script = [([call("1", "todo_write", todos=[{"content": "fix", "status": "in_progress"}])], 1), (["done"], 1)]
    _, _, out = go(env, script, ["go", "/todo"], features=("todo",))
    assert "[~] fix" in out


def test_clear_starts_a_new_session_directory(env):
    r, client, out = go(env, [(["one"], 1), (["two"], 1)], ["first", "/clear", "second"])
    assert "new conversation" in out and r.run_dir.name == "r2" and r.state["step"] == 1
    assert [m["content"] for m in client.calls[1]["messages"] if m["role"] == "user"] == ["second"]
    assert (env[0] / "runs" / "r1" / "session.json").exists() and (env[0] / "runs" / "r2" / "session.json").exists()


def test_resume_continues_a_saved_session(env):
    tmp, w = env
    r1, _, _ = go(env, [(["first answer"], 1)], ["remember the word banana"])
    from miniagent.session import load_session
    state = load_session(r1.run_dir)
    r2, client, out = start(env, [(["second answer"], 1)], ["what word?"], resume=state, run_dir=r1.run_dir)
    r2.run()
    assert "(resumed)" in out.getvalue() and [m["content"] for m in client.calls[0]["messages"] if m["role"] == "user"] == [
        "remember the word banana", "what word?"]


def test_background_jobs_survive_turns_and_die_with_the_session(env):
    tmp, w = env
    script = [([call("1", "bash_background", command="sleep 30")], 1), (["started"], 1),
              ([call("2", "bash_output", job_id="job1")], 1), (["still here"], 1)]
    r, client, out = start(env, script, ["start a server", "y", "check on it"], features=("background",))
    r.run()
    outputs = [m["content"] for m in client.calls[3]["messages"] if m["role"] == "tool"]
    assert "job1" in outputs[0] and "[job1: running]" in outputs[1]                          # still alive in the next turn
    assert not r.agent.background.get("job1").running                                         # killed when the client exited


# ---------------------------------------------------------------- terminal
def test_colour_only_when_wanted(monkeypatch):
    assert Terminal(inp=io.StringIO(), out=io.StringIO(), color=False).style("x", "31") == "x"
    assert Terminal(inp=io.StringIO(), out=io.StringIO(), color=True).style("x", "31") == "\033[31mx\033[0m"
    monkeypatch.setenv("NO_COLOR", "1")

    class Tty(io.StringIO):
        def isatty(self): return True

    assert Terminal(inp=io.StringIO(), out=Tty()).color is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "xterm")
    assert Terminal(inp=io.StringIO(), out=Tty()).color is True
    assert Terminal(inp=io.StringIO(), out=io.StringIO()).color is False                       # not a tty


def test_colour_diff_lines():
    t = Terminal(inp=io.StringIO(), out=io.StringIO(), color=True)
    assert R.colour_diff_line(t, "+a").startswith("\033[32m") and R.colour_diff_line(t, "-a").startswith("\033[31m")
    assert R.colour_diff_line(t, "@@ -1 +1 @@").startswith("\033[36m") and R.colour_diff_line(t, " ctx") == " ctx"


# ---------------------------------------------------------------- the command line
def test_cli_without_a_task_starts_the_interactive_client(env, monkeypatch, capsys):
    from miniagent import cli
    tmp, w = env
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([(["Hi from the model"], 10)]))
    monkeypatch.setattr("sys.stdin", io.StringIO("hello\n/exit\n"))
    assert cli.main(["-w", str(w), "-m", "fake", "--runs-dir", str(tmp / "runs")]) == 0
    out = capsys.readouterr().out
    assert "Hi from the model" in out and "[done]" in out
    assert (next((tmp / "runs").iterdir()) / "session.json").exists()


def test_cli_interactive_needs_the_session_files(env, capsys):
    from miniagent import cli
    with pytest.raises(SystemExit):
        cli.main(["-w", str(env[1]), "--no-session"])
    assert "interactive client keeps its conversation" in capsys.readouterr().err


def test_cli_interactive_resume_and_yes(env, monkeypatch, capsys):
    from miniagent import cli
    tmp, w = env
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([([call("1", "write_file", path="n.txt", content="x")], 1), (["written"], 1)]))
    monkeypatch.setattr("sys.stdin", io.StringIO("make n.txt\n"))
    assert cli.main(["-w", str(w), "-m", "fake", "-y", "--runs-dir", str(tmp / "runs")]) == 0
    run_dir = next((tmp / "runs").iterdir())
    assert (w / "n.txt").exists()
    s2 = Script([(["resumed ok"], 1)])
    monkeypatch.setattr(cli, "ChatClient", lambda: s2)
    monkeypatch.setattr("sys.stdin", io.StringIO("go on\n"))
    assert cli.main(["-w", str(w), "-m", "fake", "--resume", str(run_dir)]) == 0
    assert "(resumed)" in capsys.readouterr().out and s2.calls[0]["messages"][0]["content"] == "make n.txt"


def test_stuck_detector_events_are_shown(env):
    step = ([call("1", "grep", pattern="two")], 1)
    _, _, out = go(env, [step] * 5, ["go"], stuck_warn_after=2, stuck_stop_after=3)
    assert "stuck detector: repeat ×2 → warn" in out and "stuck detector: repeat ×3 → stop" in out and "[stuck]" in out


def test_diff_skips_files_too_big_to_have_been_copied(env, monkeypatch):
    tmp, w = env
    from miniagent import session as sess
    (w / "huge.txt").write_text("x" * 100)
    monkeypatch.setattr(sess, "MAX_FILE_BYTES", 10)
    base = tmp / "base"
    sess.snapshot(w, base)
    (w / "huge.txt").write_text("y" * 100)
    (w / "a.txt").write_text("one\nCHANGED\n")
    lines = workspace_diff(base, w)
    assert any("+CHANGED" in l for l in lines) and not any("huge" in l for l in lines)


def test_cli_interactive_keeps_background_jobs_between_turns(env, monkeypatch, capsys):
    from miniagent import cli
    tmp, w = env
    seen = []

    class Spy(Script):
        def chat(self, **kw):
            r = super().chat(**kw)
            seen.append([m["content"] for m in kw["messages"] if m["role"] == "tool"])
            return r

    monkeypatch.setattr(cli, "ChatClient", lambda: Spy([([call("1", "bash_background", command="sleep 30")], 1), (["started"], 1),
                                                        ([call("2", "bash_output", job_id="job1")], 1), (["alive"], 1)]))
    monkeypatch.setattr("sys.stdin", io.StringIO("start it\ncheck it\n"))
    assert cli.main(["-w", str(w), "-m", "fake", "-y", "--features", "background", "--runs-dir", str(tmp / "runs")]) == 0
    assert any("[job1: running]" in m for call_msgs in seen for m in call_msgs)
