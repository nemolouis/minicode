"""read-before-edit, post-edit diagnostics, permission rules, session resume and rewind (no network)."""
import copy
import json

import pytest

from miniagent import session as sess
from miniagent.agent import Agent
from miniagent.client import ModelResponse, ToolCall
from miniagent.permissions import Permissions, RuleError, parse_rule
from miniagent.tools import ToolError, Workspace
from miniagent.trajectory import Trajectory


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        content, tok = self.script.pop(0)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 5, "cache_hit_tokens": 0})


def make(tmp_path, script, ws_kw=None, run="run", **kw):
    return Agent(Script(script), "fake", Workspace(tmp_path, **(ws_kw or {})), trajectory=Trajectory(tmp_path / run), **kw)


def events(path):
    return [json.loads(l) for l in (path / "trajectory.jsonl").read_text().splitlines()]


def results(a):
    return [m["content"] for m in a.client.calls[-1]["messages"] if m["role"] == "tool"]


# ---------------------------------------------------------------- read before edit
def test_edit_requires_a_prior_read_when_enabled(tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    ws = Workspace(tmp_path, require_read=True)
    with pytest.raises(ToolError, match="not read"):
        ws.edit_file({"path": "f.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert (tmp_path / "f.py").read_text() == "x = 1\n"
    ws.read_file({"path": "f.py"})
    ws.edit_file({"path": "f.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert (tmp_path / "f.py").read_text() == "x = 2\n"


def test_written_files_count_as_known_and_the_guard_is_off_by_default(tmp_path):
    ws = Workspace(tmp_path, require_read=True)
    ws.write_file({"path": "new.txt", "content": "a"})
    ws.edit_file({"path": "new.txt", "old_string": "a", "new_string": "b"})
    (tmp_path / "other.txt").write_text("a")
    Workspace(tmp_path).edit_file({"path": "other.txt", "old_string": "a", "new_string": "b"})     # default: no guard


def test_read_of_another_path_or_a_dotdot_alias_does_not_unlock(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    ws = Workspace(tmp_path, require_read=True)
    ws.read_file({"path": "a.txt"})
    with pytest.raises(ToolError):
        ws.edit_file({"path": "b.txt", "old_string": "b", "new_string": "c"})
    (tmp_path / "d").mkdir()
    ws.edit_file({"path": "d/../a.txt", "old_string": "a", "new_string": "z"})                      # same file, other spelling: fine


# ---------------------------------------------------------------- diagnostics
def test_diagnostics_warn_when_python_or_json_stops_parsing(tmp_path):
    ws = Workspace(tmp_path, diagnostics=True)
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    out = ws.edit_file({"path": "m.py", "old_string": "return 1", "new_string": "return ("})
    assert out.startswith("edited m.py") and "warning: m.py no longer parses" in out and "line 2" in out
    assert "warning" not in ws.edit_file({"path": "m.py", "old_string": "return (", "new_string": "return 2"})
    assert "no longer parses" in ws.write_file({"path": "c.json", "content": "{bad"})
    assert "warning" not in ws.write_file({"path": "notes.txt", "content": "{bad"})                 # other file types: no check
    assert "warning" not in Workspace(tmp_path).write_file({"path": "m2.py", "content": "def ("})  # off by default


def test_diagnostics_never_execute_the_code(tmp_path):
    ws = Workspace(tmp_path, diagnostics=True)
    ws.write_file({"path": "evil.py", "content": f"open({str(tmp_path / 'pwned')!r}, 'w').write('x')\n"})
    assert not (tmp_path / "pwned").exists()


# ---------------------------------------------------------------- permission rules
def decide(rules, name, **args):
    return Permissions.from_dict(rules).decide(name, args)


def test_rule_parsing():
    assert parse_rule("bash(git commit*)").pattern == "git commit*" and parse_rule("write_file").pattern is None
    for bad in ("", "(x)", "a b"):
        with pytest.raises(RuleError):
            parse_rule(bad)
    with pytest.raises(RuleError):
        Permissions.from_dict({"allw": []})
    with pytest.raises(RuleError):
        Permissions.from_dict({"allow": "bash"})
    assert Permissions.from_dict({}) is None


def test_deny_beats_ask_beats_allow_beats_default():
    r = {"allow": ["bash(pytest*)", "bash(curl*)"], "ask": ["bash(pytest -x*)"], "deny": ["bash(curl*)"]}
    assert decide(r, "bash", command="pytest -q").action == "allow"
    assert decide(r, "bash", command="pytest -x").action == "ask"
    assert decide(r, "bash", command="curl http://x").action == "deny"
    assert decide(r, "bash", command="ls").action == "default"
    assert decide(r, "read_file", path="a").action == "default"


def test_bash_compound_commands_do_not_ride_on_an_allow_rule():
    r = {"allow": ["bash(pytest*)", "bash(ls*)"], "deny": ["bash(rm *)"]}
    assert decide(r, "bash", command="pytest -q && ls").action == "allow"      # every segment is covered
    for cmd in ("pytest; echo hi", "pytest | tee out", "pytest\ncat /etc/passwd", "pytest && rm -rf x", "pytest $(id)",
                "pytest `id`", "pytest > ~/.bashrc", "pytest <(id)", "pytest & curl x"):
        assert decide(r, "bash", command=cmd).action != "allow", cmd
    assert decide(r, "bash", command="cd x && rm -rf y").action == "deny"     # a deny fires on any segment
    assert decide(r, "bash", command="ls; rm a").action == "deny"


def test_path_rules_normalise_before_matching():
    r = {"deny": ["write_file(.env*)", "edit_file(**/secrets/**)"], "allow": ["write_file(tests/**)"]}
    for p in (".env", ".env.local", "src/../.env", "./.env"):
        assert decide(r, "write_file", path=p).action == "deny", p
    assert decide(r, "edit_file", path="a/b/secrets/k.txt").action == "deny"
    assert decide(r, "edit_file", path="secrets/k.txt").action == "deny"             # `**/` also matches the top level
    assert decide(r, "write_file", path="tests/a/b.py").action == "allow"
    assert decide(r, "write_file", path="tests/../src/x.py").action == "default"      # does not stay inside tests/
    assert decide(r, "write_file", path="src/.env").action == "default"               # `.env*` has no `/`: top level only
    assert decide({"deny": ["write_file(src/*)"]}, "write_file", path="src/a.py").action == "deny"
    assert decide({"deny": ["write_file(src/*)"]}, "write_file", path="src/a/b.py").action == "default"   # `*` stops at `/`
    assert decide(r, "write_file", content="x").action == "default"                   # no path: a patterned rule never matches


def test_absolute_paths_inside_the_workspace_are_matched_relative(tmp_path):
    p = Permissions.from_dict({"deny": ["write_file(.env*)"]})
    assert p.decide("write_file", {"path": str(tmp_path / ".env")}, tmp_path).action == "deny"
    assert p.decide("write_file", {"path": str(tmp_path / ".env")}).action == "default"   # no root known: cannot relativise


def test_tool_name_globs():
    r = {"allow": ["mcp__notes__*"], "deny": ["*_file(secret*)"]}
    assert decide(r, "mcp__notes__read_note", path="a").action == "allow"
    assert decide(r, "read_file", path="secret.txt").action == "deny"


def test_agent_applies_rules_deny_allow_ask(tmp_path):
    (tmp_path / "f.txt").write_text("hi")
    asked = []

    def confirm(name, args):
        asked.append(name)
        return True

    perms = Permissions(allow=["bash(echo*)"], ask=["read_file(f.txt)"], deny=["write_file(.env*)"])
    a = make(tmp_path, [([call("1", "write_file", path=".env", content="K=1"), call("2", "bash", command="echo ok"),
                          call("3", "read_file", path="f.txt")], 10), (["done"], 10)], confirm=confirm, permissions=perms)
    a.run("t")
    assert not (tmp_path / ".env").exists()
    assert "denied by permission rule 'write_file(.env*)'" in results(a)[0]
    assert "ok" in results(a)[1] and "hi" in results(a)[2]
    assert asked == ["read_file"]                                  # allow skipped the prompt, ask forced one, deny never asked
    ev = [e for e in events(tmp_path / "run") if e["event"] == "tool_result"]
    assert [(e["permission"], e["approved"]) for e in ev] == [("deny", False), ("allow", True), ("ask", True)]


def test_allow_rule_does_not_bypass_the_hard_deny_list(tmp_path):
    a = make(tmp_path, [([call("1", "bash", command="sudo ls")], 10), (["done"], 10)], permissions=Permissions(allow=["bash(sudo*)"]))
    a.run("t")
    assert "blocked by safety policy" in results(a)[0]


def test_an_ask_rule_on_a_read_tool_keeps_it_out_of_the_concurrent_batch(tmp_path):
    (tmp_path / "a").write_text("A")
    (tmp_path / "b").write_text("B")
    order = []
    a = make(tmp_path, [([call("1", "read_file", path="a"), call("2", "read_file", path="b")], 10), (["ok"], 10)],
             permissions=Permissions(ask=["read_file(a)"]), confirm=lambda n, args: order.append(args["path"]) or True)
    a.run("t")
    assert order == ["a"]
    assert not any("parallel" in e for e in events(tmp_path / "run"))


# ---------------------------------------------------------------- sessions: resume
def test_session_is_saved_and_resumed(tmp_path):
    (tmp_path / "f.txt").write_text("one\n")
    a = make(tmp_path, [([call("1", "read_file", path="f.txt")], 100), (["first answer"], 120)], session=True)
    a.run("look at f.txt")
    state = sess.load_session(tmp_path / "run")
    assert state["step"] == 2 and state["messages"][-1]["content"] == "first answer" and state["workspace"] == str(tmp_path.resolve())

    b = make(tmp_path, [(["second answer"], 150)], session=True)
    r = b.run("and now?", resume=state)
    sent = b.client.calls[0]["messages"]
    assert sent[0]["content"] == "look at f.txt" and sent[-1] == {"role": "user", "content": "and now?"}
    assert any(m["role"] == "tool" for m in sent)                  # the earlier tool exchange is part of the context
    assert r.steps == 3 and sess.load_session(tmp_path / "run")["step"] == 3
    ev = events(tmp_path / "run")
    assert [e["event"] for e in ev].count("run_resume") == 1 and [e["step"] for e in ev if e["event"] == "model_call"] == [1, 2, 3]


def test_no_session_files_unless_asked(tmp_path):
    make(tmp_path, [(["x"], 10)]).run("t")
    assert not (tmp_path / "run" / "session.json").exists()


def test_load_session_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        sess.load_session(tmp_path)


# ---------------------------------------------------------------- sessions: rewind
def project(tmp_path):
    w = tmp_path / "work"
    w.mkdir()
    (w / "a.txt").write_text("A0\n")
    (w / "keep.txt").write_text("K\n")
    return w


def test_rewind_restores_files_conversation_and_todos(tmp_path):
    w = project(tmp_path)
    script = [([call("1", "read_file", path="a.txt")], 10),
              ([call("2", "edit_file", path="a.txt", old_string="A0", new_string="A1"), call("t", "todo_write", todos=[{"content": "x", "status": "pending"}])], 10),
              ([call("3", "write_file", path="new.txt", content="N"), call("4", "bash", command="rm keep.txt")], 10),
              (["done"], 10)]
    a = make(w, script, run="../run", session=True, features=("todo",))
    a.run("task")
    assert (w / "a.txt").read_text() == "A1\n" and (w / "new.txt").exists() and not (w / "keep.txt").exists()
    assert sess.checkpoint_steps(tmp_path / "run") == [2, 3]       # step 1 only read: no checkpoint

    info = sess.rewind(tmp_path / "run", 3, w)                     # back to the start of step 3
    assert (w / "a.txt").read_text() == "A1\n" and not (w / "new.txt").exists() and (w / "keep.txt").read_text() == "K\n"
    assert info["removed"] == 1
    st = sess.load_session(tmp_path / "run")
    assert st["step"] == 2 and st["messages"][-1]["role"] == "tool" and st["todos"] == [{"content": "x", "status": "pending"}]
    assert sorted(st["step_index"]) == [1, 2] and sess.checkpoint_steps(tmp_path / "run") == [2]

    sess.rewind(tmp_path / "run", 1, w)                            # all the way back
    assert (w / "a.txt").read_text() == "A0\n" and st["messages"] and sess.load_session(tmp_path / "run")["messages"] == [
        {"role": "user", "content": "task"}]
    assert sess.load_session(tmp_path / "run")["todos"] == []


def test_resume_after_rewind_continues_from_the_cut(tmp_path):
    w = project(tmp_path)
    a = make(w, [([call("1", "write_file", path="a.txt", content="BAD")], 10), (["done"], 10)], run="../run", session=True)
    a.run("task")
    sess.rewind(tmp_path / "run", 1, w)
    assert (w / "a.txt").read_text() == "A0\n"
    b = make(w, [(["ok"], 10)], run="../run", session=True)
    b.run("try something else", resume=sess.load_session(tmp_path / "run"))
    assert [m["content"] for m in b.client.calls[0]["messages"] if m["role"] == "user"] == ["task", "try something else"]
    assert not any(m["role"] == "tool" for m in b.client.calls[0]["messages"])         # the bad attempt is gone from the context


def test_rewind_checks_before_touching_anything(tmp_path):
    w = project(tmp_path)
    make(w, [([call("1", "write_file", path="a.txt", content="X")], 10), (["done"], 10)], run="../run", session=True).run("t")
    with pytest.raises(ValueError, match="rewindable steps"):
        sess.rewind(tmp_path / "run", 9, w)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="ran in"):
        sess.rewind(tmp_path / "run", 1, other)
    assert (w / "a.txt").read_text() == "X"


def test_rewind_cannot_cross_a_compaction(tmp_path):
    w = project(tmp_path)
    (w / "big.txt").write_text("y" * 4000)
    steps = [([call(str(i), "read_file", path="big.txt")], 100) for i in range(3)]
    steps += [(["SUMMARY"], 10), ([call("9", "read_file", path="big.txt")], 100), (["done"], 10)]
    steps[2] = ([call("2", "read_file", path="big.txt")], 90_000)         # an input count that triggers compaction at step 4
    a = make(w, steps, run="../run", session=True, context_limit=1000, keep_recent=2)
    a.run("t")
    assert 1 not in sess.load_session(tmp_path / "run")["step_index"]
    with pytest.raises(ValueError, match="latest compaction"):
        sess.rewind(tmp_path / "run", 1, w)


def test_run_directory_inside_the_workspace_is_never_snapshotted_or_deleted(tmp_path):
    w = project(tmp_path)
    run = w / "runs" / "r1"
    (w / "runs" / "older").mkdir(parents=True)
    (w / "runs" / "older" / "big.log").write_text("old")
    a = make(w, [([call("1", "write_file", path="a.txt", content="X")], 10), (["done"], 10)], run="runs/r1", session=True)
    a.run("t")
    snap = run / "checkpoints" / "step-1"
    assert not any("runs" in p.relative_to(snap / "files").parts for p in (snap / "files").rglob("*"))
    assert (snap / "files" / "a.txt").exists()
    sess.rewind(run, 1, w)
    assert (w / "a.txt").read_text() == "A0\n" and (run / "session.json").exists() and (w / "runs" / "older" / "big.log").exists()


def test_snapshot_limits_skip_large_files_and_refuse_incomplete_restores(tmp_path, monkeypatch):
    w = project(tmp_path)
    (w / "large.bin").write_text("z" * 50)
    monkeypatch.setattr(sess, "MAX_FILE_BYTES", 10)
    info = sess.snapshot(w, tmp_path / "snap")
    assert info["skipped_large"] == 1 and info["complete"]
    (w / "large.bin").write_text("changed")
    sess.restore(w, tmp_path / "snap")
    assert (w / "large.bin").read_text() == "changed"              # skipped files are neither restored nor deleted
    monkeypatch.setattr(sess, "MAX_FILE_BYTES", 1000)
    monkeypatch.setattr(sess, "MAX_SNAPSHOT_BYTES", 3)
    assert not sess.snapshot(w, tmp_path / "snap2")["complete"]


def test_cli_resume_and_rewind(tmp_path, monkeypatch):
    from miniagent import cli
    w = project(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([([call("1", "write_file", path="a.txt", content="X")], 10), (["done"], 10)]))
    assert cli.main(["edit it", "-w", str(w), "-y", "-m", "fake", "--runs-dir", str(runs)]) == 0
    run_dir = next(runs.iterdir())
    assert (w / "a.txt").read_text() == "X"
    assert cli.main(["--rewind", str(run_dir), "--to-step", "1", "-w", str(w)]) == 0
    assert (w / "a.txt").read_text() == "A0\n"
    s2 = Script([(["hello again"], 10)])
    monkeypatch.setattr(cli, "ChatClient", lambda: s2)
    assert cli.main(["next", "-w", str(w), "-y", "-m", "fake", "--resume", str(run_dir)]) == 0
    assert s2.calls[0]["messages"][-1]["content"] == "next"
    assert cli.main(["next", "-w", str(tmp_path), "-m", "fake", "--resume", str(run_dir)]) == 2      # wrong workspace
    assert cli.main(["--rewind", str(run_dir), "--to-step", "99", "-w", str(w)]) == 2                  # unreachable step: error, no crash


def test_rewind_keeps_preexisting_empty_dirs_and_removes_new_ones(tmp_path):
    w = project(tmp_path)
    (w / "empty_pkg").mkdir()
    script = [([call("1", "bash", command="mkdir -p made/inner && echo x > made/inner/f.txt")], 10), (["done"], 10)]
    make(w, script, run="../run", session=True).run("t")
    assert (w / "made" / "inner" / "f.txt").exists()
    sess.rewind(tmp_path / "run", 1, w)
    assert (w / "empty_pkg").is_dir() and not (w / "made").exists()


def test_rewind_refuses_an_incomplete_checkpoint(tmp_path, monkeypatch):
    w = project(tmp_path)
    monkeypatch.setattr(sess, "MAX_SNAPSHOT_BYTES", 1)
    make(w, [([call("1", "write_file", path="a.txt", content="X")], 10), (["done"], 10)], run="../run", session=True).run("t")
    with pytest.raises(ValueError, match="incomplete"):
        sess.rewind(tmp_path / "run", 1, w)
    assert (w / "a.txt").read_text() == "X"
