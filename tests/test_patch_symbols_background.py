"""apply_patch (atomic multi-file edits), the symbols outline tool, and background shell jobs."""
import json
import os
import time

import pytest

from miniagent import patch as P
from miniagent.agent import Agent
from miniagent.background import MAX_JOBS, BackgroundJobs, JobError
from miniagent.client import ModelResponse, ToolCall
from miniagent.permissions import Permissions
from miniagent.symbols import outline
from miniagent.tools import ToolError, Workspace
from miniagent.trajectory import Trajectory


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, **kw):
        import copy
        self.calls.append(copy.deepcopy(kw))
        content, tok = self.script.pop(0)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 5, "cache_hit_tokens": 0})


def patch(*sections): return "*** Begin Patch\n" + "\n".join(sections) + "\n*** End Patch"


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "w"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("def area(r):\n    return 3 * r * r\n\n\ndef perimeter(r):\n    return 2 * r\n")
    (root / "b.txt").write_text("one\ntwo\nthree\n")
    return Workspace(root)


def read(ws, rel): return (ws.root / rel).read_text()


# ---------------------------------------------------------------- apply_patch: parsing and applying
def test_update_add_delete_in_one_patch(ws):
    text = patch("*** Update File: pkg/a.py", " def area(r):", "-    return 3 * r * r", "+    return 3.14159 * r * r",
                 "*** Add File: pkg/new.py", "+X = 1", "+Y = 2",
                 "*** Delete File: b.txt")
    out = ws.apply_patch({"patch": text})
    assert "updated pkg/a.py" in out and "added pkg/new.py" in out and "deleted b.txt" in out
    assert "3.14159" in read(ws, "pkg/a.py") and read(ws, "pkg/new.py") == "X = 1\nY = 2\n" and not (ws.root / "b.txt").exists()
    assert read(ws, "pkg/a.py").endswith("return 2 * r\n")                                    # the rest of the file is untouched


def test_two_hunks_in_one_file_and_anchor(ws):
    text = patch("*** Update File: pkg/a.py", "@@ def area", " def area(r):", "-    return 3 * r * r", "+    return r * r",
                 "@@ def perimeter", " def perimeter(r):", "-    return 2 * r", "+    return 6 * r")
    ws.apply_patch({"patch": text})
    assert read(ws, "pkg/a.py") == "def area(r):\n    return r * r\n\n\ndef perimeter(r):\n    return 6 * r\n"


def test_anchor_disambiguates_a_repeated_block(ws):
    (ws.root / "dup.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 1\n")
    with pytest.raises(ToolError, match="match 2 places"):
        ws.apply_patch({"patch": patch("*** Update File: dup.py", "-    return 1", "+    return 2")})
    ws.apply_patch({"patch": patch("*** Update File: dup.py", "@@ def g", "-    return 1", "+    return 2")})
    assert read(ws, "dup.py") == "def f():\n    return 1\n\n\ndef g():\n    return 2\n"


def test_trailing_whitespace_is_forgiven_but_nothing_else(ws):
    (ws.root / "w.txt").write_text("keep   \nchange me  \n")
    ws.apply_patch({"patch": patch("*** Update File: w.txt", " keep", "-change me", "+changed")})
    assert read(ws, "w.txt") == "keep   \nchanged\n"
    with pytest.raises(ToolError, match="not found"):
        ws.apply_patch({"patch": patch("*** Update File: w.txt", "-  changed", "+x")})                # leading whitespace differs


def test_file_without_trailing_newline_keeps_that(ws):
    (ws.root / "n.txt").write_text("a\nb")
    ws.apply_patch({"patch": patch("*** Update File: n.txt", " a", "-b", "+c")})
    assert read(ws, "n.txt") == "a\nc"


# ---------------------------------------------------------------- apply_patch: all or nothing
def snapshot(ws): return {p.relative_to(ws.root).as_posix(): p.read_text() for p in sorted(ws.root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("bad", [
    patch("*** Update File: pkg/a.py", " def area(r):", "-    return 99", "+    return 0"),                    # hunk does not match
    patch("*** Update File: missing.py", "-x", "+y"),                                                           # no such file
    patch("*** Add File: b.txt", "+x"),                                                                          # already exists
    patch("*** Delete File: nope.txt"),                                                                          # nothing to delete
    patch("*** Update File: ../outside.py", "-x", "+y"),                                                         # escapes the workspace
    patch("*** Add File: /etc/evil", "+x"),
])
def test_a_failing_section_leaves_every_file_untouched(ws, bad):
    good = ["*** Update File: pkg/a.py", " def area(r):", "-    return 3 * r * r", "+    return 0", "*** Add File: ok.txt", "+fine"]
    before = snapshot(ws)
    body = bad.replace("*** Begin Patch\n", "").replace("\n*** End Patch", "")
    with pytest.raises(ToolError):
        ws.apply_patch({"patch": patch(*good, body)})
    assert snapshot(ws) == before


def test_a_write_failure_rolls_back_the_files_already_written(ws, monkeypatch):
    real = type(ws.root).write_text
    calls = []

    def flaky(self, data, *a, **kw):
        calls.append(self.name)
        if self.name == "second.txt":
            raise OSError("disk full")
        return real(self, data, *a, **kw)

    before = snapshot(ws)
    monkeypatch.setattr(type(ws.root), "write_text", flaky)
    text = patch("*** Update File: pkg/a.py", " def area(r):", "-    return 3 * r * r", "+    return 0",
                 "*** Add File: first.txt", "+1", "*** Add File: second.txt", "+2")
    with pytest.raises(ToolError, match="disk full|OSError"):
        ws.apply_patch({"patch": text})
    monkeypatch.undo()
    assert snapshot(ws) == before and "second.txt" in calls


@pytest.mark.parametrize("text,msg", [
    ("", "Begin Patch"), ("*** Begin Patch\n*** Update File: a\n-x", "End Patch"), (patch(), "no files"),
    (patch("junk"), "header"), (patch("*** Add File: a", "no plus"), "must start with"),
    (patch("*** Update File: a", " only context"), "at least one"), (patch("*** Update File: a", "?bad", "-x"), "hunk lines start with"),
    (patch("*** Delete File: a", "body"), "no body"),
    (patch("*** Update File: a", "-x", "+y", "*** Update File: a", "-x", "+y"), "appears twice"),
])
def test_malformed_patches_are_explained(ws, text, msg):
    with pytest.raises(ToolError, match=msg):
        ws.apply_patch({"patch": text})


def test_require_read_applies_to_updates_but_not_to_new_files(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    (root / "a.txt").write_text("x\n")
    ws = Workspace(root, require_read=True)
    with pytest.raises(ToolError, match="not read"):
        ws.apply_patch({"patch": patch("*** Update File: a.txt", "-x", "+y")})
    ws.apply_patch({"patch": patch("*** Add File: n.txt", "+1")})
    ws.apply_patch({"patch": patch("*** Update File: n.txt", "-1", "+2")})                        # written by us: known
    ws.read_file({"path": "a.txt"})
    ws.apply_patch({"patch": patch("*** Update File: a.txt", "-x", "+y")})


def test_diagnostics_warn_about_a_python_file_that_stops_parsing(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    (root / "m.py").write_text("x = 1\n")
    ws = Workspace(root, diagnostics=True)
    out = ws.apply_patch({"patch": patch("*** Update File: m.py", "-x = 1", "+x = (")})
    assert "m.py no longer parses" in out


def test_patch_paths_reads_only_the_headers():
    t = patch("*** Update File: a/b.py", "+*** Delete File: fake.py", "*** Add File: c.txt", "+hello", "*** Delete File: d")
    assert P.patch_paths(t) == ["a/b.py", "c.txt", "d"]
    assert P.patch_paths(None) == [] and P.patch_paths("garbage") == []


# ---------------------------------------------------------------- permissions cannot be dodged through apply_patch
def test_permission_rules_for_file_tools_also_cover_apply_patch():
    perms = Permissions(deny=["write_file(.env*)", "edit_file(**/secrets/**)"], allow=["write_file(tests/**)"])
    d = lambda text: perms.decide("apply_patch", {"patch": text})
    assert d(patch("*** Add File: .env", "+K=1")).action == "deny"
    assert d(patch("*** Update File: a/secrets/k.txt", "-a", "+b")).action == "deny"
    assert d(patch("*** Update File: ok.py", "-a", "+b", "*** Delete File: src/../.env")).action == "deny"      # one bad path is enough
    assert d(patch("*** Add File: tests/t1.py", "+1", "*** Add File: tests/t2.py", "+2")).action == "allow"
    assert d(patch("*** Add File: tests/t1.py", "+1", "*** Add File: src/x.py", "+2")).action == "default"     # allow needs ALL paths
    assert d("not a patch").action == "default"


# ---------------------------------------------------------------- the tool inside the agent
def test_agent_exposes_apply_patch_and_symbols_only_when_asked(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    assert not {"apply_patch", "symbols"} & set(Agent(Script([]), "m", Workspace(root)).tools)
    a = Agent(Script([]), "m", Workspace(root), features=("patch", "symbols"))
    assert {"apply_patch", "symbols"} <= set(a.tools) and a.tools["apply_patch"].needs_confirm and a.tools["symbols"].read_only


def test_apply_patch_through_the_loop_with_a_rejected_patch(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    (root / "a.txt").write_text("x\n")
    bad = patch("*** Update File: a.txt", "-nope", "+y")
    good = patch("*** Update File: a.txt", "-x", "+y")
    ag = Agent(Script([([call("1", "apply_patch", patch=bad)], 10), ([call("2", "apply_patch", patch=good)], 10), (["done"], 10)]), "m",
               Workspace(root), trajectory=Trajectory(tmp_path / "run"), features=("patch",))
    assert ag.run("t").status == "done"
    msgs = [m["content"] for m in ag.client.calls[-1]["messages"] if m["role"] == "tool"]
    assert "error:" in msgs[0] and "not found" in msgs[0] and "updated a.txt" in msgs[1] and (root / "a.txt").read_text() == "y\n"


# ---------------------------------------------------------------- symbols
SRC = '''"""module doc"""
LIMIT = 10
_hidden = 1


class Shape(Base):
    """A shape."""
    def area(self, scale: float = 1.0, *rest, flag=False, **kw) -> float:
        """Compute the area."""

    async def fetch(self): ...


def top(a, b=2, /, c=3):
    def inner(): ...
'''


def test_outline_lists_classes_functions_signatures_and_nesting():
    got = outline(SRC)
    text = "\n".join(f"{n}:{'  ' * d}{t}" for n, d, t in got)
    assert "2:LIMIT = ..." in text and "_hidden" not in text
    assert "6:class Shape(Base)  # A shape." in text
    assert "8:  def area(self, scale: float=1.0, *rest, flag=False, **kw) -> float  # Compute the area." in text
    assert "11:  async def fetch(self)" in text
    assert "14:def top(a, b=2, c=3)" in text and "15:  def inner()" in text


def test_symbols_tool_over_a_tree(ws):
    (ws.root / "pkg" / "broken.py").write_text("def (:\n")
    (ws.root / "node_modules").mkdir()
    (ws.root / "node_modules" / "x.py").write_text("def hidden(): ...\n")
    out = ws.symbols({})
    assert "pkg/a.py:" in out and "def area(r)" in out and "def perimeter(r)" in out
    assert "pkg/broken.py: (could not be parsed)" in out and "hidden" not in out
    assert "def perimeter" not in ws.symbols({"query": "area"}) and "def area" in ws.symbols({"query": "AREA"})
    assert ws.symbols({"path": "pkg/a.py", "query": "zzz"}) == "no definitions match"
    with pytest.raises(ToolError):
        ws.symbols({"path": "nope"})
    with pytest.raises(ToolError, match="escapes"):
        ws.symbols({"path": "../.."})


def test_symbols_does_not_execute_the_code(ws):
    (ws.root / "evil.py").write_text(f"open({str(ws.root / 'pwned')!r}, 'w').write('x')\ndef f(): ...\n")
    ws.symbols({})
    assert not (ws.root / "pwned").exists()


def test_symbols_output_is_capped(ws):
    (ws.root / "big.py").write_text("\n".join(f"def f{i}(): ..." for i in range(500)))
    out = ws.symbols({"path": "big.py"})
    assert "stopped at 200 lines" in out and out.count("def f") <= 200


# ---------------------------------------------------------------- background jobs
@pytest.fixture
def jobs(tmp_path):
    j = BackgroundJobs(tmp_path)
    yield j
    j.kill_all()


def wait_for(cond, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_start_read_incrementally_and_exit_code(jobs):
    j = jobs.start("echo first; sleep 0.3; echo second; exit 3")
    assert wait_for(lambda: "first" in jobs.read(j.id, tail=5))
    assert wait_for(lambda: not j.running)
    out = jobs.read(j.id)
    assert "first" in out and "second" in out and "exited with code 3" in out
    again = jobs.read(j.id)
    assert "first" not in again and "exited with code 3" in again                         # only new output is returned
    assert "second" in jobs.read(j.id, tail=1)


def test_kill_stops_the_whole_process_group(jobs, tmp_path):
    marker = tmp_path / "alive"
    j = jobs.start(f"(while true; do touch {marker}; sleep 0.1; done) & wait")
    assert wait_for(marker.exists)
    jobs.kill(j.id)
    assert not j.running
    marker.unlink()
    time.sleep(0.4)
    assert not marker.exists()                                                           # the grandchild loop is gone too


def test_job_limit_and_unknown_job(jobs):
    for _ in range(MAX_JOBS):
        jobs.start("sleep 30")
    with pytest.raises(JobError, match="already running"):
        jobs.start("sleep 30")
    with pytest.raises(JobError, match="no such job"):
        jobs.read("job99")
    jobs.kill("job1")
    jobs.start("sleep 30")                                                               # a slot freed up


def test_kill_all_leaves_nothing_running(jobs):
    started = [jobs.start("sleep 30") for _ in range(3)]
    jobs.kill_all()
    assert not any(j.running for j in started)


def test_background_tools_in_the_agent_and_cleanup_after_the_run(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    script = [([call("1", "bash_background", command="echo up; sleep 30")], 10)]
    ag = Agent(Script(script + [([call("2", "bash_output", job_id="job1", tail=3)], 10),
                                ([call("3", "bash_background", command="sudo ls")], 10), (["done"], 10)]),
               "m", Workspace(root), features=("background",))
    time.sleep(0)
    assert {"bash_background", "bash_output", "bash_kill"} <= set(ag.tools)
    r = ag.run("t")
    assert r.status == "done"
    msgs = [m["content"] for m in ag.client.calls[-1]["messages"] if m["role"] == "tool"]
    assert "job1" in msgs[0] and "blocked by safety policy" in msgs[2]
    job = ag.background.get("job1")
    assert not job.running                                                                # the run's end killed it


def test_background_is_refused_with_the_sandbox(tmp_path):
    class FakeSandbox:
        pass
    with pytest.raises(ValueError, match="sandbox"):
        Agent(Script([]), "m", Workspace(tmp_path, sandbox=FakeSandbox()), features=("background",))


def test_background_start_asks_for_confirmation(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    asked = []
    ag = Agent(Script([([call("1", "bash_background", command="sleep 5")], 10), (["done"], 10)]), "m", Workspace(root),
               features=("background",), confirm=lambda n, a: asked.append(n) or False)
    ag.run("t")
    assert asked == ["bash_background"] and not ag.background._jobs
    assert os.path.isdir(root)


# ---------------------------------------------------------------- extra cases found by mutation checks
def test_an_exact_match_wins_over_a_whitespace_lookalike(ws):
    (ws.root / "x.txt").write_text("x = 1\nmid\nx = 1   \n")
    ws.apply_patch({"patch": patch("*** Update File: x.txt", "-x = 1", "+x = 2")})                 # one exact match, one loose: take the exact
    assert read(ws, "x.txt") == "x = 2\nmid\nx = 1   \n"


def test_pure_insertion_goes_after_the_anchor_or_to_the_end(ws):
    ws.apply_patch({"patch": patch("*** Update File: pkg/a.py", "@@ def area", "+    # radius squared")})
    assert read(ws, "pkg/a.py").startswith("def area(r):\n    # radius squared\n    return 3")
    ws.apply_patch({"patch": patch("*** Update File: b.txt", "+four")})
    assert read(ws, "b.txt") == "one\ntwo\nthree\nfour\n"


def test_a_deleted_file_is_no_longer_considered_read(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    (root / "a.txt").write_text("x\n")
    ws = Workspace(root, require_read=True)
    ws.read_file({"path": "a.txt"})
    ws.apply_patch({"patch": patch("*** Delete File: a.txt")})
    (root / "a.txt").write_text("something else entirely\n")                                    # recreated behind our back
    with pytest.raises(ToolError, match="not read"):
        ws.apply_patch({"patch": patch("*** Update File: a.txt", "-something else entirely", "+y")})


def test_kill_also_reaps_a_straggler_that_ignores_sigterm(jobs, tmp_path):
    marker = tmp_path / "stubborn"
    j = jobs.start(f"(trap '' TERM; while true; do touch {marker}; sleep 0.1; done) & wait")
    assert wait_for(marker.exists)
    jobs.kill(j.id)
    marker.unlink()
    time.sleep(0.5)
    assert not marker.exists()
