"""--git: the agent works on its own branch and commits what it changed (real git in a temporary repository)."""
import json
import subprocess

import pytest

from miniagent.agent import Agent
from miniagent.client import ModelResponse, ToolCall
from miniagent.gitwork import AUTHOR, BRANCH_PREFIX, GitError, GitSession
from miniagent.tools import Workspace
from miniagent.trajectory import Trajectory


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Human")
    git(root, "config", "user.email", "human@example.com")
    git(root, "config", "commit.gpgsign", "false")
    (root / "a.txt").write_text("a\n")
    (root / "b.txt").write_text("b\n")
    (root / ".gitignore").write_text("ignored.log\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    def __init__(self, script):
        self.script = list(script)

    def chat(self, **kw):
        content, tok = self.script.pop(0)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 5, "cache_hit_tokens": 0})


def run_agent(repo, script, tmp_path, **kw):
    gs = GitSession(repo, "run1", exclude=kw.pop("exclude", []))
    info = gs.start()
    a = Agent(Script(script), "m", Workspace(repo), trajectory=Trajectory(tmp_path / "traj"), git=gs, **kw)
    return a, gs, info, a.run("t")


def test_branch_is_cut_and_original_branch_is_untouched(repo, tmp_path):
    base = git(repo, "rev-parse", "main")
    a, gs, info, r = run_agent(repo, [([call("1", "write_file", path="new.txt", content="x")], 10), (["done"], 10)], tmp_path)
    assert r.status == "done" and info == {"branch": "miniagent/run1", "base_branch": "main", "base_sha": base, "reused": False}
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "miniagent/run1"
    assert git(repo, "rev-parse", "main") == base                                           # main did not move
    assert git(repo, "log", "--format=%s", f"{base}..HEAD") == "miniagent: step 1: write_file"
    assert git(repo, "show", "--name-only", "--format=", "HEAD") == "new.txt"


def test_one_commit_per_changing_turn_and_none_for_read_only_turns(repo, tmp_path):
    script = [([call("1", "read_file", path="a.txt")], 10),
              ([call("2", "edit_file", path="a.txt", old_string="a", new_string="A")], 10),
              ([call("3", "bash", command="true")], 10),
              ([call("4", "write_file", path="c.txt", content="c"), call("5", "bash", command="rm b.txt")], 10), (["done"], 10)]
    a, gs, _, _ = run_agent(repo, script, tmp_path)
    msgs = git(repo, "log", "--format=%s", f"{gs.base_sha}..HEAD").splitlines()
    assert msgs == ["miniagent: step 4: bash, write_file", "miniagent: step 2: edit_file"]
    assert git(repo, "show", "--name-status", "--format=", "HEAD").splitlines() == ["D\tb.txt", "A\tc.txt"]    # the deletion is committed too
    assert len(gs.commits) == 2
    ev = [json.loads(l) for l in (tmp_path / "traj" / "trajectory.jsonl").read_text().splitlines() if '"git_commit"' in l]
    assert [e["step"] for e in ev] == [2, 4] and ev[1]["sha"] == gs.commits[1] and set(ev[1]["files"]) == {"b.txt", "c.txt"}


def test_commits_are_authored_by_miniagent_not_by_the_user(repo, tmp_path):
    run_agent(repo, [([call("1", "write_file", path="n.txt", content="x")], 10), (["done"], 10)], tmp_path)
    assert git(repo, "log", "-1", "--format=%an <%ae> / %cn <%ce>") == "miniagent <miniagent@localhost> / miniagent <miniagent@localhost>"
    assert "-c" in AUTHOR


def test_a_users_untracked_files_are_never_committed_but_new_ones_are(repo, tmp_path):
    (repo / "scratch.txt").write_text("my notes\n")                                         # untracked before the run
    script = [([call("1", "write_file", path="agent.txt", content="x"), call("2", "bash", command="echo more >> scratch.txt")], 10), (["done"], 10)]
    _, gs, _, _ = run_agent(repo, script, tmp_path)
    committed = git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert committed == ["agent.txt"] and "scratch.txt" in git(repo, "status", "--porcelain")      # still untracked, edits left alone


def test_ignored_files_are_never_added(repo, tmp_path):
    script = [([call("1", "bash", command="echo log > ignored.log; echo x > kept.txt")], 10), (["done"], 10)]
    run_agent(repo, script, tmp_path)
    assert git(repo, "show", "--name-only", "--format=", "HEAD").splitlines() == ["kept.txt"]


def test_run_directory_inside_the_repository_is_excluded(repo, tmp_path):
    gs = GitSession(repo, "r", exclude=["runs"])
    gs.start()
    (repo / "runs").mkdir()
    (repo / "runs" / "trajectory.jsonl").write_text("{}\n")
    (repo / "real.txt").write_text("x")
    made = gs.commit_step(1, "bash")
    assert made["files"] == ["real.txt"]


def test_dirty_tree_is_refused_and_nothing_changes(repo):
    (repo / "a.txt").write_text("edited by the human\n")
    with pytest.raises(GitError, match="uncommitted changes"):
        GitSession(repo, "r").start()
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main" and (repo / "a.txt").read_text() == "edited by the human\n"
    git(repo, "add", "a.txt")                                                               # staged but uncommitted counts too
    with pytest.raises(GitError, match="uncommitted"):
        GitSession(repo, "r").start()


def test_not_a_repository_and_no_commit_yet(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GitError, match="not inside a git repository"):
        GitSession(plain, "r").start()
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q", "-b", "main")
    with pytest.raises(GitError, match="no commit yet"):
        GitSession(empty, "r").start()


def test_branch_name_collision_gets_a_suffix(repo):
    git(repo, "branch", "miniagent/r")
    gs = GitSession(repo, "r")
    assert gs.start()["branch"] == "miniagent/r-2"


def test_an_agent_branch_is_continued_not_forked(repo, tmp_path):
    a, gs, _, _ = run_agent(repo, [([call("1", "write_file", path="n.txt", content="x")], 10), (["done"], 10)], tmp_path)
    again = GitSession(repo, "resumed")
    info = again.start()
    assert info["reused"] and info["branch"] == "miniagent/run1" and git(repo, "branch", "--list", "miniagent/resumed") == ""


def test_summary_text(repo, tmp_path):
    _, gs, _, _ = run_agent(repo, [([call("1", "write_file", path="n.txt", content="x")], 10), (["done"], 10)], tmp_path)
    s = gs.summary()
    assert "1 commit(s) on miniagent/run1" in s and "1 file changed" in s and "git checkout main" in s
    quiet = GitSession(repo, "q")
    quiet.branch = "miniagent/q"
    assert "no commits" in quiet.summary()


def test_a_git_failure_does_not_end_the_run(repo, tmp_path):
    class Broken(GitSession):
        def commit_step(self, step, summary):
            raise GitError("disk full")

    gs = Broken(repo, "b")
    gs.start()
    a = Agent(Script([([call("1", "write_file", path="n.txt", content="x")], 10), (["done"], 10)]), "m", Workspace(repo),
              trajectory=Trajectory(tmp_path / "traj"), git=gs)
    assert a.run("t").status == "done"
    ev = [json.loads(l) for l in (tmp_path / "traj" / "trajectory.jsonl").read_text().splitlines() if '"git_commit"' in l]
    assert ev[0]["error"] == "disk full" and ev[0]["sha"] == ""


def test_hooks_do_not_run_for_agent_commits(repo, tmp_path):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho ran > " + str(tmp_path / "hook-ran") + "\nexit 1\n")
    hook.chmod(0o755)
    run_agent(repo, [([call("1", "write_file", path="n.txt", content="x")], 10), (["done"], 10)], tmp_path)
    assert not (tmp_path / "hook-ran").exists() and git(repo, "log", "-1", "--format=%s").startswith("miniagent:")


def test_cli_git_flag(repo, tmp_path, monkeypatch, capsys):
    from miniagent import cli
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([([call("1", "write_file", path="cli.txt", content="x")], 10), (["done"], 10)]))
    runs = tmp_path / "runs"
    assert cli.main(["go", "-w", str(repo), "-m", "fake", "-y", "--git", "--runs-dir", str(runs)]) == 0
    out = capsys.readouterr()
    assert "git: 1 commit(s) on miniagent/" in out.out and git(repo, "rev-parse", "--abbrev-ref", "HEAD").startswith(BRANCH_PREFIX)
    (repo / "dirty.txt").write_text("x")
    git(repo, "add", "dirty.txt")
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([(["x"], 10)]))
    assert cli.main(["go", "-w", str(repo), "-m", "fake", "--git", "--runs-dir", str(runs)]) == 2
    assert "uncommitted changes" in capsys.readouterr().err


def test_cli_git_flag_with_the_run_directory_inside_the_repo(repo, tmp_path, monkeypatch, capsys):
    from miniagent import cli
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([([call("1", "write_file", path="x.txt", content="x")], 10), (["done"], 10)]))
    assert cli.main(["go", "-w", str(repo), "-m", "fake", "-y", "--git", "--runs-dir", str(repo / "runs")]) == 0
    files = git(repo, "log", "--name-only", "--format=").split()
    assert "x.txt" in files and not any(f.startswith("runs/") for f in files)                 # the session files are not committed
