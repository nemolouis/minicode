"""What a "hi" must not cost: read-only commands allowed without asking in the interactive clients, the conversational prompt, and the
repair of HTML-escaped shell commands."""
import json

import pytest

from miniagent.agent import INTERACTIVE_SYSTEM_PROMPT, SYSTEM_PROMPT, Agent
from miniagent.client import ModelResponse, ToolCall
from miniagent.permissions import READ_ONLY_DEFAULTS, Permissions
from miniagent.tools import Workspace
from miniagent.trajectory import Trajectory

P = Permissions.merge(READ_ONLY_DEFAULTS)


def action(cmd): return P.decide("bash", {"command": cmd}).action


# ---------------------------------------------------------------- the defaults
@pytest.mark.parametrize("cmd", [
    "ls", "ls -la", "pwd", "cat pyproject.toml", "head -40 README.md", "wc -l src/a.py", "grep -rn 'x' src | wc -l", "rg foo", "tree -L 2",
    "git status", "git log --oneline -5", "git diff", "git diff HEAD~1 -- src", "git show HEAD", "git branch", "git branch -a", "git rev-parse HEAD",
    "ls -la && git log --oneline -5 2>/dev/null && git status 2>/dev/null | head -30",              # the command from the bug report
    "find . -maxdepth 2 -not -path './.git*' -not -path './node_modules*' | head -60",
    "git log --oneline -5; echo \"---\"; git status | head -30",
    "ls tests tests/misc tests/core; echo \"---\"; cat pyproject.toml | head -40",
    "git log 2>&1 | head", "grep -r x . > /dev/null", "cd src && ls",
])
def test_read_only_commands_are_allowed_without_asking(cmd):
    assert action(cmd) == "allow", cmd


@pytest.mark.parametrize("cmd,expected", [
    ("cat ~/.ssh/id_rsa", "ask"), ("ls /etc", "ask"), ("cat ../secrets.txt", "ask"), ("cat $HOME/.netrc", "ask"), ("ls ..", "ask"),
    ("find . -name '*.pyc' -delete", "deny"), ("find . -exec rm {} +", "deny"), ("find . -fprint out.txt", "deny"),
    ("git diff --output=x.patch", "deny"), ("git diff --ext-diff", "deny"), ("rg --pre ./evil foo", "deny"), ("git -c core.pager=evil log", "deny"),
    ("git branch -D main", "default"), ("git branch newbranch", "default"), ("git branch --delete x", "default"),
    ("echo hi > out.txt", "default"), ("cat a >> b", "default"), ("pytest -q", "default"), ("rm -rf build", "default"),
    ("ls; rm x", "default"), ("cat a.py | python -", "default"), ("ls $(whoami)", "default"), ("ls `id`", "default"), ("git push", "default"),
    ("git commit -am x", "default"), ("lsof -i", "default"), ("catalog x", "default"),
])
def test_everything_else_still_asks_or_is_refused(cmd, expected):
    assert action(cmd) == expected, cmd


def test_a_users_own_rules_win_over_the_defaults():
    mine = Permissions(deny=["bash(git log*)"], ask=["bash(ls*)"])
    merged = Permissions.merge(mine, READ_ONLY_DEFAULTS)
    assert merged.decide("bash", {"command": "git log"}).action == "deny"
    assert merged.decide("bash", {"command": "ls"}).action == "ask"
    assert merged.decide("bash", {"command": "cat x"}).action == "allow"


def test_merge_edge_cases():
    assert Permissions.merge(None, None) is None and Permissions.merge({}, Permissions()) is None
    a, b = Permissions(allow=["bash(a)"]), {"allow": ["bash(b)"], "deny": ["bash(c)"]}
    assert Permissions.merge(a, b).to_dict() == {"allow": ["bash(a)", "bash(b)"], "ask": [], "deny": ["bash(c)"]}


def test_harmless_redirects_do_not_matter_but_real_ones_do():
    assert action("cat a 2>/dev/null") == "allow" and action("cat a 2> /dev/null") == "allow" and action("cat a > /dev/null 2>&1") == "allow"
    assert action("cat a > b.txt") == "default" and action("cat a 2>err.log") == "default"


# ---------------------------------------------------------------- the interactive prompt and wiring
def test_the_interactive_prompt_is_conversational_and_the_default_is_not_changed():
    p = INTERACTIVE_SYSTEM_PROMPT
    assert "greetings" in p and "WITHOUT using any tools" in p and "Never explore" in p and "clarifying question" in p and "HTML entities" in p
    assert "Your job is to fix the bug" in SYSTEM_PROMPT and "greetings" not in SYSTEM_PROMPT


class Script:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, **kw):
        import copy
        self.calls.append(copy.deepcopy(kw))
        content, tok = self.script.pop(0)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 1})


def test_cli_interactive_uses_the_conversational_prompt_and_the_read_only_defaults(tmp_path, monkeypatch, capsys):
    import io

    from miniagent import cli
    w = tmp_path / "w"
    w.mkdir()
    (w / "a.txt").write_text("hello\n")
    s = Script([([ToolCall("1", "bash", json.dumps({"command": "cat a.txt"}))], 1), (["It says hello."], 1)])
    monkeypatch.setattr(cli, "ChatClient", lambda: s)
    monkeypatch.setattr("sys.stdin", io.StringIO("what is in a.txt?\n"))          # no -y: without the defaults this would ask and get end-of-input = no
    assert cli.main(["-w", str(w), "-m", "fake", "--ui", "repl", "--runs-dir", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "needs your approval" not in out and "It says hello." in out
    assert s.calls[0]["system"] == INTERACTIVE_SYSTEM_PROMPT


def test_cli_ask_everything_brings_the_prompts_back(tmp_path, monkeypatch, capsys):
    import io

    from miniagent import cli
    w = tmp_path / "w"
    w.mkdir()
    (w / "a.txt").write_text("hello\n")
    s = Script([([ToolCall("1", "bash", json.dumps({"command": "cat a.txt"}))], 1), (["denied, ok."], 1)])
    monkeypatch.setattr(cli, "ChatClient", lambda: s)
    monkeypatch.setattr("sys.stdin", io.StringIO("read it\n"))
    assert cli.main(["-w", str(w), "-m", "fake", "--ui", "repl", "--ask-everything", "--runs-dir", str(tmp_path / "runs")]) == 0
    assert "needs your approval" in capsys.readouterr().out


def test_one_shot_runs_keep_the_task_prompt_and_get_no_default_allow_rules(tmp_path, monkeypatch):
    from miniagent import cli
    w = tmp_path / "w"
    w.mkdir()
    s = Script([(["done"], 1)])
    monkeypatch.setattr(cli, "ChatClient", lambda: s)
    assert cli.main(["a task", "-w", str(w), "-m", "fake", "--runs-dir", str(tmp_path / "runs")]) == 0
    assert s.calls[0]["system"] == SYSTEM_PROMPT
    start = json.loads(next((tmp_path / "runs").iterdir()).joinpath("trajectory.jsonl").read_text().splitlines()[0])
    assert "permissions" not in start["config"]                                                    # the read-only defaults are for the interactive clients only


# ---------------------------------------------------------------- HTML-escaped commands
def test_html_escaped_commands_are_repaired_only_when_that_makes_them_valid(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "page.html").write_text("a &amp;&amp; b\n")
    broken = "echo one &amp;&amp; echo two 2&gt;/dev/null"
    out = ws.bash({"command": broken})
    assert out.startswith("exit code: 0") and "one" in out and "two" in out                       # decoded and run
    legit = "grep -c '&amp;&amp;' page.html"
    assert ws.bash({"command": legit}).splitlines()[1] == "1"                                      # valid as written: left alone
    assert Workspace._repair_html_entities("echo plain") == "echo plain"
    assert Workspace._repair_html_entities("echo &amp;") == "echo &amp;"                           # not valid either way: untouched
    still_broken = "echo &amp;&amp; &"                                                             # decoding does not help
    assert Workspace._repair_html_entities(still_broken) == still_broken


def test_the_agent_runs_the_repaired_command_and_logs_what_was_asked(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    s = Script([([ToolCall("1", "bash", json.dumps({"command": "echo a &amp;&amp; echo b"}))], 1), (["ok"], 1)])
    ag = Agent(s, "m", Workspace(w), trajectory=Trajectory(tmp_path / "run"))
    assert ag.run("t").status == "done"
    result = [m["content"] for m in s.calls[1]["messages"] if m["role"] == "tool"][0]
    assert result.startswith("exit code: 0") and "a" in result and "b" in result
