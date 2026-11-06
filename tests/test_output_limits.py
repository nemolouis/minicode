"""Tool output sizes: line-aware read_file continuation, paged grep, truncation hints, and the context-aware output budget."""
import json

import pytest

from miniagent.agent import Agent
from miniagent.client import ModelResponse, ToolCall
from miniagent.tools import MAX_OUTPUT_CHARS, ToolError, Workspace, output_limit_for, truncate
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


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    return Workspace(root)


# ---------------------------------------------------------------- read_file
def test_read_file_cut_by_size_says_where_to_continue(ws):
    (ws.root / "big.txt").write_text("\n".join(f"line {i} " + "x" * 90 for i in range(1, 501)) + "\n")
    ws.output_limit = 5000
    out = ws.read_file({"path": "big.txt"})
    shown = [l for l in out.splitlines() if "\tline " in l]
    last = int(shown[-1].split("\t")[0])
    assert 30 < last < 60 and len(out) < 5400                                                  # whole lines, about 5000 chars
    assert f"{500 - last} more lines after line {last}" in out and f"offset={last + 1}" in out and "output size limit" in out
    nxt = ws.read_file({"path": "big.txt", "offset": last + 1})
    assert nxt.splitlines()[0].split("\t")[0].strip() == str(last + 1)                          # nothing skipped, nothing repeated


def test_read_file_cut_by_the_line_limit(ws):
    (ws.root / "f.txt").write_text("\n".join(str(i) for i in range(1, 11)) + "\n")
    out = ws.read_file({"path": "f.txt", "limit": 4})
    assert out.endswith("(6 more lines after line 4, cut by the 4-line limit: call read_file with offset=5)")
    assert "more lines" not in ws.read_file({"path": "f.txt", "limit": 10})


def test_a_single_huge_line_is_still_shown(ws):
    (ws.root / "one.txt").write_text("y" * 50_000 + "\nsecond\n")
    out = ws.read_file({"path": "one.txt"})
    assert out.startswith("     1\tyyy") and "truncated" in out and "offset=2" in out


def test_small_files_are_unchanged(ws):
    (ws.root / "s.txt").write_text("a\nb\n")
    assert ws.read_file({"path": "s.txt"}) == "     1\ta\n     2\tb"


# ---------------------------------------------------------------- grep paging
def test_grep_pages_through_a_long_result(ws):
    (ws.root / "m.txt").write_text("\n".join(f"hit {i}" for i in range(500)) + "\n")
    p1 = ws.grep({"pattern": "hit"})
    assert p1.count("m.txt:") == 200 and "showing matches 1-200" in p1 and "offset=200" in p1
    p2 = ws.grep({"pattern": "hit", "offset": 200})
    assert p2.startswith("m.txt:201:hit 200") and "showing matches 201-400" in p2
    p3 = ws.grep({"pattern": "hit", "offset": 400})
    assert p3.count("m.txt:") == 100 and "more exist" not in p3 and p3.splitlines()[-1] == "m.txt:500:hit 499"
    assert ws.grep({"pattern": "hit", "offset": 500}) == "no matches after the first 500"
    assert ws.grep({"pattern": "hit", "max_results": 3}).count("m.txt:") == 3
    assert ws.grep({"pattern": "hit", "max_results": 10_000}).count("m.txt:") == 200            # the cap cannot be raised
    assert ws.grep({"pattern": "nothing"}) == "no matches"
    with pytest.raises(ToolError, match="integers"):
        ws.grep({"pattern": "x", "offset": "many"})


def test_grep_exactly_at_the_cap_has_no_false_more_marker(ws):
    (ws.root / "e.txt").write_text("\n".join(f"hit {i}" for i in range(200)) + "\n")
    assert "more exist" not in ws.grep({"pattern": "hit"})


# ---------------------------------------------------------------- bash / truncate
def test_truncate_names_what_was_cut_and_how_to_get_it():
    out = truncate("a" * 5000 + "\n" + "b" * 5000, 1000, "use head")
    assert "truncated 9001 of 10001 chars" in out and "2 lines in all" in out and "use head" in out
    assert out.startswith("a" * 500) and out.endswith("b" * 500) and truncate("short", 1000) == "short"


def test_bash_output_uses_the_limit_and_explains_how_to_see_more(ws):
    ws.output_limit = 3000
    out = ws.bash({"command": "python3 -c \"print('z' * 20000)\""})
    assert "truncated" in out and "| head" in out and len(out) < 3400


# ---------------------------------------------------------------- the dynamic budget
def test_output_limit_scales_with_the_room_left():
    assert output_limit_for(100_000, 0) == MAX_OUTPUT_CHARS                                    # fresh context: the full limit
    assert output_limit_for(100_000, 60_000) == 15_000                                         # 10000 tokens of headroom -> half in chars
    assert output_limit_for(100_000, 64_000) == 9000                                           # 6000 tokens -> 9000 chars
    assert output_limit_for(100_000, 69_000) == 2000                                           # almost full: the floor
    assert output_limit_for(100_000, 90_000) == 2000 and output_limit_for(100_000, 500_000) == 2000
    assert output_limit_for(10_000, 0) == 10_500 and output_limit_for(5_000, 0) == 5_250


def test_agent_lowers_the_limit_as_the_input_grows_and_not_when_off(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    (root / "f.txt").write_text("\n".join("z" * 100 for _ in range(400)) + "\n")
    script = [([call("1", "read_file", path="f.txt")], 1000), ([call("2", "read_file", path="f.txt")], 68_000), (["done"], 69_000)]
    on = Agent(Script(script), "m", Workspace(root), trajectory=Trajectory(tmp_path / "run1"), output_budget=True, context_limit=100_000)
    on.run("t")
    outs = [m["content"] for m in on.client.calls[-1]["messages"] if m["role"] == "tool"]
    assert len(outs[0]) > 15_000 and len(outs[1]) < 3300 and "offset=" in outs[1]               # second read: 3000 chars, not 20000
    off = Agent(Script(script), "m", Workspace(root), trajectory=Trajectory(tmp_path / "run2"), context_limit=100_000)
    off.run("t")
    outs = [m["content"] for m in off.client.calls[-1]["messages"] if m["role"] == "tool"]
    assert len(outs[1]) > 15_000


def test_config_and_cli_know_output_budget(tmp_path):
    from miniagent.eval.config import validate_config
    assert validate_config({"name": "t", "model": "m", "output_budget": True}, check_files=False) == []
    assert validate_config({"name": "t", "model": "m", "output_budget": "yes"}, check_files=False)
