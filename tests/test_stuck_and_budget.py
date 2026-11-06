"""Stuck detection and spending limits."""
import copy
import json

import pytest

from minicode.agent import Agent
from minicode.budget import Budget
from minicode.client import ModelResponse, ToolCall
from minicode.stuck import StuckDetector
from minicode.tools import Workspace
from minicode.trajectory import Trajectory


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    def __init__(self, script, cache=0):
        self.script, self.calls, self.cache = list(script), [], cache

    def chat(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        content, tok = self.script.pop(0)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 10, "cache_hit_tokens": self.cache})


def make(tmp_path, script, **kw):
    ws = tmp_path / "w"                      # the trajectory lives outside the workspace: grep would otherwise find itself in it
    ws.mkdir(exist_ok=True)
    return Agent(Script(script, kw.pop("cache", 0)), "fake", Workspace(ws), trajectory=Trajectory(tmp_path / "run"), **kw)


def events(tmp_path, kind=None):
    ev = [json.loads(l) for l in (tmp_path / "run" / "trajectory.jsonl").read_text().splitlines()]
    return [e for e in ev if kind is None or e["event"] == kind]


def C(name="bash", **inp): return {"id": "x", "name": name, "input": inp or {"command": "pytest"}}
def R(out="same", err=False): return {"output": out, "is_error": err}


# ---------------------------------------------------------------- detector
def test_identical_call_with_identical_result_escalates_from_warn_to_stop():
    d = StuckDetector(3, 5)
    seen = [d.observe(C(), R()) for _ in range(5)]
    assert seen[:2] == [None, None]
    assert [(e.kind, e.count, e.action) for e in seen[2:]] == [("repeat", 3, "warn"), ("repeat", 4, "warn"), ("repeat", 5, "stop")]


def test_same_call_with_a_different_result_is_not_a_repeat():
    d = StuckDetector(3, 6)
    outs = ["1 failed", "1 failed", "2 passed", "2 passed", "2 passed"]
    events_ = [d.observe(C(), R(o)) for o in outs]
    assert events_[:4] == [None] * 4 and events_[4].count == 3 and events_[4].kind == "repeat"    # the edit in between reset the count


def test_a_different_call_resets_the_count():
    d = StuckDetector(3, 6)
    d.observe(C(), R()); d.observe(C(), R())
    assert d.observe(C(command="ls"), R()) is None
    assert d.observe(C(), R()) is None


def test_same_arguments_in_a_different_key_order_are_the_same_call():
    d = StuckDetector(2, 4)
    d.observe(C("grep", pattern="a", path="x"), R())
    assert d.observe(C("grep", path="x", pattern="a"), R()).count == 2


def test_consecutive_errors_of_one_kind_but_different_arguments():
    d = StuckDetector(3, 5)
    got = [d.observe(C("edit_file", old_string=f"v{i}"), R("error: old_string not found (must match exactly)", True)) for i in range(4)]
    assert got[:2] == [None, None] and got[2].kind == "errors" and got[3].count == 4
    assert d.observe(C("edit_file", old_string="ok"), R("edited f.py")) is None            # a success ends the streak
    assert d.observe(C("edit_file", old_string="z"), R("error: old_string not found (must match exactly)", True)) is None


def test_errors_of_different_kinds_do_not_add_up():
    d = StuckDetector(3, 5)
    msgs = ["error: not a file: a", "error: old_string not found", "error: not a file: a", "error: old_string not found"]
    assert [d.observe(C("edit_file", n=i), R(m, True)) for i, m in enumerate(msgs)] == [None] * 4


def test_thresholds_are_validated():
    for bad in ((1, 5), (3, 3), (5, 2)):
        with pytest.raises(ValueError):
            StuckDetector(*bad)


# ---------------------------------------------------------------- in the loop
def test_loop_appends_a_notice_then_stops_with_status_stuck(tmp_path):
    (tmp_path / "w").mkdir(exist_ok=True) or (tmp_path / "w" / "f.txt").write_text("x")
    step = ([call("a", "grep", pattern="x")], 10)
    a = make(tmp_path, [step] * 8, stuck_warn_after=3, stuck_stop_after=5)
    r = a.run("t")
    assert r.status == "stuck" and r.steps == 5 and len(a.client.calls) == 5
    msgs = [m["content"] for m in a.client.calls[-1]["messages"] if m["role"] == "tool"]
    assert "[notice]" not in msgs[1] and "3 times in a row" in msgs[2] and "4 times" in msgs[3]
    ev = events(tmp_path, "stuck")
    assert [(e["count"], e["action"]) for e in ev] == [(3, "warn"), (4, "warn"), (5, "stop")]
    assert events(tmp_path)[-1]["status"] == "stuck"


def test_off_by_default_and_a_working_run_is_untouched(tmp_path):
    step = ([call("a", "grep", pattern="x")], 10)
    assert make(tmp_path, [step] * 4 + [(["done"], 10)]).run("t").status == "done"
    assert not events(tmp_path, "stuck")


def test_edit_between_identical_test_runs_does_not_trigger(tmp_path):
    (tmp_path / "w").mkdir(exist_ok=True) or (tmp_path / "w" / "f.txt").write_text("a")
    script = [([call("1", "bash", command="cat f.txt")], 10),
              ([call("2", "edit_file", path="f.txt", old_string="a", new_string="b")], 10),
              ([call("3", "bash", command="cat f.txt")], 10),
              ([call("4", "edit_file", path="f.txt", old_string="b", new_string="c")], 10),
              ([call("5", "bash", command="cat f.txt")], 10), (["done"], 10)]
    assert make(tmp_path, script, stuck_warn_after=2, stuck_stop_after=3).run("t").status == "done"


def test_classify_knows_the_new_statuses():
    from minicode.eval.grade import classify
    for status, cls in (("stuck", "stuck"), ("budget", "over_budget")):
        assert classify([{"event": "run_end", "status": status}], False, False, []) == cls
    assert classify([{"event": "run_end", "status": "stuck"}], True, False, []) == "pass"


# ---------------------------------------------------------------- budget
PRICE = {"input": 1.0, "cache_read": 0.1, "output": 2.0}


def test_budget_validation():
    for kw in ({"max_cost_usd": 0}, {"max_total_tokens": 0}, {"max_cost_usd": 1.0}, {"max_cost_usd": 1.0, "price": {"input": 1}}):
        with pytest.raises(ValueError):
            Budget(**kw)
    assert Budget(max_total_tokens=5).cost(1, 1, 1) is None
    assert Budget(max_cost_usd=1, price=PRICE).cost(1_000_000, 1_000_000, 500_000) == pytest.approx(1.0 + 0.1 + 1.0)


def test_token_limit_stops_before_the_next_request(tmp_path):
    step = ([call("a", "grep", pattern="x")], 100)                 # 100 in + 10 out per step = 110
    a = make(tmp_path, [step] * 10, budget=Budget(max_total_tokens=300))
    r = a.run("t")
    assert r.status == "budget" and r.steps == 3 and len(a.client.calls) == 3          # 330 >= 300 after step 3: no step 4
    b = events(tmp_path, "budget")[0]
    assert b["step"] == 4 and b["spent_tokens"] == 330 and "token limit" in b["reason"]


def test_cost_limit_uses_cached_and_uncached_prices(tmp_path):
    step = ([call("a", "grep", pattern="x")], 1_000_000)            # 1M input of which 400k cached, 10 output
    a = make(tmp_path, [step] * 5, budget=Budget(max_cost_usd=1.0, price=PRICE), cache=400_000)
    r = a.run("t")
    # one step = 600k*1.0 + 400k*0.1 + 10*2e-6 = 0.64002 USD: after two steps it is 1.28 >= 1.0
    assert r.status == "budget" and r.steps == 2
    end = events(tmp_path, "run_end")[0]
    assert end["cost_usd"] == pytest.approx(1.28004, abs=1e-4)


def test_run_end_reports_cost_even_when_no_limit_is_hit(tmp_path):
    a = make(tmp_path, [(["done"], 1_000_000)], budget=Budget(max_cost_usd=100, price=PRICE))
    assert a.run("t").status == "done"
    assert events(tmp_path, "run_end")[0]["cost_usd"] == pytest.approx(1.00002, abs=1e-4)


def test_subagent_tokens_count_towards_the_limit(tmp_path):
    a = make(tmp_path, [([call("a", "grep", pattern="x")], 100)] * 3, budget=Budget(max_total_tokens=1000))
    a.add_sub_usage(900, 0)                                          # a sub-agent already spent 900
    r = a.run("t")
    assert r.status == "budget" and r.steps == 1


# ---------------------------------------------------------------- configuration plumbing
def test_config_keys_are_validated():
    from minicode.eval.config import validate_config
    ok = {"name": "t", "model": "m"}
    assert validate_config({**ok, "stuck_warn_after": 3, "stuck_stop_after": 6, "max_cost_usd": 0.5, "max_total_tokens": 1000,
                            "price_tier": "peak"}, check_files=False) == []
    for bad in ({"stuck_warn_after": 1}, {"stuck_warn_after": 5, "stuck_stop_after": 5}, {"max_cost_usd": 0}, {"max_total_tokens": -1},
                {"price_tier": "cheap"}, {"stuck_warn_after": "3"}):
        assert validate_config({**ok, **bad}, check_files=False), bad
    assert validate_config({**ok, "stuck_warn_after": 0}, check_files=False) == []        # 0 = off


def test_budget_from_config(tmp_path):
    from minicode.eval.runner import budget_from_config
    assert budget_from_config({"model": "m"}) is None
    assert budget_from_config({"model": "m", "max_total_tokens": 10}).max_total_tokens == 10
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps({"m": {"off_peak": PRICE, "peak": {**PRICE, "input": 2.0}}}))
    b = budget_from_config({"model": "m", "max_cost_usd": 1.0, "prices_file": str(prices)})
    assert b.price == PRICE
    assert budget_from_config({"model": "m", "max_cost_usd": 1.0, "prices_file": str(prices), "price_tier": "peak"}).price["input"] == 2.0
    with pytest.raises(ValueError, match="needs a price"):
        budget_from_config({"model": "unknown", "max_cost_usd": 1.0, "prices_file": str(prices)})
    with pytest.raises(ValueError, match="needs a price"):
        budget_from_config({"model": "m", "max_cost_usd": 1.0, "prices_file": str(tmp_path / "missing.json")})


def test_cli_flags(tmp_path, monkeypatch, capsys):
    from minicode import cli
    (tmp_path / "w").mkdir()
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps({"fake": {"off_peak": PRICE}}))
    step = ([call("a", "grep", pattern="zzz")], 100)
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([step] * 10))
    base = ["go", "-w", str(tmp_path / "w"), "-m", "fake", "--runs-dir", str(tmp_path / "runs"), "-y"]
    assert cli.main(base + ["--stuck-warn", "2", "--stuck-stop", "3"]) == 1                    # status stuck -> exit code 1
    assert "stuck" in capsys.readouterr().out
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([step] * 10))
    assert cli.main(base + ["--max-total-tokens", "250"]) == 1 and "budget" in capsys.readouterr().out
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([step] * 10))
    assert cli.main(base + ["--max-cost", "0.0001", "--prices", str(prices)]) == 1
    assert cli.main(base + ["--max-cost", "1", "--prices", str(tmp_path / "nope.json")]) == 2             # no price table: refuse, not guess
    assert cli.main(["go", "-w", str(tmp_path / "w"), "-m", "other", "--max-cost", "1", "--prices", str(prices),
                     "--runs-dir", str(tmp_path / "runs")]) == 2


def test_two_cli_runs_in_the_same_second_get_separate_run_directories(tmp_path, monkeypatch):
    from minicode import cli
    (tmp_path / "w").mkdir()
    monkeypatch.setattr(cli.time, "strftime", lambda fmt: "20261106-000000")
    for _ in range(2):
        monkeypatch.setattr(cli, "ChatClient", lambda: Script([(["ok"], 10)]))
        assert cli.main(["go", "-w", str(tmp_path / "w"), "-m", "fake", "--runs-dir", str(tmp_path / "runs")]) == 0
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["20261106-000000", "20261106-000000-1"]


def test_limit_is_reached_exactly_means_stop(tmp_path):
    step = ([call("a", "grep", pattern="x")], 100)                 # 110 tokens per step
    r = make(tmp_path, [step] * 10, budget=Budget(max_total_tokens=220)).run("t")
    assert r.status == "budget" and r.steps == 2                   # spent == limit after step 2: no step 3
