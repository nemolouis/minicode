import glob
import json

import pytest

from miniagent.eval.config import load_config, validate_config

OK = {"name": "t", "model": "m"}


def errs(**kw):
    return validate_config({**OK, **kw}, check_files=False)


def test_every_shipped_config_is_valid():
    files = [f for f in glob.glob("configs/*.json") if not f.endswith("prices.json")]
    assert len(files) > 30
    for f in files:
        load_config(f)                                   # raises with every problem listed


def test_minimal_config_is_valid():
    assert validate_config(OK) == []


def test_typo_is_an_error_with_a_suggestion():
    e = errs(require_reed=True)
    assert len(e) == 1 and "unknown key 'require_reed'" in e[0] and "did you mean 'require_read'" in e[0]
    assert "unknown key" in errs(totally_different=1)[0]


def test_missing_and_empty_required_keys():
    assert any("missing required key 'model'" in e for e in validate_config({"name": "x"}))
    assert any("must not be empty" in e for e in errs(name=" "))


@pytest.mark.parametrize("kw", [
    {"max_steps": "50"}, {"max_steps": 0}, {"max_steps": True}, {"temperature": "0"}, {"temperature": 3}, {"require_read": "yes"},
    {"require_read": 1}, {"features": ["todo", "telepathy"]}, {"features": "todo"}, {"disabled_tools": ["bsah"]},
    {"api_style": "claude"}, {"base_url": "api.example.com"}, {"permissions": {"allw": []}}, {"permissions": {"allow": ["bad rule("]}},
    {"api_retries": -1}, {"compact_regrowth": -0.5}, {"extra_body": []}, {"mcp_servers": [{"name": "x"}]},
    {"base_url": "https://a", "base_url_env": "X"},
])
def test_bad_values_are_rejected(kw):
    assert errs(**kw), kw


@pytest.mark.parametrize("kw", [
    {"temperature": None}, {"temperature": 0}, {"temperature": 0.7}, {"features": ["todo", "explore"]}, {"disabled_tools": ["bash", "mcp__x__y"]},
    {"permissions": {"deny": ["bash(curl*)"]}}, {"api_style": "gemini"}, {"extra_body": {"a": {"b": 1}}}, {"sandbox": True},
    {"mcp_servers": [{"name": "n", "command": ["python"], "allow": ["t"]}]},
])
def test_good_values_are_accepted(kw):
    assert errs(**kw) == [], kw


def test_all_problems_are_reported_at_once(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"name": "t", "model": "m", "max_steps": -1, "bogus": 1, "features": ["x"]}))
    with pytest.raises(ValueError) as e:
        load_config(p)
    msg = str(e.value)
    assert "max_steps" in msg and "bogus" in msg and "features" in msg


def test_system_prompt_file_must_exist(tmp_path):
    assert any("no such file" in e for e in validate_config({**OK, "system_prompt_file": str(tmp_path / "nope.txt")}))
    f = tmp_path / "p.txt"
    f.write_text("x")
    assert validate_config({**OK, "system_prompt_file": str(f)}) == []


def test_unreadable_or_non_object_config(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError):
        load_config(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(tmp_path / "list.json")


def test_cli_refuses_a_config_with_a_typo(tmp_path, capsys):
    from miniagent.eval.cli import main
    (tmp_path / "c.json").write_text(json.dumps({"name": "t", "model": "m", "diagnostcs": True}))
    with pytest.raises(SystemExit) as e:
        main(["run", "--config", str(tmp_path / "c.json")])
    assert "diagnostcs" in str(e.value) and "diagnostics" in str(e.value)
