"""Strict loading of evaluation configs.

An experiment config is a plain JSON object. A key the code does not know used to be ignored silently, so a typo such as
`require_reed` produced a run that looked normal and measured nothing. Now every key is checked: unknown keys are an error
(with a "did you mean" suggestion), types and ranges are checked, and the value of `features`, `permissions`, `api_style`,
`disabled_tools` and `system_prompt_file` is validated against what the code accepts.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

from ..features import FEATURES
from ..permissions import Permissions, RuleError
from ..tools import build_tools, Workspace

_NUM = (int, float)
FEATURE_TOOLS = {"glob", "todo_write", "explore", "apply_patch", "symbols", "bash_background", "bash_output", "bash_kill"}


def _spec():
    # key -> (python types, description / extra check name)
    return {
        "name": (str, "required"), "model": (str, "required"), "hypothesis": (str, ""),
        "temperature": ((int, float, type(None)), "range0_2"),
        "max_steps": (int, "pos"), "max_tokens": (int, "pos"), "context_limit": (int, "pos"), "keep_recent": (int, "pos"),
        "compact_regrowth": (_NUM, "nonneg"), "explore_max_steps": (int, "pos"),
        "disabled_tools": (list, "tools"), "features": (list, "features"),
        "log_reasoning": (bool, ""), "sandbox": (bool, ""), "edit_hints": (bool, ""), "require_read": (bool, ""),
        "diagnostics": (bool, ""), "parallel_tools": (bool, ""), "validate_args": (bool, ""),
        "tool_result_json": (bool, ""), "text_tool_calls": (bool, ""), "prompt_cache": (bool, ""),
        "permissions": (dict, "permissions"), "mcp_servers": (list, "mcp"),
        "system_prompt_file": (str, "file"),
        "output_budget": (bool, ""), "stuck_warn_after": (int, "stuck"), "stuck_stop_after": (int, "pos"),
        "max_cost_usd": (_NUM, "pos"), "max_total_tokens": (int, "pos"), "price_tier": (str, "tier"), "prices_file": (str, "file"),
        "api_key_env": (str, ""), "base_url": (str, "url"), "base_url_env": (str, ""), "api_style": (str, "api_style"),
        "api_timeout": (_NUM, "pos"), "api_retries": (int, "nonneg"), "extra_body": (dict, ""),
    }


def validate_config(cfg, *, check_files: bool = True) -> list[str]:
    """All problems with a config, as readable strings; an empty list means it is valid."""
    if not isinstance(cfg, dict):
        return [f"config must be a JSON object, got {type(cfg).__name__}"]
    spec, errs = _spec(), []
    for k in cfg:
        if k not in spec:
            near = difflib.get_close_matches(k, spec, n=1, cutoff=0.6)
            errs.append(f"unknown key {k!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    builtin = set(build_tools(Workspace(".")))
    for k, (types, check) in spec.items():
        if k not in cfg:
            if check == "required":
                errs.append(f"missing required key {k!r}")
            continue
        v = cfg[k]
        bool_ok = types is bool or (isinstance(types, tuple) and bool in types)
        if isinstance(v, bool) and not bool_ok:
            errs.append(f"{k}: expected {_tname(types)}, got a boolean")
            continue
        if not isinstance(v, types):
            errs.append(f"{k}: expected {_tname(types)}, got {type(v).__name__} {v!r}"[:200])
            continue
        if check == "required" and not v.strip():
            errs.append(f"{k}: must not be empty")
        elif check == "pos" and v <= 0:
            errs.append(f"{k}: must be > 0, got {v}")
        elif check == "nonneg" and v < 0:
            errs.append(f"{k}: must be >= 0, got {v}")
        elif check == "range0_2" and v is not None and not 0 <= v <= 2:
            errs.append(f"temperature: must be between 0 and 2, got {v}")
        elif check == "tools":
            if not all(isinstance(t, str) for t in v):
                errs.append("disabled_tools: must be a list of strings")
            else:
                known = builtin | FEATURE_TOOLS
                bad = [t for t in v if t not in known and not t.startswith("mcp__")]
                if bad:
                    errs.append(f"disabled_tools: unknown tools {bad}; known: {sorted(known)}")
        elif check == "features":
            bad = [f for f in v if f not in FEATURES]
            if bad or not all(isinstance(f, str) for f in v):
                errs.append(f"features: unknown {bad}; known: {sorted(FEATURES)}")
        elif check == "permissions":
            try:
                Permissions.from_dict(v)
            except RuleError as e:
                errs.append(f"permissions: {e}")
        elif check == "stuck" and v != 0 and v < 2:
            errs.append(f"stuck_warn_after: 0 (off) or >= 2, got {v}")
        elif check == "tier" and v not in ("off_peak", "peak"):
            errs.append(f"price_tier: must be 'off_peak' or 'peak', got {v!r}")
        elif check == "api_style" and v not in ("openai", "gemini", "anthropic"):
            errs.append(f"api_style: must be 'openai', 'gemini' or 'anthropic', got {v!r}")
        elif check == "url" and not v.startswith(("http://", "https://")):
            errs.append(f"base_url: must start with http:// or https://, got {v!r}")
        elif check == "file" and check_files and not Path(v).is_file():
            errs.append(f"system_prompt_file: no such file {v!r}")
        elif check == "mcp":
            for i, s in enumerate(v):
                if not isinstance(s, dict) or not s.get("allow") or ("command" in s) == ("url" in s):
                    errs.append(f"mcp_servers[{i}]: needs a non-empty 'allow' and exactly one of 'command' / 'url'")
    w, s_ = cfg.get("stuck_warn_after", 0), cfg.get("stuck_stop_after", 6)
    if isinstance(w, int) and isinstance(s_, int) and w >= 2 and s_ <= w:
        errs.append(f"stuck_stop_after ({s_}) must be greater than stuck_warn_after ({w})")
    if cfg.get("base_url") and cfg.get("base_url_env"):
        errs.append("base_url and base_url_env are both set; use one")
    return errs


def _tname(types) -> str:
    ts = types if isinstance(types, tuple) else (types,)
    return "/".join("null" if t is type(None) else t.__name__ for t in ts)


def load_config(path: str | Path, *, check_files: bool = True) -> dict:
    """Read and validate a config file. Raises ValueError listing every problem at once."""
    try:
        cfg = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: {e}")
    errs = validate_config(cfg, check_files=check_files)
    if errs:
        raise ValueError(f"{path}:\n  - " + "\n  - ".join(errs))
    return cfg
