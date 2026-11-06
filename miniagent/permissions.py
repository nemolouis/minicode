"""Configurable permission rules: which tool calls are allowed without asking, always asked, or refused.

    {"deny":  ["bash(curl*)", "write_file(.env*)", "edit_file(**/secrets/**)"],
     "ask":   ["bash(git commit*)"],
     "allow": ["bash(pytest*)", "bash(python -m pytest*)", "write_file(tests/**)", "mcp__notes__*"]}

A rule is `tool` or `tool(pattern)`. The tool part is a glob over the tool name. The pattern is matched against the call's
*subject*: the command for `bash`, the path for the file tools, nothing for other tools (a rule with a pattern on them never
matches). `*` matches anything except `/` in paths; `**` matches across `/`; in commands `*` matches anything.

Decision order: deny > ask > allow > the tool's own default (writes and shell ask, reads do not). So `--yes` auto-approves what
would be asked, but never overrides a deny rule, and the hard deny-list in safety.py still applies after an allow.

Shell commands are split on `;`, `&`, `|` and newlines before matching. A deny rule fires if ANY segment matches it. An allow rule
only applies if EVERY segment matches some allow rule and the command has no command substitution (`$(`, backtick), process
substitution or redirection: `pytest; rm -rf x` and `pytest > ~/.bashrc` are not covered by `bash(pytest*)`. They fall back to
asking. This is conservative on purpose; it is a convenience filter, not a sandbox.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SEPARATORS = re.compile(r"&&|\|\||[;&|\n]")
_UNSAFE_FOR_ALLOW = re.compile(r"\$\(|`|<\(|>\(|[<>]")
_RULE = re.compile(r"^\s*([^()\s]+)\s*(?:\((.*)\))?\s*$", re.S)


class RuleError(ValueError):
    pass


def _glob_to_regex(pat: str, path: bool) -> re.Pattern:
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i:i + 2] == "**":
                out.append(".*")
                i += 2
                if path and pat[i:i + 1] == "/":          # `**/x` also matches `x` at the top
                    out[-1] = "(?:.*/)?"
                    i += 1
                continue
            out.append("[^/]*" if path else ".*")
        elif c == "?":
            out.append("[^/]" if path else ".")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z", re.S)


@dataclass(frozen=True)
class Rule:
    text: str
    tool: re.Pattern
    pattern: str | None

    def matches_tool(self, name: str) -> bool:
        return bool(self.tool.match(name))

    def matches_subject(self, name: str, subject: str | None) -> bool:
        if not self.matches_tool(name):
            return False
        if self.pattern is None:
            return True
        if subject is None:
            return False
        return bool(_glob_to_regex(self.pattern, path=name != "bash").match(subject))


def parse_rule(text: str) -> Rule:
    m = _RULE.match(text or "")
    if not m:
        raise RuleError(f"bad permission rule {text!r}: expected `tool` or `tool(pattern)`")
    return Rule(text.strip(), _glob_to_regex(m.group(1), path=False), m.group(2))


# apply_patch writes files, so the rules written for the single-file tools apply to it as well (a deny on `write_file(.env*)` must
# not be dodged by putting the same change in a patch)
_ALIASES = {"apply_patch": ("apply_patch", "edit_file", "write_file")}


def _subject(name: str, args: dict) -> str | None:
    if name == "bash":
        v = args.get("command")
    else:
        v = args.get("path")
    return v if isinstance(v, str) else None


def _norm_path(p: str) -> str:
    parts: list[str] = []
    for seg in p.replace("\\", "/").split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            else:
                parts.append("..")
        else:
            parts.append(seg)
    return "/".join(parts)


@dataclass
class Decision:
    action: str            # "deny" | "ask" | "allow" | "default"
    rule: str | None = None


class Permissions:
    def __init__(self, allow=(), ask=(), deny=()):
        self.allow = [parse_rule(r) for r in allow]
        self.ask = [parse_rule(r) for r in ask]
        self.deny = [parse_rule(r) for r in deny]

    @classmethod
    def from_dict(cls, d: dict | None) -> "Permissions | None":
        if not d:
            return None
        unknown = set(d) - {"allow", "ask", "deny"}
        if unknown:
            raise RuleError(f"unknown permission keys {sorted(unknown)}; expected allow / ask / deny")
        for k in d:
            if not isinstance(d[k], list) or not all(isinstance(x, str) for x in d[k]):
                raise RuleError(f"permissions.{k} must be a list of strings")
        return cls(d.get("allow", ()), d.get("ask", ()), d.get("deny", ()))

    def to_dict(self) -> dict:
        return {k: [r.text for r in getattr(self, k)] for k in ("allow", "ask", "deny")}

    def _decide_patch(self, args: dict, root) -> Decision:
        """A patch touches several paths: deny/ask fire if ANY path matches a rule (of apply_patch, edit_file or write_file); allow
        needs EVERY path to be covered by an allow rule."""
        from .patch import patch_paths
        paths = []
        for raw in patch_paths(args.get("patch")):
            if root is not None and raw.startswith("/"):
                try:
                    raw = str(Path(raw).relative_to(root))
                except ValueError:
                    pass
            paths.append(_norm_path(raw))
        if not paths:
            return Decision("default")
        names = _ALIASES["apply_patch"]
        for rules, action in ((self.deny, "deny"), (self.ask, "ask")):
            for r in rules:
                if any(r.matches_subject(n, p) for p in paths for n in names):
                    return Decision(action, r.text)
        used = []
        for p in paths:
            hit = next((r for r in self.allow if any(r.matches_subject(n, p) for n in names)), None)
            if hit is None:
                return Decision("default")
            used.append(hit.text)
        return Decision("allow", used[0])

    def decide(self, name: str, args: dict, root=None) -> Decision:
        """`root`: the workspace root; an absolute path inside it is matched as the relative path, so `/work/repo/.env`
        cannot dodge a `.env*` rule. (Symlinks inside the workspace are not followed here: they are a limit of path rules.)"""
        if name == "apply_patch":
            return self._decide_patch(args, root)
        subject = _subject(name, args)
        if name != "bash" and subject is not None:
            if root is not None and subject.startswith("/"):
                try:
                    subject = str(Path(subject).relative_to(root))
                except ValueError:
                    pass
            subject = _norm_path(subject)                  # `src/../.env` must not slip past a `.env*` rule
        segments = [s.strip() for s in _SEPARATORS.split(subject)] if name == "bash" and subject is not None else [subject]
        segments = [s for s in segments if s != ""] or [subject]
        for r in self.deny:
            if r.matches_subject(name, subject) or any(r.matches_subject(name, s) for s in segments):
                return Decision("deny", r.text)
        for r in self.ask:
            if r.matches_subject(name, subject) or any(r.matches_subject(name, s) for s in segments):
                return Decision("ask", r.text)
        if name == "bash":
            if subject is None or _UNSAFE_FOR_ALLOW.search(subject):
                return Decision("default")
            used = []
            for seg in segments:
                hit = next((r for r in self.allow if r.matches_subject(name, seg)), None)
                if hit is None:
                    return Decision("default")
                used.append(hit.text)
            return Decision("allow", used[0])
        for r in self.allow:
            if r.matches_subject(name, subject):
                return Decision("allow", r.text)
        return Decision("default")
