"""Tool implementations: read_file, write_file, edit_file, bash, grep.

Every tool takes a plain dict (the model's parsed arguments) and returns a string.
Failures raise ToolError; the agent loop turns those into is_error results so
the model can recover instead of the run crashing.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .proc import run_shell
from .safety import check_command

MAX_OUTPUT_CHARS = 20_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


class ToolError(Exception):
    pass


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS, hint: str = "") -> str:
    """Keep the head and the tail of a long output and say how much was cut and (with `hint`) how to get the rest."""
    if len(text) <= limit:
        return text
    half = limit // 2
    lines = text.count("\n") + 1
    return (f"{text[:half]}\n...[truncated {len(text) - limit} of {len(text)} chars from the middle ({lines} lines in all)"
            f"{'; ' + hint if hint else ''}]...\n{text[-half:]}")


def output_limit_for(context_limit: int, last_input_tokens: int, trigger_fraction: float = 0.7) -> int:
    """How many characters one tool output may take, given how much room is left before the context gets compacted: half of the
    headroom (3 chars per token, the same estimate compaction uses), but never below 2000 or above MAX_OUTPUT_CHARS. A fresh
    context allows the full 20000; a nearly full one still gets 2000 characters, enough to see an error."""
    headroom = max(int(context_limit * trigger_fraction) - last_input_tokens, 0)
    return max(2000, min(MAX_OUTPUT_CHARS, headroom * 3 // 2))


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    func: Callable[[dict], str]
    needs_confirm: bool = False
    read_only: bool = False   # no side effects and no confirmation: calls of such tools in one turn may run concurrently

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.input_schema}}


class Workspace:
    def __init__(self, root: str | Path, sandbox=None, edit_hints: bool = False, require_read: bool = False,
                 diagnostics: bool = False):
        """`sandbox`: a minicode.sandbox.Sandbox over `root`. With it, `bash` runs in the container instead of the host.
        `edit_hints`: edit_file failures also show the closest real lines / file names, so the model can retry.
        `require_read`: edit_file refuses an existing file this run has not read (or written) with read_file/write_file:
        an old_string written from memory is almost always wrong.
        `diagnostics`: after write_file / edit_file on a .py or .json file, append a warning if the result does not parse."""
        self.root = Path(root).resolve()
        self.sandbox = sandbox
        self.edit_hints = edit_hints
        self.require_read = require_read
        self.diagnostics = diagnostics
        self.seen: set[Path] = set()        # files the model has read or written in this run (resolved paths)
        self.cancel = None                      # a threading.Event set by the agent: a running command is killed when it is set
        self.output_limit = MAX_OUTPUT_CHARS   # characters per tool output; the agent lowers it as the context fills (output_budget)
        if not self.root.is_dir():
            raise ValueError(f"workspace is not a directory: {self.root}")

    def resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path escapes workspace: {path}")
        return p

    # ---- tools ----
    def read_file(self, args: dict) -> str:
        p = self.resolve(args["path"])
        if not p.is_file():
            raise ToolError(f"not a file: {args['path']}")
        offset = max(int(args.get("offset", 1)), 1)
        limit = int(args.get("limit", 2000))
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            raise ToolError("file is not valid UTF-8 text")
        self.seen.add(p)
        chunk = lines[offset - 1 : offset - 1 + limit]
        shown, size = [], 0
        for i, l in enumerate(chunk, start=offset):                  # whole lines only, so the model knows exactly where to continue
            row = f"{i:>6}\t{l}"
            if shown and size + len(row) + 1 > self.output_limit:
                break
            shown.append(row)
            size += len(row) + 1
        out = "\n".join(shown)
        last = offset - 1 + len(shown)                              # number of the last line shown
        if last < len(lines):
            why = "output size limit" if len(shown) < len(chunk) else f"{limit}-line limit"
            out += f"\n... ({len(lines) - last} more lines after line {last}, cut by the {why}: call read_file with offset={last + 1})"
        return truncate(out, max(self.output_limit, 2000)) or "(empty file)"

    def write_file(self, args: dict) -> str:
        p = self.resolve(args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        self.seen.add(p)
        return f"wrote {len(args['content'])} chars to {args['path']}" + self._diagnose(p, args["content"])

    def edit_file(self, args: dict) -> str:
        p = self.resolve(args["path"])
        if not p.is_file():
            raise ToolError(f"not a file: {args['path']}" + (self._similar_files(p) if self.edit_hints else ""))
        if self.require_read and p not in self.seen:
            raise ToolError(f"you have not read {args['path']} in this session: call read_file on it first, then copy old_string "
                            "from the output")
        old, new = args["old_string"], args["new_string"]
        if old == new:
            raise ToolError("old_string and new_string are identical" + (
                "; nothing would change. Put the corrected text in new_string." if self.edit_hints else ""))
        text = p.read_text(encoding="utf-8")
        n = text.count(old)
        if n == 0:
            raise ToolError("old_string not found (must match exactly, including whitespace)"
                            + (self._closest_lines(text, old) if self.edit_hints else ""))
        if n > 1:
            where = ""
            if self.edit_hints:
                lines = [text.count("\n", 0, m.start()) + 1 for m in re.finditer(re.escape(old), text)]
                where = f" (at lines {', '.join(map(str, lines[:8]))})"
            raise ToolError(f"old_string matches {n} places{where}; include more context to make it unique")
        updated = text.replace(old, new, 1)
        p.write_text(updated, encoding="utf-8")
        return f"edited {args['path']}" + self._diagnose(p, updated)

    def _diagnose(self, p: Path, text: str) -> str:
        """A one-line warning when a Python/JSON file the model just wrote does not parse. Compiles only, never runs the code."""
        if not self.diagnostics:
            return ""
        try:
            if p.suffix == ".py":
                compile(text, str(p.name), "exec")
            elif p.suffix == ".json":
                json.loads(text)
        except SyntaxError as e:
            return f"\nwarning: {p.name} no longer parses: {e.msg} (line {e.lineno}). Fix it before running anything."
        except ValueError as e:                                           # json.JSONDecodeError, null bytes in source
            return f"\nwarning: {p.name} no longer parses: {e}. Fix it before running anything."
        return ""

    def _similar_files(self, p: Path) -> str:
        d = p.parent if p.parent.is_dir() else self.root
        names = sorted(f.name for f in d.iterdir() if f.is_file())
        close = difflib.get_close_matches(p.name, names, n=3, cutoff=0.4)
        rel = d.relative_to(self.root).as_posix()
        return (f". Did you mean: {', '.join(close)}?" if close else "") + \
            f" Files in {rel}: {', '.join(names[:20])}" if names else ""

    @staticmethod
    def _closest_lines(text: str, old: str) -> str:
        """The file lines most similar to `old`, numbered, so the model can copy the exact text."""
        lines, want = text.splitlines(), old.splitlines() or [old]
        k = min(len(want), len(lines), 12)
        if k == 0:
            return ""
        norm = lambda ls: "\n".join(l.strip() for l in ls)
        target, best, at = norm(want[:k]), 0.0, 0
        for i in range(len(lines) - k + 1):
            r = difflib.SequenceMatcher(None, norm(lines[i:i + k]), target).quick_ratio()
            if r > best:
                best, at = r, i
        if best >= 0.5:
            best = difflib.SequenceMatcher(None, norm(lines[at:at + k]), target).ratio()
        if best < 0.5:
            return ". No similar text in this file; read_file it first and copy the lines you want to change."
        shown = "\n".join(f"{i:>6}\t{l}" for i, l in enumerate(lines[at:at + k], start=at + 1))
        same = norm(lines[at:at + k]) == target
        note = "differs only in indentation/whitespace" if same else "closest match in the file"
        return f". {note} (lines {at + 1}-{at + k}):\n{shown}"

    def bash(self, args: dict) -> str:
        cmd = self._repair_html_entities(args["command"])
        reason = check_command(cmd)
        if reason:
            raise ToolError(f"command blocked by safety policy: {reason}")
        timeout = min(int(args.get("timeout", 60)), 300)
        if self.sandbox is not None:
            r = self.sandbox.exec(cmd, timeout=timeout)
        else:
            r = run_shell(cmd, shell=True, cwd=self.root, timeout=timeout, cancel=self.cancel,
                          env={**os.environ, "GIT_PAGER": "cat", "PAGER": "cat"})
        if getattr(r, "cancelled", False):
            raise ToolError("command stopped by the user")
        if r.timed_out:
            raise ToolError(f"command timed out after {timeout}s")
        return truncate(f"exit code: {r.returncode}\n{r.stdout}{r.stderr}".rstrip(), self.output_limit,
                        "to see the cut part, re-run with `| head -n N`, `| tail -n N` or `> out.txt` and use grep / read_file on the file")

    def apply_patch(self, args: dict) -> str:
        from . import patch as P
        try:
            changes = P.plan(self.root, self.resolve, args["patch"],
                             read_ok=(lambda p: p in self.seen) if self.require_read else None)
            P.commit(changes)
        except P.PatchError as e:
            raise ToolError(str(e))
        except OSError as e:
            raise ToolError(f"patch not applied, files restored: {type(e).__name__}: {e}")
        notes = []
        for kind, p, old, new in changes:
            rel = p.relative_to(self.root).as_posix()
            if kind == "delete":
                notes.append(f"deleted {rel}")
                self.seen.discard(p)
                continue
            self.seen.add(p)
            added = len(new.splitlines()) - (len(old.splitlines()) if old is not None else 0)
            notes.append(f"{'added' if kind == 'add' else 'updated'} {rel} ({added:+d} lines)" + self._diagnose(p, new))
        return "applied patch: " + "; ".join(notes)

    def symbols(self, args: dict) -> str:
        from .symbols import symbols
        return symbols(self.root, self.resolve, args)

    def glob(self, args: dict) -> str:
        pat = args["pattern"]
        if pat.startswith("/") or ".." in Path(pat).parts:
            raise ToolError("pattern must be relative and stay inside the workspace")
        out = []
        for f in sorted(self.root.glob(pat)):
            if not f.is_file() or SKIP_DIRS & set(f.relative_to(self.root).parts):
                continue
            if self.root not in f.resolve().parents:          # a symlink pointing out of the workspace
                continue
            out.append(f.relative_to(self.root).as_posix())
            if len(out) >= 200:
                return "\n".join(out) + "\n... (stopped at 200 paths)"
        return "\n".join(out) or "no matches"

    @staticmethod
    def _repair_html_entities(cmd: str) -> str:
        """Some models write `&amp;&amp;` / `&gt;` in the arguments of a tool call. If the command is not valid shell as written but is
        once the entities are decoded, run the decoded one; a command that parses is never touched (grep '&amp;' in HTML is legitimate)."""
        if "&" not in cmd or not re.search(r"&(amp|lt|gt|quot|#39);", cmd):
            return cmd
        import html
        fixed = html.unescape(cmd)
        try:
            ok_as_written = subprocess.run(["sh", "-n", "-c", cmd], capture_output=True, timeout=5).returncode == 0
            ok_decoded = subprocess.run(["sh", "-n", "-c", fixed], capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return cmd
        return fixed if (not ok_as_written and ok_decoded) else cmd

    def grep(self, args: dict) -> str:
        try:
            rx = re.compile(args["pattern"])
        except re.error as e:
            raise ToolError(f"invalid regex: {e}")
        base = self.resolve(args.get("path", "."))
        glob = args.get("glob")
        files = [base] if base.is_file() else (
            f for f in sorted(base.rglob(glob or "*"))
            if f.is_file() and not (SKIP_DIRS & set(f.relative_to(self.root).parts))
        )
        try:
            skip, cap = max(int(args.get("offset", 0)), 0), min(max(int(args.get("max_results", 200)), 1), 200)
        except (TypeError, ValueError):
            raise ToolError("offset and max_results must be integers")
        hits: list[str] = []
        seen = 0                                                     # matches found so far, including the skipped ones
        for f in files:
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        seen += 1
                        if seen <= skip:
                            continue
                        if len(hits) >= cap:                        # one more than we may show: there is more
                            return ("\n".join(hits) + f"\n... (showing matches {skip + 1}-{skip + cap}; more exist: call grep again with "
                                    f"offset={skip + cap}, or narrow the pattern, path or glob)")
                        hits.append(f"{f.relative_to(self.root)}:{i}:{line[:300]}")
            except (UnicodeDecodeError, OSError):
                continue
        if not hits and skip:
            return f"no matches after the first {skip}"
        return truncate("\n".join(hits), self.output_limit) or "no matches"


def build_tools(ws: Workspace) -> dict[str, Tool]:
    S = lambda **props: {"type": "object", "properties": props, "required": [k for k, v in props.items() if not v.pop("optional", False)]}
    tools = [
        Tool("read_file", "Read a text file with line numbers. Paths are relative to the workspace root.",
             S(path={"type": "string"}, offset={"type": "integer", "description": "1-based start line", "optional": True},
               limit={"type": "integer", "description": "max lines (default 2000)", "optional": True}),
             ws.read_file, read_only=True),
        Tool("write_file", "Create or overwrite a file with the given content.",
             S(path={"type": "string"}, content={"type": "string"}), ws.write_file, needs_confirm=True),
        Tool("edit_file", "Replace one exact occurrence of old_string with new_string. Fails if old_string is missing or not unique.",
             S(path={"type": "string"}, old_string={"type": "string"}, new_string={"type": "string"}),
             ws.edit_file, needs_confirm=True),
        Tool("bash", "Run a shell command in the workspace root. Use for running tests, git, etc.",
             S(command={"type": "string"}, timeout={"type": "integer", "description": "seconds, default 60, max 300", "optional": True}),
             ws.bash, needs_confirm=True),
        Tool("grep", "Regex search over files (skips .git, node_modules, venvs). Returns path:line:text.",
             S(pattern={"type": "string"}, path={"type": "string", "description": "file or dir, default '.'", "optional": True},
               glob={"type": "string", "description": "e.g. '*.py'", "optional": True},
               offset={"type": "integer", "description": "skip this many matches (to page through a long result)", "optional": True},
               max_results={"type": "integer", "description": "matches to return (default and maximum 200)", "optional": True}),
             ws.grep, read_only=True),
    ]
    return {t.name: t for t in tools}


def patch_tool(ws: Workspace) -> Tool:
    """Opt-in (features "patch"): several files edited in one call, all or nothing. See minicode.patch for the format."""
    return Tool("apply_patch",
                "Edit several files in one call, atomically: if any part fails, no file is changed. Format:\n"
                "*** Begin Patch\n*** Update File: path\n@@ optional anchor text (a line of the file to search after)\n context line\n"
                "-line to remove\n+line to add\n*** Add File: path\n+content line\n*** Delete File: path\n*** End Patch\n"
                "Context and removed lines must match the file exactly and occur once (use @@ to disambiguate). Read files first.",
                {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}, ws.apply_patch, needs_confirm=True)


def symbols_tool(ws: Workspace) -> Tool:
    """Opt-in (features "symbols"): where Python classes and functions are defined, without reading the files."""
    return Tool("symbols",
                "Outline of Python definitions (classes, functions with signatures, UPPERCASE constants) with line numbers, for a file "
                "or a directory tree. Use `query` to keep only names containing a substring. Definitions only, no references.",
                {"type": "object", "properties": {"path": {"type": "string", "description": "file or dir, default '.'"},
                                                   "query": {"type": "string", "description": "substring of the name"}}},
                ws.symbols, read_only=True)


def glob_tool(ws: Workspace) -> Tool:
    """Opt-in (Agent features=("glob",)): list files by pattern. Kept out of build_tools so existing runs keep their tool set."""
    return Tool("glob", "List files matching a glob pattern relative to the workspace root, e.g. '**/*.py' or 'tests/*.py' "
                "(skips .git, node_modules, venvs). Returns up to 200 paths.",
                {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
                ws.glob, read_only=True)
