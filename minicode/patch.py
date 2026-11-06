"""`apply_patch`: edit several files in one call, all or nothing.

Format (the one OpenAI's Codex CLI uses; no line numbers, which models get wrong in unified diffs):

    *** Begin Patch
    *** Update File: pkg/mod.py
    @@ def area(r):                 optional anchor: the hunk is searched after the first line containing this text
     context line (kept)
    -line to remove
    +line to add
    *** Add File: pkg/new.py
    +first line
    +second line
    *** Delete File: old.py
    *** End Patch

A hunk is the run of ` ` (context), `-` and `+` lines after an `@@` line (the first hunk of a file needs no `@@`). Its context and
removed lines must occur in the file exactly once (or, with an anchor, once after the anchor); a hunk that does not match, matches
twice, or a missing/already existing file makes the WHOLE patch fail and nothing is written. Everything is computed in memory first;
the writes happen only when every file checked out, and are rolled back if one of them fails.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADER = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+?)\s*$")


class PatchError(Exception):
    pass


@dataclass
class Hunk:
    anchor: str | None
    lines: list[tuple[str, str]] = field(default_factory=list)      # (" " | "-" | "+", text)


@dataclass
class FileOp:
    kind: str                       # "add" | "update" | "delete"
    path: str
    hunks: list[Hunk] = field(default_factory=list)
    content: list[str] = field(default_factory=list)                 # for add


def patch_paths(text: str) -> list[str]:
    """The paths a patch touches, from its headers only (used by the permission rules; never raises)."""
    return [m.group(2) for line in str(text).splitlines() if (m := HEADER.match(line))]


def parse(text: str) -> list[FileOp]:
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise PatchError("a patch must start with '*** Begin Patch'")
    if lines[-1].strip() != "*** End Patch":
        raise PatchError("a patch must end with '*** End Patch'")
    ops: list[FileOp] = []
    cur: FileOp | None = None
    hunk: Hunk | None = None
    for n, line in enumerate(lines[1:-1], start=2):
        m = HEADER.match(line)
        if m:
            cur, hunk = FileOp(m.group(1).lower(), m.group(2)), None
            ops.append(cur)
            continue
        if cur is None:
            if not line.strip():
                continue
            raise PatchError(f"line {n}: expected a '*** Add/Update/Delete File: path' header, got {line[:60]!r}")
        if cur.kind == "delete":
            if line.strip():
                raise PatchError(f"line {n}: a Delete File section has no body, got {line[:60]!r}")
            continue
        if cur.kind == "add":
            if not line.startswith("+"):
                raise PatchError(f"line {n}: every line of an Add File section must start with '+', got {line[:60]!r}")
            cur.content.append(line[1:])
            continue
        if line.startswith("@@"):
            hunk = Hunk(line[2:].strip() or None)
            cur.hunks.append(hunk)
            continue
        if hunk is None:
            hunk = Hunk(None)
            cur.hunks.append(hunk)
        if line[:1] in (" ", "-", "+"):
            hunk.lines.append((line[0], line[1:]))
        elif line == "":
            hunk.lines.append((" ", ""))                              # a blank context line whose leading space was stripped
        else:
            raise PatchError(f"line {n}: hunk lines start with ' ', '-' or '+', got {line[:60]!r}")
    if not ops:
        raise PatchError("the patch contains no files")
    seen: set[str] = set()
    for op in ops:
        if op.path in seen:
            raise PatchError(f"{op.path} appears twice in the patch; merge its hunks into one section")
        seen.add(op.path)
        if op.kind == "update" and not any(any(k != " " for k, _ in h.lines) for h in op.hunks):
            raise PatchError(f"{op.path}: an Update File section needs at least one '-' or '+' line")
    return ops


def _find(lines: list[str], block: list[str], start: int) -> list[int]:
    n = len(block)
    exact = [i for i in range(start, len(lines) - n + 1) if lines[i:i + n] == block]
    if exact:
        return exact
    loose = [x.rstrip() for x in block]                              # fall back to ignoring trailing whitespace
    return [i for i in range(start, len(lines) - n + 1) if [x.rstrip() for x in lines[i:i + n]] == loose]


def apply_hunks(path: str, original: str, hunks: list[Hunk]) -> str:
    had_newline = original.endswith("\n")
    lines = original.split("\n")
    if had_newline:
        lines.pop()
    out = list(lines)
    for k, h in enumerate(hunks, start=1):
        old = [t for kind, t in h.lines if kind != "+"]
        new = [t for kind, t in h.lines if kind != "-"]               # (used for pure insertions)
        start = 0
        if h.anchor:
            anchors = [i for i, ln in enumerate(out) if h.anchor in ln]
            if not anchors:
                raise PatchError(f"{path}, hunk {k}: anchor {h.anchor!r} not found in the file")
            start = anchors[0]
        if not old:                                                  # pure insertion with no context: only at the end (or after an anchor)
            at = len(out) if not h.anchor else start + 1
            out[at:at] = new
            continue
        found = _find(out, old, start)
        if not found:
            first = old[0].strip()
            near = next((i + 1 for i, ln in enumerate(out) if first and first in ln), None)
            raise PatchError(f"{path}, hunk {k}: the context/removed lines were not found" +
                             (f" (the first of them appears near line {near}: read the file and copy the lines exactly)" if near else
                              ": read the file and copy the lines exactly"))
        if len(found) > 1:
            raise PatchError(f"{path}, hunk {k}: the context/removed lines match {len(found)} places (lines "
                             f"{', '.join(str(i + 1) for i in found[:6])}); add more context or an @@ anchor")
        i = found[0]
        rebuilt, k = [], i                                   # context lines keep the file's own text (it may differ in trailing space)
        for kind, text in h.lines:
            if kind == "+":
                rebuilt.append(text)
            else:
                if kind == " ":
                    rebuilt.append(out[k])
                k += 1
        out[i:i + len(old)] = rebuilt
    text = "\n".join(out)
    return text + "\n" if (had_newline or not original) and out else text


def plan(root: Path, resolve, text: str, *, read_ok=None) -> list[tuple[str, Path, str | None, str | None]]:
    """Check the whole patch against the files on disk. Returns [(kind, path, old_text, new_text)]; raises PatchError for the first
    problem. Nothing is written. `resolve` maps a patch path to a Path inside the workspace; `read_ok(path)` may veto an update."""
    ops = parse(text)
    result = []
    for op in ops:
        p = resolve(op.path)
        if op.kind == "add":
            if p.exists():
                raise PatchError(f"{op.path}: Add File, but the file already exists (use Update File)")
            body = "\n".join(op.content) + ("\n" if op.content else "")
            result.append(("add", p, None, body))
        elif op.kind == "delete":
            if not p.is_file():
                raise PatchError(f"{op.path}: Delete File, but there is no such file")
            result.append(("delete", p, p.read_text(encoding="utf-8"), None))
        else:
            if not p.is_file():
                raise PatchError(f"{op.path}: Update File, but there is no such file")
            if read_ok is not None and not read_ok(p):
                raise PatchError(f"you have not read {op.path} in this session: call read_file on it first, then copy the lines from the output")
            try:
                old = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise PatchError(f"{op.path}: not a UTF-8 text file")
            result.append(("update", p, old, apply_hunks(op.path, old, op.hunks)))
    return result


def commit(changes: list[tuple[str, Path, str | None, str | None]]) -> None:
    """Write a planned patch; if any write fails, put every file back as it was and re-raise."""
    done: list[tuple[str, Path, str | None]] = []
    try:
        for kind, p, old, new in changes:
            if kind == "delete":
                p.unlink()
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(new, encoding="utf-8")
            done.append((kind, p, old))
    except Exception:
        for kind, p, old in reversed(done):
            if kind == "add":
                p.unlink(missing_ok=True)
            else:
                p.write_text(old, encoding="utf-8")
        raise
