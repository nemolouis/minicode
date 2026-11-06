"""`symbols`: a Python outline of the workspace built with the standard library `ast`, so the model can find where things are defined
without reading whole files or paging through grep output. Not a language server: Python only, definitions only (no references,
no type information), nothing is imported or executed."""
from __future__ import annotations

import ast
from pathlib import Path

from .tools import SKIP_DIRS, ToolError

MAX_LINES = 200
MAX_FILE_BYTES = 500_000


def _sig(node: ast.AST) -> str:
    a = node.args
    parts = []
    pos = [*a.posonlyargs, *a.args]
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults):
        parts.append(arg.arg + (f": {ast.unparse(arg.annotation)}" if arg.annotation else "") + (f"={ast.unparse(d)}" if d is not None else ""))
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + (f"={ast.unparse(d)}" if d is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({', '.join(parts)}){ret}"


def _doc(node: ast.AST) -> str:
    d = ast.get_docstring(node)
    return f"  # {d.strip().splitlines()[0][:80]}" if d else ""


def outline(source: str) -> list[tuple[int, int, str]]:
    """(line, depth, text) for every class / function defined in `source`; nested definitions are indented. Raises SyntaxError."""
    out: list[tuple[int, int, str]] = []

    def walk(body, depth):
        for node in body:
            if isinstance(node, ast.ClassDef):
                bases = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
                out.append((node.lineno, depth, f"class {node.name}{bases}{_doc(node)}"))
                walk(node.body, depth + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                out.append((node.lineno, depth, f"{kw} {node.name}{_sig(node)}{_doc(node)}"))
                walk(node.body, depth + 1)
            elif isinstance(node, ast.Assign) and depth == 0:
                names = [t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()]
                if names:
                    out.append((node.lineno, depth, f"{', '.join(names)} = ..."))

    walk(ast.parse(source).body, 0)
    return out


def symbols(root: Path, resolve, args: dict) -> str:
    """Outline of a file or directory (`path`, default "."), optionally only definitions whose name contains `query`."""
    base = resolve(args.get("path", "."))
    query = (args.get("query") or "").lower()
    if base.is_file():
        files = [base]
    elif base.is_dir():
        files = [f for f in sorted(base.rglob("*.py")) if f.is_file() and not (SKIP_DIRS & set(f.relative_to(root).parts))]
    else:
        raise ToolError(f"no such file or directory: {args.get('path')}")
    lines: list[str] = []
    skipped = 0
    for f in files:
        rel = f.relative_to(root).as_posix()
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                skipped += 1
                continue
            items = outline(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            lines.append(f"{rel}: (could not be parsed)")
            continue
        items = [it for it in items if not query or query in it[2].split("(")[0].lower()]
        if not items:
            continue
        room = MAX_LINES - len(lines) - 1
        if room <= 0:
            lines.append(f"... (stopped at {MAX_LINES} lines: narrow `path` or add a `query`)")
            break
        lines.append(f"{rel}:")
        lines += [f"  {n:>5}: {'    ' * depth}{text}" for n, depth, text in items[:room]]
        if len(items) > room:
            lines.append(f"... (stopped at {MAX_LINES} lines: narrow `path` or add a `query`)")
            break
    if skipped:
        lines.append(f"({skipped} file(s) over {MAX_FILE_BYTES // 1000} KB skipped)")
    return "\n".join(lines) or ("no definitions match" if query else "no Python definitions found")
