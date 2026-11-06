"""The interactive terminal client: a multi-turn conversation with the agent in one process.

    $ minicode -w my/repo -m deepseek-chat --stream --git
    minicode> fix the failing test in stats.py
    ... streamed answer, one line per tool call, a diff preview before every edit that needs your OK ...
    minicode> /diff        everything changed since this session started
    minicode> /undo        take the last turn back (files and conversation)
    minicode> /cost

It drives the same `Agent` as the one-shot command and uses the session machinery for everything stateful: each turn continues the saved
conversation (`Agent.run(text, resume=...)`), `/rewind` and `/undo` are `session.rewind`, Ctrl-C during a turn is the agent's graceful
interruption, and the whole conversation is one trajectory that `minicode-eval view` can replay. The terminal is injected (`Terminal`), so
the loop is tested with scripted input.
"""
from __future__ import annotations

import difflib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

from .controller import ControllerError, SessionController, workspace_diff  # noqa: F401  (workspace_diff re-exported)

PROMPT = "minicode> "
HELP = """commands (everything else is sent to the agent):
  /help                 this text
  /status               model, workspace, mode, tokens so far
  /cost                 tokens and, when the model has a price, USD spent in this session
  /diff                 what changed in the workspace since the session started (unified diff)
  /steps                the points you can rewind to, with what happened there
  /rewind N             put files and conversation back to the start of step N
  /undo                 take back the last turn (files and conversation)
  /todo                 the agent's todo list (needs --features todo)
  /model NAME           switch model for the next turns
  /stream on|off        show the reply while it is generated
  /yes on|off           approve every edit and command without asking (the hard deny-list still applies)
  /clear                start a new conversation (a new session directory)
  /exit                 leave (also Ctrl-D)
Ctrl-C stops the current turn and keeps the conversation. A line ending in \\ continues on the next line; \"\"\" opens a multi-line block."""


class Terminal:
    """Input and output of the client; replaced by a scripted one in tests."""

    def __init__(self, inp=None, out=None, color: bool | None = None):
        self.inp, self.out = inp or sys.stdin, out or sys.stdout
        tty = hasattr(self.out, "isatty") and self.out.isatty()
        self.color = (tty and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb") if color is None else color
        self.interactive = inp is None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        self._at_line_start = True

    def style(self, text: str, *codes: str) -> str:
        return f"\033[{';'.join(codes)}m{text}\033[0m" if self.color and codes else text

    def write(self, text: str) -> None:
        if text:
            self.out.write(text)
            self.out.flush()
            self._at_line_start = text.endswith("\n")

    def line(self, text: str = "") -> None:
        if not self._at_line_start:
            self.write("\n")
        self.write(text + "\n")

    def read(self, prompt: str) -> str | None:
        """One line, or None at end of input."""
        if self.interactive:
            try:
                import readline  # noqa: F401  (importing it gives input() history and line editing)
            except ImportError:
                pass
            try:
                return input(self.style(prompt, "1", "36") if self.color else prompt)
            except EOFError:
                return None
        self.write(prompt)
        raw = self.inp.readline()
        if raw == "":
            return None
        self.write(raw if raw.endswith("\n") else raw + "\n")                  # echo the line: scripted input should read like a transcript
        return raw.rstrip("\n")


# ---------------------------------------------------------------- what to show before asking
def describe_action(name: str, args: dict, root: Path, *, max_lines: int = 40) -> list[str]:
    """Lines that tell the user what an action will do, with a diff for edits."""
    def clip(lines: list[str]) -> list[str]:
        return lines[:max_lines] + ([f"... ({len(lines) - max_lines} more lines)"] if len(lines) > max_lines else [])

    path = str(args.get("path", ""))
    if name == "edit_file":
        diff = difflib.unified_diff(str(args.get("old_string", "")).splitlines(), str(args.get("new_string", "")).splitlines(), lineterm="", n=0)
        return [f"edit {path}"] + clip([l for l in diff if not l.startswith(("---", "+++", "@@"))])
    if name == "write_file":
        new = str(args.get("content", ""))
        target = (root / path)
        if target.is_file():
            try:
                old = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return [f"overwrite {path} ({len(new.splitlines())} lines)"]
            diff = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2) if not l.startswith(("---", "+++"))]
            return [f"overwrite {path}"] + clip(diff)
        return [f"create {path} ({len(new.splitlines())} lines)"] + clip(["+" + l for l in new.splitlines()])
    if name == "apply_patch":
        return ["apply patch"] + clip(str(args.get("patch", "")).splitlines())
    if name in ("bash", "bash_background"):
        return [("run in background: " if name == "bash_background" else "run: ") + str(args.get("command", ""))]
    return [f"{name} {json.dumps(args, ensure_ascii=False)[:300]}"]


def colour_diff_line(term: Terminal, line: str) -> str:
    if line.startswith("+"):
        return term.style(line, "32")
    if line.startswith("-"):
        return term.style(line, "31")
    if line.startswith("@@"):
        return term.style(line, "36")
    return line


class Repl:
    def __init__(self, *, make_agent: Callable[..., object], run_dir: Path, workspace: Path, term: Terminal, new_run_dir: Callable[[], Path],
                 resume: dict | None = None, git=None, price: dict | None = None, auto_approve: bool = False, stream: bool = False):
        self.term = term
        self.ctl = SessionController(make_agent=make_agent, run_dir=run_dir, workspace=workspace, new_run_dir=new_run_dir, resume=resume,
                                     git=git, price=price)
        self.auto_approve, self.stream = auto_approve, stream

    # the controller's state, under the names the rest of the code (and the tests) use
    run_dir = property(lambda self: self.ctl.run_dir)
    workspace = property(lambda self: self.ctl.workspace)
    state = property(lambda self: self.ctl.state)
    turns = property(lambda self: self.ctl.turns)
    tokens = property(lambda self: self.ctl.tokens)
    agent = property(lambda self: self.ctl.agent)
    git = property(lambda self: self.ctl.git)

    # ---------------------------------------------------------------- lifecycle
    def run(self) -> int:
        t = self.term
        self._open()
        t.line(t.style("minicode", "1") + f"  {self.agent.model}  {self.workspace}" + ("  (resumed)" if self.state else ""))
        t.line(t.style("type a task, or /help. Ctrl-C stops a turn, /exit or Ctrl-D leaves.", "2"))
        try:
            while True:
                line = self._read_message()
                if line is None:
                    break
                if not line.strip():
                    continue
                if line.lstrip().startswith("/"):
                    if not self.command(line.strip()):
                        break
                else:
                    self.turn(line)
        except KeyboardInterrupt:
            t.line()
        finally:
            self.close()
        return 0

    def _open(self) -> None:
        self.ctl.open(on_event=self._event, on_text=self._stream_text if self.stream else None, confirm=self.confirm)

    def close(self) -> None:
        self.ctl.close()
        t = self.term
        if self.turns:
            t.line(t.style(f"session: {self.run_dir}", "2"))
            t.line(t.style(f"continue later with: minicode -w {self.workspace} --resume {self.run_dir}", "2"))
            if self.git is not None:
                t.line(self.git.summary())

    def _read_message(self) -> str | None:
        first = self.term.read(PROMPT)
        if first is None:
            return None
        if first.rstrip().endswith("\\") or first.strip().startswith('"""'):
            block = first.strip()
            lines = []
            if block.startswith('"""'):
                rest = block[3:]
                if rest.endswith('"""') and len(rest) >= 3:
                    return rest[:-3]
                lines.append(rest)
                while True:
                    nxt = self.term.read("... ")
                    if nxt is None:
                        break
                    if nxt.strip().endswith('"""'):
                        lines.append(nxt.rstrip()[:-3])
                        break
                    lines.append(nxt)
                return "\n".join(lines)
            lines.append(first.rstrip()[:-1])
            while True:
                nxt = self.term.read("... ")
                if nxt is None:
                    break
                if nxt.rstrip().endswith("\\"):
                    lines.append(nxt.rstrip()[:-1])
                else:
                    lines.append(nxt)
                    break
            return "\n".join(lines)
        return first

    # ---------------------------------------------------------------- one turn
    def turn(self, text: str) -> None:
        t = self.term
        t0 = time.time()
        res = self.ctl.run_turn(text)
        if self.stream:
            t.line()
        elif res.final_text:
            t.line(res.final_text)
        colour = {"done": "32", "interrupted": "33"}.get(res.status, "31")
        t.line(t.style(f"[{res.status}] {res.steps} steps total, {res.input_tokens} in / {res.output_tokens} out tokens, {time.time() - t0:.1f}s", colour))

    # ---------------------------------------------------------------- what the agent tells us
    def _stream_text(self, piece: str) -> None:
        self.term.write(piece)

    def _event(self, kind: str, d: dict) -> None:
        t = self.term
        if kind == "tool_result":
            name, args = d["name"], d["input"]
            brief = args.get("command") or args.get("path") or args.get("pattern") or args.get("question") or args.get("query") or ""
            if name == "apply_patch":
                brief = "patch"
            mark = t.style("✗", "31") if d["is_error"] else t.style("✓", "32")
            extra = []
            if d.get("parallel"):
                extra.append(f"parallel×{d['parallel']}")
            if d.get("permission") == "deny":
                extra.append("denied by rule")
            first = str(d["output"]).strip().splitlines()[0][:100] if str(d["output"]).strip() else ""
            t.line(f"  {mark} {t.style(name, '1')} {str(brief)[:90]}" + (f"  ({', '.join(extra)})" if extra else "") +
                   (f"  {t.style(first, '2')}" if d["is_error"] else ""))
        elif kind == "compaction" and d.get("kind") == "summary":
            t.line(t.style("  (history summarised to save context)", "2"))
        elif kind == "stuck":
            t.line(t.style(f"  stuck detector: {d['kind']} ×{d['count']} → {d['action']}", "33"))
        elif kind == "budget":
            t.line(t.style(f"  budget: {d['reason']}", "33"))
        elif kind == "git_commit" and d.get("sha"):
            t.line(t.style(f"  git {d['sha'][:8]}  {', '.join(d['files'][:4])}", "2"))

    def confirm(self, name: str, args: dict) -> bool:
        if self.auto_approve:
            return True
        t = self.term
        t.line(t.style("  needs your approval:", "33"))
        for line in describe_action(name, args, self.workspace):
            t.line("    " + colour_diff_line(t, line))
        while True:
            ans = t.read("  allow? [y]es / [n]o / [a]ll (rest of this session): ")
            if ans is None:
                return False
            ans = ans.strip().lower()
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no", ""):
                return False
            if ans in ("a", "all"):
                self.auto_approve = True
                return True

    # ---------------------------------------------------------------- slash commands
    def command(self, line: str) -> bool:
        """Returns False when the client should exit."""
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        t = self.term
        handler = {
            "/help": lambda: t.line(HELP), "/?": lambda: t.line(HELP),
            "/exit": None, "/quit": None,
            "/status": self._status, "/cost": lambda: t.line(self.ctl.cost_text()), "/diff": self._diff, "/steps": self._steps,
            "/todo": self._todo, "/undo": lambda: self._attempt(self.ctl.undo), "/clear": lambda: t.line(self._clear()),
            "/rewind": lambda: self._rewind(arg), "/model": lambda: self._model(arg),
            "/stream": lambda: self._toggle("stream", arg), "/yes": lambda: self._toggle("yes", arg),
        }
        if name not in handler:
            t.line(t.style(f"unknown command {name}; /help lists them", "31"))
            return True
        if handler[name] is None:
            return False
        handler[name]()
        return True

    def _attempt(self, action) -> None:
        """Run a controller action and show its message; a ControllerError is shown as is (red if it says the action cannot be done)."""
        try:
            self.term.line(action())
        except ControllerError as e:
            self.term.line(self.term.style(str(e), "31") if str(e).startswith("cannot") else str(e))

    def _status(self) -> None:
        for line in self.ctl.status_lines([n for n, on in (("stream", self.stream), ("auto-approve", self.auto_approve)) if on]):
            self.term.line(line)

    def _diff(self) -> None:
        t = self.term
        lines = self.ctl.diff_lines()
        if not lines:
            t.line("no changes since the session started")
            return
        for line in lines:
            t.line(colour_diff_line(t, line))

    def _steps(self) -> None:
        t = self.term
        steps = self.ctl.steps()
        if not steps:
            t.line("nothing to rewind to yet")
            return
        for s in steps:
            t.line(f"  step {s.step:>3}  {s.what:<40}" + ("  (files changed here)" if s.files_changed else ""))

    def _rewind(self, arg: str) -> None:
        if not arg.isdigit():
            self.term.line("usage: /rewind N   (see /steps)")
            return
        self._attempt(lambda: self.ctl.rewind(int(arg)))

    def _todo(self) -> None:
        t = self.term
        from .features import render_todos
        if self.agent.todos:
            t.line(render_todos(self.agent.todos))
        else:
            t.line("(no todo list" + ("" if "todo" in self.agent.features else "; start with --features todo") + ")")

    def _model(self, arg: str) -> None:
        if not arg:
            self.term.line(f"model: {self.agent.model}")
            return
        self.ctl.set_model(arg)
        self.term.line(f"model for the next turns: {arg}")

    def _toggle(self, what: str, arg: str) -> None:
        on = {"on": True, "off": False}.get(arg.lower())
        if on is None:
            self.term.line(f"usage: /{'stream' if what == 'stream' else 'yes'} on|off")
            return
        if what == "stream":
            self.stream = on
            self.agent.on_text = self._stream_text if on else None
        else:
            self.auto_approve = on
        self.term.line(f"{what}: {'on' if on else 'off'}")

    def _clear(self) -> str:
        return self.ctl.clear()
