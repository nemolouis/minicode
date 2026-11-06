"""The state and operations behind both interactive clients (the line REPL and the full-screen terminal UI): one conversation in one
session directory, continued turn after turn, with /diff, /undo, /rewind, cost and status. No printing and no terminal here: the
front ends call these methods and decide how to show the results, so the logic is written (and tested) once."""
from __future__ import annotations

import difflib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import session as sess
from .trajectory import Trajectory, read_events


class ControllerError(Exception):
    """Something the user did not mean to cause (rewinding to a step that is gone): the message is meant to be shown as is."""


@dataclass
class StepInfo:
    step: int
    what: str                  # tool names of that step, or "final answer"
    files_changed: bool        # a workspace checkpoint exists for it


def workspace_diff(baseline: Path, workspace: Path, exclude: list[str] = ()) -> list[str]:
    """Unified diff of the workspace against a `session.snapshot` taken earlier (text files only; others are listed)."""
    meta = json.loads((baseline / "manifest.json").read_text())
    before = set(meta["files"])
    now = {p.relative_to(workspace).as_posix() for p in sess._files(workspace, exclude)}
    out: list[str] = []
    for rel in sorted(before | now):
        old_p, new_p = baseline / "files" / rel, workspace / rel
        if rel in before and rel in meta["skipped"]:
            continue                                                       # too large to have been copied: cannot say
        try:
            old = old_p.read_text(encoding="utf-8").splitlines() if rel in before else []
            new = new_p.read_text(encoding="utf-8").splitlines() if rel in now else []
        except (UnicodeDecodeError, OSError):
            if rel in before and rel in now and old_p.read_bytes() != new_p.read_bytes():
                out.append(f"Binary file {rel} differs")
            continue
        label_a, label_b = (f"a/{rel}" if rel in before else "/dev/null"), (f"b/{rel}" if rel in now else "/dev/null")
        out += list(difflib.unified_diff(old, new, label_a, label_b, lineterm=""))
    return out


class _TeeTrajectory(Trajectory):
    """A trajectory file that also reports every event to a callback (how the front end learns what the agent is doing)."""

    def __init__(self, run_dir, on_event: Callable[[str, dict], None]):
        super().__init__(run_dir)
        self._on_event = on_event

    def log(self, event: str, **data) -> None:
        super().log(event, **data)
        self._on_event(event, data)


class SessionController:
    def __init__(self, *, make_agent: Callable[..., object], run_dir: Path, workspace: Path, new_run_dir: Callable[[], Path],
                 resume: dict | None = None, git=None, price: dict | None = None):
        self.make_agent, self.run_dir, self.workspace = make_agent, Path(run_dir), Path(workspace).resolve()
        self.new_run_dir, self.git, self.price = new_run_dir, git, price
        self.state: dict | None = resume                 # the saved session the next turn continues (None before the first turn)
        self.turn_starts: list[int] = []                 # step number at which each turn began (for undo)
        self.tokens = {"in": 0, "out": 0}
        self.turns = 0
        self.agent = None
        self._baseline = Path(tempfile.mkdtemp(prefix="minicode-base-"))
        self._exclude: list[str] = []
        self._hooks: dict = {}

    # ---------------------------------------------------------------- lifecycle
    def open(self, *, on_event: Callable[[str, dict], None], on_text=None, confirm=None) -> None:
        """Snapshot the workspace (the baseline for `diff`) and create the agent; the callbacks are how the front end hears about it."""
        self._hooks = {"on_event": on_event, "on_text": on_text, "confirm": confirm}
        self._exclude = sess.exclusions(self.workspace, self.run_dir.resolve())
        sess.snapshot(self.workspace, self._baseline, exclude=self._exclude)
        self._new_agent()

    def _new_agent(self) -> None:
        h = self._hooks
        self.agent = self.make_agent(trajectory=_TeeTrajectory(self.run_dir, h["on_event"]), on_text=h["on_text"], confirm=h["confirm"])

    def close(self) -> None:
        self.stop_background()
        shutil.rmtree(self._baseline, ignore_errors=True)

    def stop_background(self) -> None:
        if self.agent is not None and getattr(self.agent, "background", None) is not None:
            self.agent.background.kill_all()

    # ---------------------------------------------------------------- a turn
    def run_turn(self, text: str):
        """One user message: continue the conversation. Returns the agent's RunResult."""
        self.turn_starts.append(self.state["step"] if self.state else 0)
        res = self.agent.run(text, resume=self.state)
        self.turns += 1
        self.tokens["in"] += res.input_tokens
        self.tokens["out"] += res.output_tokens
        self.reload_state()
        return res

    def reload_state(self) -> None:
        try:
            self.state = sess.load_session(self.run_dir)
        except (FileNotFoundError, ValueError):
            self.state = None

    def cancel(self) -> None:
        if self.agent is not None:
            self.agent.cancel()

    # ---------------------------------------------------------------- inspection
    def diff_lines(self) -> list[str]:
        return workspace_diff(self._baseline, self.workspace, self._exclude)

    def steps(self) -> list[StepInfo]:
        if not self.state or not self.state["step_index"]:
            return []
        what: dict[int, str] = {}
        try:
            for e in read_events(self.run_dir / "trajectory.jsonl"):
                if e["event"] == "model_call":
                    names = [c["name"] for c in e["tool_calls"]]
                    what[e["step"]] = ", ".join(names) if names else "final answer"
        except OSError:
            pass
        ckpt = set(sess.checkpoint_steps(self.run_dir))
        return [StepInfo(s, what.get(s, "?"), s in ckpt) for s in sorted(self.state["step_index"])]

    def cost_text(self) -> str:
        line = f"{self.tokens['in']} input + {self.tokens['out']} output tokens in {self.turns} turn(s)"
        p = self.price
        if p and all(p.get(k) is not None for k in ("input", "output")):
            usd = (self.tokens["in"] * p["input"] + self.tokens["out"] * p["output"]) / 1e6
            return line + f" ≈ ${usd:.4f} (cache discounts not applied: an upper bound)"
        return line + " (no price known for this model: pass --prices with an entry for it)"

    def cost_usd(self) -> float | None:
        p = self.price
        if p and all(p.get(k) is not None for k in ("input", "output")):
            return (self.tokens["in"] * p["input"] + self.tokens["out"] * p["output"]) / 1e6
        return None

    # ---------------------------------------------------------------- changing the session
    def rewind(self, step: int) -> str:
        """Put files and conversation back to the start of `step`. Returns a message for the user; raises ControllerError."""
        try:
            info = sess.rewind(self.run_dir, step, self.workspace)
        except (ValueError, FileNotFoundError) as e:
            raise ControllerError(f"cannot rewind: {e}")
        self.reload_state()
        msg = f"rewound to the start of step {step}: {info['restored']} file(s) restored, {info['removed']} removed; the next message continues from there"
        if self.git is not None:
            msg += "\nnote: --git commits made after that point stay in the branch history"
        return msg

    def undo(self) -> str:
        """Take back the last turn: files and conversation, including the user's message of that turn."""
        if not self.turn_starts:
            raise ControllerError("nothing to undo")
        step = self.turn_starts[-1] + 1
        if step not in (self.state or {}).get("step_index", {}):
            raise ControllerError("cannot undo that turn (a compaction since then removed the rewind point; try the step list)")
        self.turn_starts.pop()
        msg = self.rewind(step)
        # rewinding to the first step of a turn keeps that turn's user message (the model call at that step sees it); undoing the turn
        # means the message goes too
        state = sess.load_session(self.run_dir)
        if state["messages"] and state["messages"][-1]["role"] == "user":
            state["messages"].pop()
            sess.save_session(self.run_dir, state)
            self.reload_state()
        return msg

    def clear(self) -> str:
        """A new conversation in a new session directory."""
        self.stop_background()
        self.run_dir = Path(self.new_run_dir())
        self.state, self.turn_starts, self.turns = None, [], 0
        self._exclude = sess.exclusions(self.workspace, self.run_dir.resolve())
        self._new_agent()
        return f"new conversation; session directory {self.run_dir}"

    def set_model(self, name: str) -> None:
        self.agent.model = name

    def status_lines(self, flags: list[str]) -> list[str]:
        a = self.agent
        feat = f"features={','.join(sorted(a.features))}" if a.features else ""
        return [f"model {a.model} | workspace {self.workspace} | session {self.run_dir}",
                "mode: " + (", ".join(f for f in [*flags, "git" if self.git is not None else "", "session" if a.session else "", feat] if f) or "default")
                + f" | tools: {', '.join(sorted(a.tools))}",
                f"{self.turns} turn(s), step {self.state['step'] if self.state else 0}, {self.tokens['in']} in / {self.tokens['out']} out tokens"]
