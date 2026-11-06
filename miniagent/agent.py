"""The agent loop, written directly against the chat-completions tool-calling protocol."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import compaction
from .client import ModelResponse
from .budget import Budget
from .compaction import SUMMARY_PROMPT  # noqa: F401  (re-exported: it used to live here)
from .executor import ToolExecutor
from .features import FEATURES, background_tools, explore_tool, render_todos, todo_tool, with_project_memory
from .permissions import Permissions
from .provenance import provenance
from .session import Recorder
from .stuck import StuckDetector, notice as stuck_notice
from .tools import Tool, Workspace, build_tools, glob_tool, output_limit_for, patch_tool, symbols_tool
from .trajectory import SCHEMA_VERSION, Trajectory

SYSTEM_PROMPT = """You are a coding agent working inside a git-style repository. \
Your job is to fix the bug or complete the task the user describes.

Work method: explore first (grep, read_file), form a hypothesis, make the smallest \
edit that fixes the root cause, then run the project's tests with bash to verify. \
Do not edit tests to make them pass. Paths are relative to the workspace root, and bash already runs there: do not cd elsewhere. \
When you are done, reply with a short summary of what you changed and the test result, \
and make no further tool calls."""

class CancelledByUser(KeyboardInterrupt):
    """Raised inside run() when cancel() was called from another thread. A KeyboardInterrupt, so it takes the same graceful path as Ctrl-C."""


@dataclass
class RunResult:
    status: str  # done | max_steps | length | error
    steps: int
    final_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class Agent:
    client: object  # miniagent.client.ChatClient or anything with a compatible .chat()
    model: str
    workspace: Workspace
    confirm: Callable[[str, dict], bool] = lambda name, args: True
    trajectory: Trajectory = field(default_factory=lambda: Trajectory(None))
    max_steps: int = 30
    max_tokens: int = 4096
    context_limit: int = 100_000  # compact when input tokens exceed 70% of this
    keep_recent: int = 8  # messages kept verbatim during compaction
    # If a compaction cannot bring the next request below the 70% threshold (the prompt, the tool definitions and the
    # verbatim recent messages are already too big), compacting again at once cannot help. The loop then waits until the
    # input has grown by this fraction of its post-compaction size (1.0 = it may double). When the last compaction did
    # get below the threshold nothing changes: the trigger stays at 70% of context_limit.
    compact_regrowth: float = 1.0
    system: str = SYSTEM_PROMPT
    temperature: float | None = None  # None -> API default; evals should set it
    disabled_tools: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)  # task_id, config name, ... -> run_start
    extra_tools: dict = field(default_factory=dict)  # e.g. MCP tools: name -> Tool (see miniagent.mcp.client)
    log_reasoning: bool = False  # also write the model's visible thinking into model_call events (large; for SFT data)
    # Run the read-only tool calls of one model turn concurrently (results and log lines keep the model's order).
    parallel_tools: bool = True
    # Check tool arguments against the tool's JSON schema before running it; a bad call gets a precise error instead of a KeyError.
    validate_args: bool = True
    # Opt-in features, see miniagent.features: "memory" (AGENTS.md in the system prompt), "todo" (todo_write tool),
    # "explore" (read-only sub-agent tool), "glob" (file listing tool).
    features: tuple[str, ...] = ()
    explore_max_steps: int = 12
    # Allow/ask/deny rules over tool calls, see miniagent.permissions. None = each tool's own default.
    permissions: Permissions | None = None
    # Write session.json and workspace checkpoints next to the trajectory so the run can be resumed / rewound (miniagent.session).
    session: bool = False
    # Stuck detection (miniagent.stuck): after this many identical calls with identical results in a row (or consecutive errors
    # of one kind) a notice is appended to the tool result; after stuck_stop_after the run ends with status "stuck". 0 = off.
    stuck_warn_after: int = 0
    stuck_stop_after: int = 6
    # Hard spending limits (miniagent.budget): the run ends with status "budget" before the request that would exceed them.
    budget: Budget | None = None
    # Scale the size of one tool output to the room left before compaction (tools.output_limit_for). Off: always 20000 characters.
    output_budget: bool = False
    # Called with each piece of reply text as it streams in (OpenAI-compatible clients only); None = no streaming.
    on_text: Callable[[str], None] | None = None
    # A started miniagent.gitwork.GitSession: one commit per tool turn that changed files, on the agent's own branch.
    git: object | None = None
    # Leave background jobs running when run() returns (the interactive client keeps one agent across turns and stops them on exit).
    keep_background: bool = False

    def __post_init__(self):
        bad = set(self.features) - FEATURES
        if bad:
            raise ValueError(f"unknown features {sorted(bad)}; known: {sorted(FEATURES)}")
        self.todos: list[dict] = []
        self.sub_usage = {"input": 0, "output": 0}      # tokens spent by sub-agents (explore); not part of this loop's model_calls
        self._usage_lock = threading.Lock()
        if "memory" in self.features:
            self.system = with_project_memory(self.system, self.workspace.root)
        tools = {**build_tools(self.workspace), **self.extra_tools}
        if "glob" in self.features or "explore" in self.features:
            tools["glob"] = glob_tool(self.workspace)
        self.background = None
        if "patch" in self.features:
            tools["apply_patch"] = patch_tool(self.workspace)
        if "symbols" in self.features:
            tools["symbols"] = symbols_tool(self.workspace)
        if "background" in self.features:
            tools.update(background_tools(self))
        if "todo" in self.features:
            tools["todo_write"] = todo_tool(self)
        if "explore" in self.features:
            tools["explore"] = explore_tool(self)
        self.tools: dict[str, Tool] = {n: t for n, t in tools.items() if n not in self.disabled_tools}
        self._recorder: Recorder | None = None             # set by run()
        self.cancel_event = threading.Event()              # set by cancel(), possibly from another thread (the terminal UI)
        self.workspace.cancel = self.cancel_event

    def cancel(self) -> None:
        """Ask the run to stop at the next safe point: before the next request, while a reply is streaming, while a shell command runs,
        between tool turns. run() then ends with status "interrupted" and a session that can be resumed, exactly as after Ctrl-C."""
        self.cancel_event.set()

    def _guard(self, callback):
        """`callback` for streamed text that aborts the stream (raising CancelledByUser) once cancel() was called."""
        def guarded(piece: str) -> None:
            self._check_cancel()
            callback(piece)
        return guarded

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledByUser

    def add_sub_usage(self, input_tokens: int, output_tokens: int) -> None:
        with self._usage_lock:
            self.sub_usage["input"] += input_tokens
            self.sub_usage["output"] += output_tokens

    # ------------------------------------------------------------ main loop
    def run(self, task: str, resume: dict | None = None) -> RunResult:
        """Run `task`. With `resume` (a state from miniagent.session.load_session) the earlier conversation is continued:
        `task` is added as a new user message and step numbers go on from where the session stopped."""
        base = resume["step"] if resume else 0
        self.cancel_event.clear()
        if resume:
            messages: list[dict] = list(resume["messages"]) + [{"role": "user", "content": task}]
            self.todos = list(resume.get("todos", []))
        else:
            messages = [{"role": "user", "content": task}]
        run_dir = self.trajectory.path.parent if (self.session and self.trajectory.path) else None
        self._recorder = rec = Recorder(
            run_dir, self.workspace.root, task=resume["task"] if resume else task, model=self.model,
            system_sha1=self._system_sha1(), step_index=resume["step_index"] if resume else None,
            todos_at=resume["todos_at"] if resume else None)
        executor = ToolExecutor(self.tools, confirm=self.confirm, trajectory=self.trajectory, root=self.workspace.root,
                                permissions=self.permissions, validate_args=self.validate_args, parallel=self.parallel_tools)
        self._log_start(task, resume is not None)
        stuck = StuckDetector(self.stuck_warn_after, self.stuck_stop_after) if self.stuck_warn_after > 0 else None
        spent = {"uncached": 0, "cached": 0, "out": 0}      # this loop's own model calls; sub-agent usage is added when checking

        in_tok = out_tok = 0
        last_input = resume.get("last_input", 0) if resume else 0
        floor: int | None = None        # input tokens measured on the first request after the latest compaction
        awaiting_floor = False
        final_text = ""
        status = "max_steps"
        step = completed = logged = base   # `completed`: last step whose turn finished; `logged`: last step whose model_call is in the trajectory
        try:
            for step in range(base + 1, base + self.max_steps + 1):
                self._check_cancel()
                if self.budget and (why := self._over_budget(spent)):
                    self.trajectory.log("budget", step=step, reason=why, **self._spent_summary(spent))
                    status = "budget"
                    step -= 1                     # this step never ran
                    break
                if last_input > self._compact_trigger(floor):
                    messages = self._compact(messages, trigger_tokens=last_input)
                    awaiting_floor = True
                    last_input = 0
                    rec.forget_positions()
                rec.begin_step(step, len(messages), self.todos)

                resp, calls = self._model_step(step, messages)
                logged = step
                self._check_cancel()
                last_input = resp.usage["input_tokens"]
                if awaiting_floor:
                    floor, awaiting_floor = last_input, False
                    if floor >= self.context_limit * compaction.TRIGGER_FRACTION:
                        self.trajectory.log("compaction_floor", step=step, input_tokens=floor,
                                            threshold=int(self.context_limit * compaction.TRIGGER_FRACTION),
                                            next_trigger=int(self._compact_trigger(floor)),
                                            note="compaction could not get below the threshold; next one waits for regrowth")
                in_tok += last_input
                out_tok += resp.usage["output_tokens"]
                cached = resp.usage.get("cache_hit_tokens", 0) or 0
                spent["cached"] += cached
                spent["uncached"] += max(last_input - cached, 0)
                spent["out"] += resp.usage["output_tokens"]
                messages.append(_assistant_message(resp))
                final_text = resp.text

                if resp.tool_calls:
                    if self.output_budget:
                        self.workspace.output_limit = output_limit_for(self.context_limit, last_input)
                    rec.checkpoint(step, calls, self.tools, self.trajectory)
                    results = executor.run(step, calls)
                    stop_stuck = False
                    if stuck:
                        stop_stuck = self._check_stuck(stuck, step, calls, results, executor.last_results)
                    if self.git is not None:
                        self._git_commit(step, calls)
                    messages.extend(results)
                    completed = step
                    rec.save(messages, step, last_input, self.todos)
                    self._check_cancel()
                    if stop_stuck:
                        status = "stuck"
                        break
                    continue
                status = "done" if resp.finish_reason == "stop" else resp.finish_reason
                rec.save(messages, step, last_input, self.todos)
                break
        except KeyboardInterrupt:
            # Ctrl-C: leave a session that --resume can continue from. A turn cut off in the middle has an assistant message whose
            # tool calls have no results; the API rejects such a history, so every unanswered call gets an explicit note.
            closed = _close_dangling_calls(messages)
            done_step = step if closed is not messages else completed     # notes were needed: the turn of `step` was cut in the middle
            done_step = max(done_step, logged)                            # a request that was answered is in the log even if its reply was dropped
            messages = closed
            rec.save(messages, done_step, last_input, self.todos)
            self.trajectory.log("run_end", status="interrupted", steps=done_step, input_tokens=in_tok, output_tokens=out_tok,
                                final_text=final_text)
            return RunResult("interrupted", done_step, final_text, in_tok, out_tok)
        except Exception as e:  # API failure etc.; keep the trajectory consistent
            self.trajectory.log("run_end", status="error", steps=step, error=repr(e),
                                input_tokens=in_tok, output_tokens=out_tok, final_text=final_text)
            return RunResult("error", step, final_text, in_tok, out_tok, repr(e))
        finally:
            if self.background is not None and not self.keep_background:
                self.background.kill_all()          # no job outlives the run

        in_tok += self.sub_usage["input"]
        out_tok += self.sub_usage["output"]       # sub-agent tokens are real spend: they count in the run totals
        cost = self.budget.cost(spent["uncached"] + self.sub_usage["input"], spent["cached"], spent["out"] + self.sub_usage["output"]) \
            if self.budget else None
        self.trajectory.log("run_end", status=status, steps=step, input_tokens=in_tok,
                            output_tokens=out_tok, final_text=final_text,
                            **({"cost_usd": round(cost, 6)} if cost is not None else {}),
                            **({"subagent_input_tokens": self.sub_usage["input"], "subagent_output_tokens": self.sub_usage["output"]}
                               if self.sub_usage["input"] or self.sub_usage["output"] else {}))
        return RunResult(status, step, final_text, in_tok, out_tok)

    def _check_stuck(self, stuck: StuckDetector, step: int, calls: list[dict], messages: list[dict], raw: list[dict]) -> bool:
        """Feed the turn's raw results to the detector, append notices to the tool `messages`, log events. True = stop the run."""
        stop = False
        for i, (call, r) in enumerate(zip(calls, raw)):
            ev = stuck.observe(call, r)
            if ev is None:
                continue
            self.trajectory.log("stuck", step=step, tool_call_id=call["id"], kind=ev.kind, count=ev.count, action=ev.action)
            messages[i]["content"] += stuck_notice(ev)
            stop = stop or ev.action == "stop"
        return stop

    def _spent_summary(self, spent: dict) -> dict:
        tokens = spent["uncached"] + spent["cached"] + spent["out"] + self.sub_usage["input"] + self.sub_usage["output"]
        cost = self.budget.cost(spent["uncached"] + self.sub_usage["input"], spent["cached"], spent["out"] + self.sub_usage["output"])
        return {"spent_tokens": tokens, **({"spent_usd": round(cost, 6)} if cost is not None else {})}

    def _over_budget(self, spent: dict) -> str | None:
        s = self._spent_summary(spent)
        return self.budget.exceeded(s["spent_tokens"], s.get("spent_usd"))

    def _git_commit(self, step: int, calls: list[dict]) -> None:
        summary = ", ".join(sorted({c["name"] for c in calls})) or "tools"
        try:
            made = self.git.commit_step(step, summary)
        except Exception as e:                           # a git problem must not end the run; it is recorded
            self.trajectory.log("git_commit", step=step, sha="", files=[], error=str(e)[:300])
            return
        if made:
            self.trajectory.log("git_commit", step=step, sha=made["sha"], files=made["files"])

    def _system_sha1(self) -> str:
        return hashlib.sha1(self.system.encode()).hexdigest()[:12]

    def _log_start(self, task: str, resumed: bool) -> None:
        opts = {k: True for k in ("require_read", "diagnostics") if getattr(self.workspace, k, False)}
        self.trajectory.log(
            "run_resume" if resumed else "run_start", schema_version=SCHEMA_VERSION, provenance=provenance(), task=task,
            model=self.model, meta=self.meta,
            config={"max_steps": self.max_steps, "context_limit": self.context_limit,
                    "keep_recent": self.keep_recent, "compact_regrowth": self.compact_regrowth, "max_tokens": self.max_tokens,
                    "temperature": self.temperature, "tools": sorted(self.tools),
                    **({"features": sorted(self.features)} if self.features else {}),
                    **({"workspace_options": opts} if opts else {}),
                    **({"permissions": self.permissions.to_dict()} if self.permissions else {}),
                    **({"output_budget": True} if self.output_budget else {}),
                    **({"stuck": {"warn_after": self.stuck_warn_after, "stop_after": self.stuck_stop_after}} if self.stuck_warn_after else {}),
                    **({"budget": {"max_cost_usd": self.budget.max_cost_usd, "max_total_tokens": self.budget.max_total_tokens,
                                   "price": self.budget.price}} if self.budget else {}),
                    "system_sha1": self._system_sha1()},
            # full text, so the exact model input can be rebuilt later (SFT export) without guessing from a hash
            system=self.system, tool_specs=[t.spec() for t in self.tools.values()])

    def _model_step(self, step: int, messages: list[dict]) -> tuple[ModelResponse, list[dict]]:
        """One model request; logs the `model_call` event. Returns the response and its parsed tool calls."""
        t0 = time.time()
        extra = {"on_text": self._guard(self.on_text)} if self.on_text is not None else {}
        resp: ModelResponse = self.client.chat(
            model=self.model, max_tokens=self.max_tokens, system=self.system,
            tools=[t.spec() for t in self.tools.values()], messages=messages, temperature=self.temperature, **extra)
        calls = [_parse_call(c) for c in resp.tool_calls]
        self.trajectory.log(
            "model_call", step=step, latency_s=round(time.time() - t0, 3),
            finish_reason=resp.finish_reason, text=resp.text, response_model=resp.model,
            # `extra` = provider fields echoed back in the history (Gemini thought signatures); kept so that a
            # replay rebuilds a history of the same size, which the compaction decisions depend on
            tool_calls=[{"id": c["id"], "name": c["name"], "input": c["input"], **({"extra": r.extra} if r.extra else {})}
                        for c, r in zip(calls, resp.tool_calls)],
            usage=resp.usage,
            **({"reasoning": resp.reasoning} if self.log_reasoning and resp.reasoning else {}),
        )
        return resp, calls

    # ----------------------------------------------------------- compaction
    def _compact_trigger(self, floor: int | None) -> float:
        return compaction.compact_trigger(self.context_limit, self.compact_regrowth, floor)

    def _compact(self, messages: list[dict], trigger_tokens: int | None = None) -> list[dict]:
        todo = f"\n\n[Current todo list]\n{render_todos(self.todos)}" if self.todos else ""
        return compaction.compact(messages, client=self.client, model=self.model, temperature=self.temperature,
                                  context_limit=self.context_limit, keep_recent=self.keep_recent, trajectory=self.trajectory,
                                  trigger_tokens=trigger_tokens, todo_note=todo)


# ------------------------------------------------------------------ helpers
def _parse_call(c) -> dict:
    try:
        args = json.loads(c.arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("expected a JSON object")
        return {"id": c.id, "name": c.name, "input": args}
    except ValueError as e:
        return {"id": c.id, "name": c.name, "input": {"_raw_arguments": c.arguments}, "parse_error": str(e)}


def _close_dangling_calls(messages: list[dict]) -> list[dict]:
    """If the last assistant message has tool calls without matching tool messages, add one per missing call."""
    last = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"), None)
    if last is None:
        return messages
    answered = {m.get("tool_call_id") for m in messages[last + 1:] if m["role"] == "tool"}
    missing = [c["id"] for c in messages[last].get("tool_calls") or [] if c["id"] not in answered]
    note = "interrupted by the user before this call finished; it may have run partly or not at all: check the files before relying on it"
    if not missing:
        return messages                                  # the same list: the caller can tell that nothing had to be added
    return messages + [{"role": "tool", "tool_call_id": cid, "content": note} for cid in missing]


def _assistant_message(resp: ModelResponse) -> dict:
    msg: dict = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}, **c.extra}
            for c in resp.tool_calls]
    return msg
