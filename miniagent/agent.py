"""The agent loop, written directly against the chat-completions tool-calling protocol."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .client import ModelResponse
from .features import FEATURES, explore_tool, render_todos, todo_tool, with_project_memory
from .sandbox import SandboxError
from . import session as sess
from .permissions import Permissions
from .schema import signature, validate
from .tools import Tool, ToolError, Workspace, build_tools, glob_tool
from .trajectory import Trajectory

SYSTEM_PROMPT = """You are a coding agent working inside a git-style repository. \
Your job is to fix the bug or complete the task the user describes.

Work method: explore first (grep, read_file), form a hypothesis, make the smallest \
edit that fixes the root cause, then run the project's tests with bash to verify. \
Do not edit tests to make them pass. Paths are relative to the workspace root, and bash already runs there: do not cd elsewhere. \
When you are done, reply with a short summary of what you changed and the test result, \
and make no further tool calls."""

SUMMARY_PROMPT = """Below is the transcript of an agent working on a coding task. Write a \
dense progress summary the agent can continue from: files inspected and what was learned, \
edits made (file + what changed), commands run and their outcomes, current hypothesis, \
and what remains. Do not invent anything."""


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
        if "todo" in self.features:
            tools["todo_write"] = todo_tool(self)
        if "explore" in self.features:
            tools["explore"] = explore_tool(self)
        self.tools: dict[str, Tool] = {n: t for n, t in tools.items() if n not in self.disabled_tools}

    def add_sub_usage(self, input_tokens: int, output_tokens: int) -> None:
        with self._usage_lock:
            self.sub_usage["input"] += input_tokens
            self.sub_usage["output"] += output_tokens

    # ------------------------------------------------------------ main loop
    def run(self, task: str, resume: dict | None = None) -> RunResult:
        """Run `task`. With `resume` (a state from miniagent.session.load_session) the earlier conversation is continued:
        `task` is added as a new user message and step numbers go on from where the session stopped."""
        base = resume["step"] if resume else 0
        self._step_index: dict[int, int] = dict(resume["step_index"]) if resume else {}
        self._todos_at: dict[int, list] = dict(resume["todos_at"]) if resume else {}
        if resume:
            messages: list[dict] = list(resume["messages"]) + [{"role": "user", "content": task}]
            self.todos = list(resume.get("todos", []))
        else:
            messages = [{"role": "user", "content": task}]
        self._session_task = resume["task"] if resume else task
        self.trajectory.log(
            "run_resume" if resume else "run_start", task=task, model=self.model, meta=self.meta,
            config={"max_steps": self.max_steps, "context_limit": self.context_limit,
                    "keep_recent": self.keep_recent, "compact_regrowth": self.compact_regrowth, "max_tokens": self.max_tokens,
                    "temperature": self.temperature, "tools": sorted(self.tools),
                    **({"features": sorted(self.features)} if self.features else {}),
                    **({"workspace_options": opts} if (opts := {k: True for k in ("require_read", "diagnostics")
                                                                 if getattr(self.workspace, k, False)}) else {}),
                    **({"permissions": self.permissions.to_dict()} if self.permissions else {}),
                    "system_sha1": hashlib.sha1(self.system.encode()).hexdigest()[:12]},
            # full text, so the exact model input can be rebuilt later (SFT export) without guessing from a hash
            system=self.system, tool_specs=[t.spec() for t in self.tools.values()])
        in_tok = out_tok = 0
        last_input = resume.get("last_input", 0) if resume else 0
        floor: int | None = None        # input tokens measured on the first request after the latest compaction
        awaiting_floor = False
        final_text = ""
        status = "max_steps"
        step = base
        try:
            for step in range(base + 1, base + self.max_steps + 1):
                if last_input > self._compact_trigger(floor):
                    messages = self._compact(messages, trigger_tokens=last_input)
                    awaiting_floor = True
                    last_input = 0
                    self._step_index.clear()          # the old positions no longer exist: only later steps can be rewound to
                    self._todos_at.clear()
                self._step_index[step] = len(messages)
                self._todos_at[step] = [dict(t) for t in self.todos]

                t0 = time.time()
                resp: ModelResponse = self.client.chat(
                    model=self.model, max_tokens=self.max_tokens, system=self.system,
                    tools=[t.spec() for t in self.tools.values()], messages=messages,
                    temperature=self.temperature,
                )
                calls = [_parse_call(c) for c in resp.tool_calls]
                last_input = resp.usage["input_tokens"]
                if awaiting_floor:
                    floor, awaiting_floor = last_input, False
                    if floor >= self.context_limit * 0.7:
                        self.trajectory.log("compaction_floor", step=step, input_tokens=floor,
                                            threshold=int(self.context_limit * 0.7), next_trigger=int(self._compact_trigger(floor)),
                                            note="compaction could not get below the threshold; next one waits for regrowth")
                in_tok += last_input
                out_tok += resp.usage["output_tokens"]
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
                messages.append(_assistant_message(resp))
                final_text = resp.text

                if resp.tool_calls:
                    self._checkpoint(step, calls)
                    messages.extend(self._run_tools(step, calls))
                    self._save_session(messages, step, last_input)
                    continue
                status = "done" if resp.finish_reason == "stop" else resp.finish_reason
                self._save_session(messages, step, last_input)
                break
        except Exception as e:  # API failure etc.; keep the trajectory consistent
            self.trajectory.log("run_end", status="error", steps=step, error=repr(e),
                                input_tokens=in_tok, output_tokens=out_tok, final_text=final_text)
            return RunResult("error", step, final_text, in_tok, out_tok, repr(e))

        in_tok += self.sub_usage["input"]
        out_tok += self.sub_usage["output"]       # sub-agent tokens are real spend: they count in the run totals
        self.trajectory.log("run_end", status=status, steps=step, input_tokens=in_tok,
                            output_tokens=out_tok, final_text=final_text,
                            **({"subagent_input_tokens": self.sub_usage["input"], "subagent_output_tokens": self.sub_usage["output"]}
                               if self.sub_usage["input"] or self.sub_usage["output"] else {}))
        return RunResult(status, step, final_text, in_tok, out_tok)

    # -------------------------------------------------------------- session
    def _run_dir(self):
        return self.trajectory.path.parent if (self.session and self.trajectory.path) else None

    def _checkpoint(self, step: int, calls: list[dict]) -> None:
        run_dir = self._run_dir()
        if run_dir is None:
            return
        if all((t := self.tools.get(c["name"])) is not None and t.read_only for c in calls):
            return                                 # nothing here can change files
        info = sess.snapshot(self.workspace.root, run_dir / "checkpoints" / f"step-{step}",
                             exclude=sess.exclusions(self.workspace.root, run_dir))
        self.trajectory.log("checkpoint", step=step, **info)

    def _save_session(self, messages: list[dict], step: int, last_input: int) -> None:
        run_dir = self._run_dir()
        if run_dir is None:
            return
        sess.save_session(run_dir, {
            "task": self._session_task, "model": self.model, "workspace": str(self.workspace.root),
            "system_sha1": hashlib.sha1(self.system.encode()).hexdigest()[:12], "messages": messages, "step": step,
            "step_index": self._step_index, "todos": self.todos, "todos_at": self._todos_at, "last_input": last_input})

    # ---------------------------------------------------------------- tools
    def _run_tools(self, step: int, calls: list[dict]) -> list[dict]:
        """Run one turn's tool calls. Consecutive read-only calls run concurrently; anything else (writes, bash, calls
        needing confirmation, bad calls) runs alone and in order, so it acts as a barrier. The results and the
        `tool_result` log lines always come out in the order the model issued the calls."""
        out: list[dict] = []
        i = 0
        while i < len(calls):
            j = i
            if self.parallel_tools:
                while j < len(calls) and self._concurrent_ok(calls[j]):
                    j += 1
            if j - i >= 2:
                batch = calls[i:j]
                with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as pool:
                    futures = [pool.submit(self._execute, c) for c in batch]
                    results = [f.result() for f in futures]           # SandboxError (infrastructure fault) re-raises here
                for c, r in zip(batch, results):
                    out.append(self._log_tool(step, c, r, parallel=len(batch)))
                i = j
            else:
                out.append(self._run_tool(step, calls[i]))
                i += 1
        return out

    def _concurrent_ok(self, call: dict) -> bool:
        tool = self.tools.get(call["name"])
        # a call with bad arguments may sit in the batch: _execute rejects it without running anything
        return bool(tool and tool.read_only and not call.get("parse_error") and not self._needs_confirm(tool, call))

    def _decision(self, call: dict):
        from .permissions import Decision
        if self.permissions is None or not isinstance(call.get("input"), dict):
            return Decision("default")
        return self.permissions.decide(call["name"], call["input"], self.workspace.root)

    def _needs_confirm(self, tool: Tool, call: dict) -> bool:
        action = self._decision(call).action
        return True if action == "ask" else False if action == "allow" else tool.needs_confirm

    def _arg_errors(self, tool: Tool, args: dict) -> list[str]:
        return validate(args, tool.input_schema) if self.validate_args else []

    def _run_tool(self, step: int, call: dict) -> dict:
        return self._log_tool(step, call, self._execute(call))

    def _execute(self, call: dict) -> dict:
        """Run one call and return {output, is_error, approved, duration_s}. Does not log (safe to call from a worker thread
        for read-only tools)."""
        name, args = call["name"], call["input"]
        t0 = time.time()
        approved, is_error = True, False
        tool = self.tools.get(name)
        if tool is None:
            out, is_error = f"unknown tool: {name}", True
        elif call.get("parse_error"):
            out, is_error = f"error: arguments are not valid JSON ({call['parse_error']})", True
        elif errs := self._arg_errors(tool, args):
            out, is_error = f"error: invalid arguments for {name}: {'; '.join(errs)}. Expected {signature(tool.input_schema)}", True
        elif (decision := self._decision(call)).action == "deny":
            approved, is_error = False, True
            out = f"denied by permission rule {decision.rule!r}; this action is not allowed, try a different approach"
        else:
            if self._needs_confirm(tool, call) and not self.confirm(name, args):
                approved, is_error, out = False, True, "user denied this action; try a different approach or explain why it is needed"
            else:
                try:
                    out = tool.func(args)
                except ToolError as e:
                    out, is_error = f"error: {e}", True
                except SandboxError:
                    raise          # infrastructure fault: ends the run as an error, is not shown to the model as a tool result
                except (KeyError, TypeError, ValueError) as e:
                    out, is_error = f"error: bad arguments ({e!r})", True
                except Exception as e:  # tool bug must not kill the run
                    out, is_error = f"error: {type(e).__name__}: {e}", True
        res = {"output": out, "is_error": is_error, "approved": approved, "duration_s": round(time.time() - t0, 3)}
        if self.permissions is not None and tool is not None and (d := self._decision(call)).action != "default":
            res["permission"], res["rule"] = d.action, d.rule
        return res

    def _log_tool(self, step: int, call: dict, r: dict, parallel: int = 1) -> dict:
        self.trajectory.log("tool_result", step=step, tool_call_id=call["id"], name=call["name"], input=call["input"],
                            output=r["output"], is_error=r["is_error"], approved=r["approved"], duration_s=r["duration_s"],
                            **({"parallel": parallel} if parallel > 1 else {}),
                            **({"permission": r["permission"], "rule": r["rule"]} if "permission" in r else {}))
        return {"role": "tool", "tool_call_id": call["id"], "content": r["output"]}

    # ----------------------------------------------------------- compaction
    def _compact_trigger(self, floor: int | None) -> float:
        """Input-token count above which the history is compacted."""
        base = self.context_limit * 0.7
        if floor is None or floor < base:        # no compaction yet, or the last one worked: original behaviour
            return base
        return floor * (1 + self.compact_regrowth)

    def _compact(self, messages: list[dict], trigger_tokens: int | None = None) -> list[dict]:
        before = _estimate_tokens(messages)
        # Stage 1: elide old tool outputs (cheap, lossless for the structure).
        cutoff = max(len(messages) - self.keep_recent, 1)
        out = [m for m in messages]
        for i in range(1, cutoff):
            m = out[i]
            if m["role"] == "tool" and len(m["content"]) > 200:
                out[i] = {**m, "content": f"[elided {len(m['content'])} chars]"}
        after = _estimate_tokens(out)
        self.trajectory.log("compaction", kind="elide", est_tokens_before=before, est_tokens_after=after,
                            trigger_input_tokens=trigger_tokens)
        if after <= self.context_limit * 0.5:
            return out

        # Stage 2: summarise the middle. Tail must start at an assistant message so
        # tool_calls/tool-result pairs stay intact and roles keep alternating.
        i = max(len(out) - self.keep_recent, 1)
        while i < len(out) and out[i]["role"] != "assistant":
            i += 1
        if i >= len(out) or i <= 1:
            return out
        transcript = _render(messages[1:i])  # un-elided: elision must not starve the summary
        resp = self.client.chat(
            model=self.model, max_tokens=1500, temperature=self.temperature,
            messages=[{"role": "user", "content": f"{SUMMARY_PROMPT}\n\n<task>\n{out[0]['content']}\n</task>\n\n<transcript>\n{transcript}\n</transcript>"}],
        )
        summary = resp.text
        if not summary.strip():
            # An empty summary (e.g. a thinking model used the whole token budget) would silently delete the middle of the
            # history. Keep it, only the elision of old tool outputs stands, and say so in the trajectory.
            self.trajectory.log("compaction", kind="summary_failed", est_tokens_before=after, est_tokens_after=after,
                                trigger_input_tokens=trigger_tokens, finish_reason=resp.finish_reason)
            return out
        todo = f"\n\n[Current todo list]\n{render_todos(self.todos)}" if self.todos else ""
        head = {"role": "user", "content": f"{out[0]['content']}\n\n[Progress so far, summarised from earlier steps]\n{summary}{todo}"}
        new = [head] + out[i:]
        self.trajectory.log("compaction", kind="summary", est_tokens_before=after,
                            est_tokens_after=_estimate_tokens(new), summary=summary, trigger_input_tokens=trigger_tokens)
        return new


# ------------------------------------------------------------------ helpers
def _parse_call(c) -> dict:
    try:
        args = json.loads(c.arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("expected a JSON object")
        return {"id": c.id, "name": c.name, "input": args}
    except ValueError as e:
        return {"id": c.id, "name": c.name, "input": {"_raw_arguments": c.arguments}, "parse_error": str(e)}


def _assistant_message(resp: ModelResponse) -> dict:
    msg: dict = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}, **c.extra}
            for c in resp.tool_calls]
    return msg


def _estimate_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False)) // 3


def _render(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if m["role"] == "tool":
            parts.append(f"[result] {str(m['content'])[:500]}")
            continue
        if m.get("content"):
            parts.append(f"[{m['role']}] {m['content']}")
        for c in m.get("tool_calls") or []:
            f = c["function"]
            parts.append(f"[call {f['name']}] {f['arguments'][:500]}")
    return "\n".join(parts)
