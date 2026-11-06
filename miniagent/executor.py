"""Running the tool calls of one model turn: argument validation, permission rules, confirmation, execution, logging, and the
concurrent execution of read-only calls. Separate from the agent loop so the loop only decides *what to ask the model*, and this
module decides *what a tool call is allowed to do and how it is run*."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .permissions import Decision, Permissions
from .sandbox import SandboxError
from .schema import signature, validate
from .tools import Tool, ToolError
from .trajectory import Trajectory

MAX_PARALLEL = 8


class ToolExecutor:
    def __init__(self, tools: dict[str, Tool], *, confirm: Callable[[str, dict], bool], trajectory: Trajectory, root: Path,
                 permissions: Permissions | None = None, validate_args: bool = True, parallel: bool = True):
        self.tools, self.confirm, self.trajectory, self.root = tools, confirm, trajectory, root
        self.permissions, self.validate_args, self.parallel = permissions, validate_args, parallel

    # ---------------------------------------------------------------- a turn
    def run(self, step: int, calls: list[dict]) -> list[dict]:
        """Run one turn's tool calls and return the `tool` messages in the order the model issued the calls. Consecutive read-only
        calls run concurrently; anything else (writes, bash, calls needing confirmation, bad calls) runs alone and in order, so it
        acts as a barrier. The `tool_result` log lines come out in the same order."""
        out: list[dict] = []
        i = 0
        while i < len(calls):
            j = i
            if self.parallel:
                while j < len(calls) and self._concurrent_ok(calls[j]):
                    j += 1
            if j - i >= 2:
                batch = calls[i:j]
                with ThreadPoolExecutor(max_workers=min(len(batch), MAX_PARALLEL)) as pool:
                    futures = [pool.submit(self.execute, c) for c in batch]
                    results = [f.result() for f in futures]           # SandboxError (infrastructure fault) re-raises here
                out += [self._log(step, c, r, parallel=len(batch)) for c, r in zip(batch, results)]
                i = j
            else:
                out.append(self._log(step, calls[i], self.execute(calls[i])))
                i += 1
        return out

    # ---------------------------------------------------------------- one call
    def _decision(self, call: dict) -> Decision:
        if self.permissions is None or not isinstance(call.get("input"), dict):
            return Decision("default")
        return self.permissions.decide(call["name"], call["input"], self.root)

    def needs_confirm(self, tool: Tool, call: dict) -> bool:
        action = self._decision(call).action
        return True if action == "ask" else False if action == "allow" else tool.needs_confirm

    def _concurrent_ok(self, call: dict) -> bool:
        tool = self.tools.get(call["name"])
        # a call with bad arguments may sit in the batch: execute() rejects it without running anything
        return bool(tool and tool.read_only and not call.get("parse_error") and not self.needs_confirm(tool, call))

    def _arg_errors(self, tool: Tool, args: dict) -> list[str]:
        return validate(args, tool.input_schema) if self.validate_args else []

    def execute(self, call: dict) -> dict:
        """Run one call and return {output, is_error, approved, duration_s[, permission, rule]}. Does not log, so it is safe to call
        from a worker thread for read-only tools."""
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
        elif self.needs_confirm(tool, call) and not self.confirm(name, args):
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

    def _log(self, step: int, call: dict, r: dict, parallel: int = 1) -> dict:
        self.trajectory.log("tool_result", step=step, tool_call_id=call["id"], name=call["name"], input=call["input"],
                            output=r["output"], is_error=r["is_error"], approved=r["approved"], duration_s=r["duration_s"],
                            **({"parallel": parallel} if parallel > 1 else {}),
                            **({"permission": r["permission"], "rule": r["rule"]} if "permission" in r else {}))
        return {"role": "tool", "tool_call_id": call["id"], "content": r["output"]}
