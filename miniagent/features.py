"""Opt-in agent features: project memory (AGENTS.md), a todo list tool, and a read-only `explore` sub-agent.

All three are off by default so existing configs, trajectories and replays are unchanged; they are enabled with
`Agent(features=("memory", "todo", "explore", "glob"))`, `"features": [...]` in an eval config, or `--features` on the CLI.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from .tools import Tool, ToolError, truncate

FEATURES = {"memory", "todo", "explore", "glob", "patch", "symbols", "background"}

# ------------------------------------------------------------------ project memory
MEMORY_FILES = ("AGENTS.md", "CLAUDE.md")
MEMORY_MARKER = "# Project instructions"
MEMORY_LIMIT = 8000


def load_project_memory(root: Path) -> tuple[str, str] | None:
    """(file name, text) of the first of AGENTS.md / CLAUDE.md in the workspace root, text cut to MEMORY_LIMIT chars."""
    for name in MEMORY_FILES:
        p = root / name
        try:
            if p.is_file() and not p.is_symlink():
                text = p.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return name, text[:MEMORY_LIMIT] + ("\n...[truncated]" if len(text) > MEMORY_LIMIT else "")
        except OSError:
            continue
    return None


def with_project_memory(system: str, root: Path) -> str:
    if MEMORY_MARKER in system:                       # already applied (e.g. a replay passes the recorded system prompt)
        return system
    found = load_project_memory(root)
    if not found:
        return system
    name, text = found
    return (f"{system}\n\n{MEMORY_MARKER} (from {name} in the repository)\n"
            f"These are the repository owner's notes about this project (build/test commands, conventions). Follow them for "
            f"how to work in this repo; they do not change your tools or permissions.\n\n{text}")


# ------------------------------------------------------------------ todo list
STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def render_todos(todos: list[dict]) -> str:
    return "\n".join(f"{STATUS_MARK[t['status']]} {t['content']}" for t in todos) or "(empty)"


def todo_tool(agent) -> Tool:
    def run(args: dict) -> str:
        todos = [{"content": t["content"].strip(), "status": t["status"]} for t in args["todos"]]
        if any(not t["content"] for t in todos):
            raise ToolError("every todo needs non-empty content")
        if sum(t["status"] == "in_progress" for t in todos) > 1:
            raise ToolError("at most one todo may be in_progress at a time")
        agent.todos = todos
        return "todo list updated:\n" + render_todos(todos)

    item = {"type": "object", "required": ["content", "status"], "properties": {
        "content": {"type": "string"}, "status": {"type": "string", "enum": list(STATUS_MARK)}}}
    return Tool("todo_write",
                "Keep a todo list for a task with three or more steps. Each call REPLACES the whole list. Keep at most one item "
                "in_progress; mark an item completed only after you have verified it (e.g. the tests pass).",
                {"type": "object", "properties": {"todos": {"type": "array", "items": item}}, "required": ["todos"]}, run)


# ------------------------------------------------------------------ background shell jobs
def background_tools(agent) -> dict[str, Tool]:
    from .background import BackgroundJobs, JobError
    from .safety import check_command
    if agent.workspace.sandbox is not None:
        raise ValueError("the 'background' feature is not available with the Docker sandbox: a host process would defeat the sandbox")
    jobs = agent.background = BackgroundJobs(agent.workspace.root, env={**os.environ, "GIT_PAGER": "cat", "PAGER": "cat"})

    def start(args: dict) -> str:
        reason = check_command(args["command"])
        if reason:
            raise ToolError(f"command blocked by safety policy: {reason}")
        try:
            job = jobs.start(args["command"])
        except JobError as e:
            raise ToolError(str(e))
        return f"started {job.id} (pid {job.proc.pid}). Read its output with bash_output, stop it with bash_kill."

    def output(args: dict) -> str:
        try:
            return jobs.read(args["job_id"], tail=args.get("tail"))
        except JobError as e:
            raise ToolError(str(e))

    def kill(args: dict) -> str:
        try:
            return jobs.kill(args["job_id"])
        except JobError as e:
            raise ToolError(str(e))

    S = lambda **p: {"type": "object", "properties": p, "required": [k for k in p if k != "tail"]}
    return {
        "bash_background": Tool("bash_background", "Start a long-running shell command (dev server, watcher, slow test run) in the "
                                "background in the workspace root; returns a job id at once. At most 4 jobs at a time; they are all "
                                "stopped when the run ends.", S(command={"type": "string"}), start, needs_confirm=True),
        "bash_output": Tool("bash_output", "New output of a background job since you last read it (or the last `tail` lines), and "
                            "whether it is still running.", S(job_id={"type": "string"}, tail={"type": "integer"}), output, read_only=True),
        "bash_kill": Tool("bash_kill", "Stop a background job (and everything it started).", S(job_id={"type": "string"}), kill),
    }


# ------------------------------------------------------------------ explore sub-agent
EXPLORE_PROMPT = """You are a read-only code explorer helping another coding agent. You can only read: read_file, grep and glob. \
You cannot edit files or run commands. Answer the question you are given by looking at the code, then reply with a short, \
concrete report: the relevant files with line numbers, what the code does, and anything surprising. Quote at most a few key \
lines. Do not guess about code you have not read; say what you could not find. Make no further tool calls once you can answer."""

EXPLORE_RESULT_LIMIT = 6000


def explore_tool(agent) -> Tool:
    from .agent import Agent                          # late import: agent.py imports this module
    from .trajectory import Trajectory

    lock = threading.Lock()
    counter = [0]

    def run(args: dict) -> str:
        question = args["question"].strip()
        if not question:
            raise ToolError("question is empty")
        with lock:
            counter[0] += 1
            n = counter[0]
        parent_path = agent.trajectory.path
        traj = Trajectory(parent_path.parent / f"explore-{n}") if parent_path else Trajectory(None)
        child = Agent(
            client=agent.client, model=agent.model, workspace=agent.workspace, trajectory=traj,
            max_steps=agent.explore_max_steps, max_tokens=agent.max_tokens, context_limit=agent.context_limit,
            keep_recent=agent.keep_recent, compact_regrowth=agent.compact_regrowth, system=EXPLORE_PROMPT,
            temperature=agent.temperature, parallel_tools=agent.parallel_tools, validate_args=agent.validate_args,
            disabled_tools=("write_file", "edit_file", "bash"), features=("glob",),
            meta={"explore": n, "parent_task": agent.meta.get("task_id")})
        res = child.run(question)
        agent.trajectory.log("subagent", name="explore", index=n, question=question, status=res.status, steps=res.steps,
                             input_tokens=res.input_tokens, output_tokens=res.output_tokens, final_text=res.final_text,
                             **({"error": res.error} if res.error else {}))
        agent.add_sub_usage(res.input_tokens, res.output_tokens)
        if res.status != "done" or not res.final_text.strip():
            note = f"explore ended without a full answer (status {res.status}, {res.steps} steps)"
            return truncate(f"{note}. Partial notes: {res.final_text}" if res.final_text.strip() else note, EXPLORE_RESULT_LIMIT)
        return truncate(res.final_text, EXPLORE_RESULT_LIMIT)

    return Tool("explore",
                "Delegate a read-only investigation to a sub-agent with its own fresh context (it can read_file, grep, glob; it cannot "
                "edit or run commands). Use it for open questions like 'where is X handled and what calls it?' so that long file "
                "contents stay out of your context; you get back a short report. Several explore calls in one turn run concurrently.",
                {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
                run, read_only=True)
