#!/usr/bin/env python3
"""Regenerate tests/golden/*: small scripted trajectories on real tasks, used as a regression test for the agent loop.

The model is scripted (it applies the task's reference solution), so there is no third-party model output in the repository. Each
trajectory is replayed through the CURRENT loop by tests/test_golden_replay.py: if a change to the loop, the tools, the compaction
or the permission/validation layers alters what happens for the same model answers, the replay diverges and the test fails.

Regenerate only on purpose (a deliberate behaviour change): `python scripts/make_golden.py`, then review `git diff tests/golden`.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from miniagent.agent import Agent                                    # noqa: E402
from miniagent.client import ModelResponse, ToolCall                 # noqa: E402
from miniagent.eval.grade import copy_repo                           # noqa: E402
from miniagent.eval.tasks import load_tasks                          # noqa: E402
from miniagent.tools import Workspace                                # noqa: E402
from miniagent.trajectory import Trajectory                          # noqa: E402

OUT = ROOT / "tests" / "golden"


def call(cid, name, **inp):
    return ToolCall(cid, name, json.dumps(inp))


class Scripted:
    """(content, input_tokens) per main step; a request without a system prompt is a compaction summary."""
    def __init__(self, script, summary="Summary: inspected the files and made the fix; the tests still need to be run."):
        self.script, self.summary = list(script), summary

    def chat(self, *, system=None, **kw):
        if system is None:
            return ModelResponse(self.summary, [], "stop", {"input_tokens": 50, "output_tokens": 20, "cache_hit_tokens": 0})
        content, tok = self.script.pop(0)
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 30, "cache_hit_tokens": 0}, "scripted-model")


def record(name: str, task_id: str, script, **agent_kw):
    task = next(t for t in load_tasks(ROOT / "tasks") if t.id == task_id)
    with tempfile.TemporaryDirectory(prefix=f"mini-{task.id}-") as tmp:                  # same layout as the eval runner and the replay
        work = Path(tmp) / "repo"
        copy_repo(task.repo, work)
        dest = OUT / name
        shutil.rmtree(dest, ignore_errors=True)
        agent = Agent(Scripted(script), "scripted-model", Workspace(work), trajectory=Trajectory(dest),
                      meta={"task_id": task.id, "task_hash": task.hash, "golden": name}, **agent_kw)
        res = agent.run(task.prompt)
    print(f"{name}: {res.status}, {res.steps} steps")


def solution(task_id: str) -> dict[str, str]:
    d = ROOT / "tasks" / task_id / "solution"
    return {p.relative_to(d).as_posix(): p.read_text() for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def main() -> None:
    # 1. the straight path: look around (concurrent reads), write the reference solution, run the tests, finish
    # (verification commands print a computed value instead of running pytest: the recorded output must not depend on the pytest version)
    t = "01-moving-average"
    (f, content), = solution(t).items()
    record("solve-01", t, [
        ([call("a", "grep", pattern="def moving_average"), call("b", "read_file", path=f), call("c", "read_file", path="test_stats.py")], 900),
        ([call("d", "write_file", path=f, content=content)], 1500),
        ([call("e", "bash", command="python -c \"from stats import moving_average as m; print(m([1, 2, 3, 4], 2))\"")], 1700),
        (["Fixed moving_average; the output is now correct."], 1800)])

    # 2. mistakes and recovery: unknown tool, missing argument, bad JSON, an edit that does not match, a blocked command
    t = "05-discount-rounding"
    (f, content), = solution(t).items()
    record("errors-05", t, [
        ([call("a", "read_file", path=f), call("b", "no_such_tool", x=1), call("c", "read_file")], 800),
        ([ToolCall("d", "grep", "{not json"), call("e", "edit_file", path=f, old_string="this text is not in the file", new_string="x"),
          call("f", "bash", command="sudo ls")], 1200),
        ([call("g", "write_file", path=f, content=content)], 1500),
        ([call("h", "bash", command="python -c \"from money import apply_percent as a; print(a(1999, 15))\"")], 1700),
        (["Done: fixed the rounding."], 1800)])

    # 3. context compaction, both kinds: one elision that is enough, later one that needs a summary (the recent window holds a
    #    huge output), and the loop keeps going afterwards. The scripted input-token counts decide when compaction triggers.
    t = "03-parse-duration"
    (f, content), = solution(t).items()
    big = "python -c \"print('x' * 9000)\""
    huge = "python -c \"print('y' * 18000)\""            # stays verbatim in the recent window, so elision alone is not enough
    record("compaction-03", t, [
        ([call("a", "read_file", path=f)], 800),
        ([call("b", "bash", command=big)], 1500),
        ([call("c", "bash", command=big)], 9000),            # above 70% of the context limit: the next step compacts first (elision is enough)
        ([call("d", "bash", command=big)], 3000),
        ([call("e", "write_file", path=f, content=content)], 3300),
        ([call("f", "bash", command=huge)], 9500),           # above the limit again, and the huge output is in the recent window: summary
        ([call("g", "bash", command="python -c \"from duration import parse_duration as p; print(p('1h30m'))\"")], 3600),
        (["Fixed after compaction."], 3700)],
        context_limit=10_000, keep_recent=2)

    # 4. never finishes: ends by running out of steps
    t = "01-moving-average"
    (f, _), = solution(t).items()
    record("max-steps-01", t, [([call(str(i), "grep", pattern="moving")], 500 + i) for i in range(4)], max_steps=4)


if __name__ == "__main__":
    main()
