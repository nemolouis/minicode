"""Append-only JSONL trajectory log. One line per event.

This is the shared data format for the eval harness (reads run_end), the replay UI, the SFT export and the deterministic replay.
`SCHEMA_VERSION` is written into `run_start`; trajectories without it are version 1 (before the field existed) and are only
checked loosely. `validate_events` is the executable definition of version 2: every producer and consumer is tested against it.

Events (all carry `ts` and `event`):
  run_start    schema_version, provenance{git_sha, git_dirty, python, ...}, task, model, config{..., tools, features?, permissions?},
               system, tool_specs, meta
  run_resume   same fields as run_start, appended to the same file when a saved session is continued
  model_call   step, latency_s, finish_reason, text, tool_calls[{id,name,input,extra?}],
               usage{input_tokens,output_tokens,cache_hit_tokens}
  tool_result  step, tool_call_id, name, input, output, is_error, approved, duration_s,
               parallel? (size of the concurrent batch, only if > 1), permission? ("allow"|"ask"|"deny") + rule?
  compaction   kind ("elide"|"summary"|"summary_failed"), est_tokens_before, est_tokens_after, summary?
  compaction_floor  step, input_tokens, threshold, next_trigger
  subagent     name, index, question, status, steps, input_tokens, output_tokens, final_text  (its own trajectory is in explore-N/)
  checkpoint   step, files, skipped_large, complete  (a workspace copy taken before the step's tools ran)
  rewind       to_step, restored, removed  (appended by `miniagent --rewind`; the steps from to_step on, above this line, are an
               abandoned branch: later steps are numbered again from to_step, so a reader must not merge the two)
  run_end      status, steps, input_tokens, output_tokens (sub-agent tokens included), final_text

The message list is append-only apart from `compaction`, so a reader can rebuild the exact model input at any step by replaying
events in order.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SCHEMA_VERSION = 2


class Trajectory:
    def __init__(self, run_dir: str | Path | None):
        self.path: Path | None = None
        if run_dir is not None:
            d = Path(run_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / "trajectory.jsonl"

    def log(self, event: str, **data) -> None:
        if self.path is None:
            return
        rec = {"ts": round(time.time(), 3), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ validation
_NUM = (int, float)
_REQUIRED: dict[str, dict[str, object]] = {
    "run_start": {"schema_version": int, "provenance": dict, "task": str, "model": str, "config": dict, "system": str, "tool_specs": list},
    "run_resume": {"schema_version": int, "task": str, "model": str, "config": dict},
    "model_call": {"step": int, "latency_s": _NUM, "finish_reason": str, "text": str, "tool_calls": list, "usage": dict},
    "tool_result": {"step": int, "tool_call_id": str, "name": str, "input": dict, "output": str, "is_error": bool, "approved": bool,
                    "duration_s": _NUM},
    "compaction": {"kind": str, "est_tokens_before": int, "est_tokens_after": int},
    "compaction_floor": {"step": int, "input_tokens": int},
    "subagent": {"name": str, "index": int, "question": str, "status": str, "steps": int, "input_tokens": int, "output_tokens": int,
                 "final_text": str},
    "checkpoint": {"step": int, "files": int, "skipped_large": int, "complete": bool},
    "rewind": {"to_step": int, "restored": int, "removed": int},
    "run_end": {"status": str, "steps": int, "input_tokens": int, "output_tokens": int, "final_text": str},
}
_OPTIONAL = {"tool_result": {"parallel": int, "permission": str, "rule": str}}


def validate_events(events: list[dict]) -> list[str]:
    """Problems found in one trajectory (a list of event dicts), as readable strings; [] means valid.

    Version 1 trajectories (no `schema_version` in run_start) are checked only for `event`/`ts` and for starting with run_start."""
    errs: list[str] = []
    if not events:
        return ["empty trajectory"]
    if events[0].get("event") != "run_start":
        return [f"first event is {events[0].get('event')!r}, expected run_start"]
    legacy = "schema_version" not in events[0]
    if not legacy and events[0]["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {events[0]['schema_version']!r} is not the current {SCHEMA_VERSION}")
    last_step, calls_by_step = 0, {}
    for i, e in enumerate(events):
        kind, where = e.get("event"), f"event {i} ({e.get('event')})"
        if not isinstance(kind, str) or not isinstance(e.get("ts"), _NUM):
            errs.append(f"{where}: needs a string `event` and a numeric `ts`")
            continue
        if legacy:
            continue
        spec = _REQUIRED.get(kind)
        if spec is None:
            errs.append(f"{where}: unknown event type")
            continue
        for key, typ in spec.items():
            if key not in e:
                errs.append(f"{where}: missing `{key}`")
            elif not isinstance(e[key], typ) or (typ is int and isinstance(e[key], bool)):
                errs.append(f"{where}: `{key}` should be {getattr(typ, '__name__', typ)}, got {type(e[key]).__name__}")
        for key, typ in _OPTIONAL.get(kind, {}).items():
            if key in e and not isinstance(e[key], typ):
                errs.append(f"{where}: `{key}` should be {typ.__name__}")
        if kind == "rewind" and isinstance(e.get("to_step"), int):
            last_step = e["to_step"] - 1                       # the steps from to_step on were abandoned; numbering restarts there
            calls_by_step = {k: v for k, v in calls_by_step.items() if k < e["to_step"]}
        if kind == "model_call" and isinstance(e.get("step"), int):
            if e["step"] <= last_step:
                errs.append(f"{where}: step {e['step']} does not increase (previous {last_step})")
            last_step = e["step"]
            tcs = e.get("tool_calls") if isinstance(e.get("tool_calls"), list) else []
            ids = []
            for tc in tcs:
                if not (isinstance(tc, dict) and isinstance(tc.get("id"), str) and isinstance(tc.get("name"), str)
                        and isinstance(tc.get("input"), dict)):
                    errs.append(f"{where}: tool_calls entries need id, name and an input object")
                else:
                    ids.append(tc["id"])
            calls_by_step[e["step"]] = ids
            if not isinstance(e.get("usage"), dict) or not all(k in e["usage"] for k in ("input_tokens", "output_tokens")):
                errs.append(f"{where}: usage needs input_tokens and output_tokens")
        if kind == "tool_result" and isinstance(e.get("step"), int):
            if e["step"] not in calls_by_step:
                errs.append(f"{where}: no model_call for step {e['step']}")
            elif e.get("tool_call_id") not in calls_by_step[e["step"]]:
                errs.append(f"{where}: tool_call_id {e.get('tool_call_id')!r} was not requested in step {e['step']}")
            if e.get("permission") not in (None, "allow", "ask", "deny"):
                errs.append(f"{where}: permission must be allow, ask or deny")
            if "permission" in e and "rule" not in e:
                errs.append(f"{where}: `permission` without `rule`")
            if isinstance(e.get("parallel"), int) and e["parallel"] < 2:
                errs.append(f"{where}: `parallel` is only written for batches of 2 or more")
        if kind == "compaction" and e.get("kind") not in ("elide", "summary", "summary_failed"):
            errs.append(f"{where}: unknown compaction kind {e.get('kind')!r}")
    return errs


def read_events(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
