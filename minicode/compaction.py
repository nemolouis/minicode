"""Context compaction: when the history gets too long, old tool outputs are elided and, if that is not enough, the middle of the
conversation is replaced by a model-written summary. Pure functions over the message list; the Agent decides *when* (see
`compact_trigger`) and logs what happened through the trajectory passed in."""
from __future__ import annotations

import json

SUMMARY_PROMPT = """Below is the transcript of an agent working on a coding task. Write a \
dense progress summary the agent can continue from: files inspected and what was learned, \
edits made (file + what changed), commands run and their outcomes, current hypothesis, \
and what remains. Do not invent anything."""

ELIDE_MIN_CHARS = 200          # tool outputs shorter than this are never elided
ELIDE_ENOUGH = 0.5             # elision alone is enough if the estimate drops to this fraction of the context limit
TRIGGER_FRACTION = 0.7         # compact when the last request had more input tokens than this fraction of the limit


def estimate_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False)) // 3


def render(messages: list[dict]) -> str:
    """The transcript handed to the summariser (tool outputs cut to 500 chars, call arguments too)."""
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


def compact_trigger(context_limit: int, regrowth: float, floor: int | None) -> float:
    """Input-token count above which the history is compacted. `floor` is the input size measured on the first request after
    the latest compaction (None before any). If that compaction could not get below the base threshold, compacting again at once
    cannot help, so the next one waits until the input has grown by `regrowth` (1.0 = may double)."""
    base = context_limit * TRIGGER_FRACTION
    if floor is None or floor < base:        # no compaction yet, or the last one worked: original behaviour
        return base
    return floor * (1 + regrowth)


def compact(messages: list[dict], *, client, model: str, temperature: float | None, context_limit: int, keep_recent: int,
            trajectory, trigger_tokens: int | None = None, todo_note: str = "") -> list[dict]:
    """Return the shortened history. Stage 1 elides old tool outputs; if the estimate is still above half the limit, stage 2
    summarises the middle with one extra model call. Logs a `compaction` event per stage."""
    before = estimate_tokens(messages)
    cutoff = max(len(messages) - keep_recent, 1)
    out = list(messages)
    for i in range(1, cutoff):
        m = out[i]
        if m["role"] == "tool" and len(m["content"]) > ELIDE_MIN_CHARS:
            out[i] = {**m, "content": f"[elided {len(m['content'])} chars]"}
    after = estimate_tokens(out)
    trajectory.log("compaction", kind="elide", est_tokens_before=before, est_tokens_after=after, trigger_input_tokens=trigger_tokens)
    if after <= context_limit * ELIDE_ENOUGH:
        return out

    # Stage 2: the tail must start at an assistant message so tool_calls/tool-result pairs stay intact and roles keep alternating.
    i = max(len(out) - keep_recent, 1)
    while i < len(out) and out[i]["role"] != "assistant":
        i += 1
    if i >= len(out) or i <= 1:
        return out
    transcript = render(messages[1:i])  # un-elided: elision must not starve the summary
    resp = client.chat(
        model=model, max_tokens=1500, temperature=temperature,
        messages=[{"role": "user", "content": f"{SUMMARY_PROMPT}\n\n<task>\n{out[0]['content']}\n</task>\n\n<transcript>\n{transcript}\n</transcript>"}],
    )
    summary = resp.text
    if not summary.strip():
        # An empty summary (e.g. a thinking model used the whole token budget) would silently delete the middle of the history.
        # Keep it, only the elision of old tool outputs stands, and say so in the trajectory.
        trajectory.log("compaction", kind="summary_failed", est_tokens_before=after, est_tokens_after=after,
                       trigger_input_tokens=trigger_tokens, finish_reason=resp.finish_reason)
        return out
    head = {"role": "user", "content": f"{out[0]['content']}\n\n[Progress so far, summarised from earlier steps]\n{summary}{todo_note}"}
    new = [head] + out[i:]
    trajectory.log("compaction", kind="summary", est_tokens_before=after, est_tokens_after=estimate_tokens(new), summary=summary,
                   trigger_input_tokens=trigger_tokens)
    return new
