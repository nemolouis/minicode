"""The compaction module on its own (the loop-level behaviour is covered in test_agent_loop.py and the golden replays)."""
from minicode import compaction
from minicode.client import ModelResponse
from minicode.trajectory import Trajectory


class Summariser:
    def __init__(self, text="SUMMARY"):
        self.text, self.calls = text, 0

    def chat(self, **kw):
        self.calls += 1
        return ModelResponse(self.text, [], "stop", {"input_tokens": 1, "output_tokens": 1})


def run(messages, client=None, **kw):
    client = client or Summariser()
    opts = dict(client=client, model="m", temperature=None, context_limit=1000, keep_recent=2, trajectory=Trajectory(None))
    opts.update(kw)
    return compaction.compact(messages, **opts), client


def asst(i): return {"role": "assistant", "content": f"a{i}", "tool_calls": [{"id": str(i), "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}
def tool(i, n=3000): return {"role": "tool", "tool_call_id": str(i), "content": "x" * n}


def test_trigger_fraction_and_regrowth():
    assert compaction.compact_trigger(10_000, 1.0, None) == 7000
    assert compaction.compact_trigger(10_000, 1.0, 6999) == 7000
    assert compaction.compact_trigger(10_000, 0.5, 8000) == 12_000


def test_tail_without_an_assistant_message_is_never_dropped():
    # the last two messages are tool results: the "tail starts at an assistant message" rule finds none, so nothing may be summarised
    msgs = [{"role": "user", "content": "task"}, asst(1), tool(1), tool(2), tool(3), tool(4)]
    out, client = run(msgs)
    assert client.calls == 0 and out[-1] == msgs[-1] and out[-2] == msgs[-2] and out[0] == msgs[0]


def test_nothing_to_summarise_when_the_tail_starts_right_after_the_task():
    msgs = [{"role": "user", "content": "task"}, asst(1), tool(1, 9000)]
    out, client = run(msgs, keep_recent=5, context_limit=100)
    assert client.calls == 0 and len(out) == 3


def test_summary_replaces_the_middle_and_keeps_the_tail_pairs_intact():
    msgs = [{"role": "user", "content": "task"}]
    for i in range(1, 6):
        msgs += [asst(i), tool(i, 6000)]
    out, client = run(msgs)
    assert client.calls == 1
    assert out[0]["content"].startswith("task") and "SUMMARY" in out[0]["content"]
    assert out[1:] == msgs[-2:]                                                          # exactly the recent window survives verbatim
    ids = [c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]]
    assert ids and all(any(m.get("tool_call_id") == i for m in out) for i in ids)       # every call keeps its result


def test_empty_summary_keeps_the_history_and_says_so(tmp_path):
    msgs = [{"role": "user", "content": "task"}]
    for i in range(1, 6):
        msgs += [asst(i), tool(i, 6000)]
    traj = Trajectory(tmp_path)
    out, _ = run(msgs, Summariser("   "), trajectory=traj)
    assert len(out) == len(msgs) and "SUMMARY" not in out[0]["content"]
    assert "summary_failed" in (tmp_path / "trajectory.jsonl").read_text()


def test_todo_note_is_appended_to_the_head():
    msgs = [{"role": "user", "content": "task"}]
    for i in range(1, 6):
        msgs += [asst(i), tool(i, 6000)]
    out, _ = run(msgs, todo_note="\n\n[Current todo list]\n[~] fix")
    assert out[0]["content"].endswith("[~] fix")
