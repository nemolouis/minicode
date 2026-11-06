"""Streaming replies (against a local SSE server) and graceful Ctrl-C."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from miniagent.agent import Agent
from miniagent.client import APIError, ChatClient, ModelResponse, ToolCall
from miniagent.session import load_session
from miniagent.tools import Tool, Workspace
from miniagent.trajectory import Trajectory


def sse(*chunks, done=True):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


class Server:
    """Replies to POST /chat/completions with the queued body; remembers the requests."""
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                code, body, ctype = outer.replies.pop(0)
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    made = []

    def make(*replies):
        s = Server(replies)
        made.append(s)
        return s
    yield make
    for s in made:
        s.stop()


def delta(**d): return {"model": "m1", "choices": [{"index": 0, "delta": d}]}
def finish(reason): return {"model": "m1", "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}
def usage(i=10, o=5, hit=3): return {"model": "m1", "choices": [], "usage": {"prompt_tokens": i, "completion_tokens": o, "prompt_cache_hit_tokens": hit}}


def client(s, **kw): return ChatClient(api_key="k", base_url=s.url, retries=0, timeout=5, **kw)


def ask(c, on_text, **kw): return c.chat(model="m", messages=[{"role": "user", "content": "hi"}], on_text=on_text, **kw)


# ---------------------------------------------------------------- streaming
def test_text_streams_in_pieces_and_the_response_is_the_same_shape(serve):
    s = serve((200, sse(delta(role="assistant", content=""), delta(content="Hel"), delta(content="lo "), delta(content="world"),
                        delta(reasoning_content="thinking"), finish("stop"), usage()), "text/event-stream"))
    seen = []
    r = ask(client(s), seen.append)
    assert seen == ["Hel", "lo ", "world"] and r.text == "Hello world" and r.finish_reason == "stop" and r.tool_calls == []
    assert r.usage["input_tokens"] == 10 and r.usage["output_tokens"] == 5 and r.usage["cache_hit_tokens"] == 3
    assert r.model == "m1" and r.reasoning == "thinking"
    req = s.requests[0]
    assert req["stream"] is True and req["stream_options"] == {"include_usage": True}


def test_tool_call_deltas_are_merged_by_index(serve):
    s = serve((200, sse(
        delta(tool_calls=[{"index": 0, "id": "c1", "type": "function", "function": {"name": "read_file", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": "{\"path\": "}}]),
        delta(tool_calls=[{"index": 1, "id": "c2", "type": "function", "function": {"name": "grep", "arguments": "{\"pattern\":\"x\"}"}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": "\"a.py\"}"}}]),
        finish("tool_calls"), usage()), "text/event-stream"))
    r = ask(client(s), lambda t: None)
    assert [(c.id, c.name, c.arguments) for c in r.tool_calls] == [("c1", "read_file", '{"path": "a.py"}'), ("c2", "grep", '{"pattern":"x"}')]
    assert r.finish_reason == "tool_calls"


def test_provider_fields_on_a_tool_call_are_kept(serve):
    s = serve((200, sse(delta(tool_calls=[{"index": 0, "id": "c", "function": {"name": "bash", "arguments": "{}"},
                                           "extra_content": {"sig": "abc"}}]), finish("tool_calls"), usage()), "text/event-stream"))
    assert ask(client(s), lambda t: None).tool_calls[0].extra == {"extra_content": {"sig": "abc"}}


def test_without_on_text_nothing_changes_and_no_stream_is_requested(serve):
    body = json.dumps({"model": "m1", "choices": [{"message": {"content": "plain"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
    s = serve((200, body, "application/json"))
    r = client(s).chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert r.text == "plain" and "stream" not in s.requests[0]


def test_http_error_before_the_stream_is_an_api_error_and_429_quota_is_recognised(serve):
    s = serve((400, b'{"error": "bad"}', "application/json"))
    with pytest.raises(APIError, match="HTTP 400"):
        ask(client(s), lambda t: None)


def test_a_server_error_before_the_stream_is_retried(serve):
    ok = sse(delta(content="ok"), finish("stop"), usage())
    s = serve((503, b"busy", "text/plain"), (200, ok, "text/event-stream"))
    c = ChatClient(api_key="k", base_url=s.url, retries=1, timeout=5)
    c._delay = lambda *a: 0.0
    assert ask(c, lambda t: None).text == "ok" and len(s.requests) == 2


def test_an_error_event_in_the_stream_and_an_empty_stream_are_errors(serve):
    s = serve((200, sse({"error": {"message": "overloaded"}}), "text/event-stream"), (200, b"", "text/event-stream"))
    with pytest.raises(APIError, match="stream error"):
        ask(client(s), lambda t: None)
    with pytest.raises(APIError, match="without any data"):
        ask(client(s), lambda t: None)


def test_a_stream_cut_off_without_done_still_returns_what_arrived(serve):
    s = serve((200, sse(delta(content="partial"), finish("length"), done=False), "text/event-stream"))
    seen = []
    r = ask(client(s), seen.append)
    assert r.text == "partial" and r.finish_reason == "length" and seen == ["partial"]


def test_garbage_lines_and_comments_are_skipped(serve):
    body = b": keep-alive\n\nevent: ping\nnot sse at all\ndata: {not json}\n\n" + sse(delta(content="fine"), finish("stop"), usage())
    s = serve((200, body, "text/event-stream"))
    assert ask(client(s), lambda t: None).text == "fine"


def test_text_tool_calls_still_work_with_streaming(serve):
    reply = 'ok {"name": "bash", "arguments": {"command": "ls"}}'
    s = serve((200, sse(delta(content=reply), finish("stop"), usage()), "text/event-stream"))
    r = ask(client(s, text_tool_calls=True), lambda t: None, tools=[{"function": {"name": "bash"}}])
    assert r.tool_calls and r.tool_calls[0].name == "bash" and r.finish_reason == "tool_calls"


# ---------------------------------------------------------------- the agent passes on_text through only when asked
class Recorder:
    def __init__(self):
        self.kwargs = []

    def chat(self, **kw):
        self.kwargs.append(sorted(kw))
        return ModelResponse("hi", [], "stop", {"input_tokens": 1, "output_tokens": 1})


def test_agent_passes_on_text_only_when_set(tmp_path):
    (tmp_path / "w").mkdir()
    r = Recorder()
    Agent(r, "m", Workspace(tmp_path / "w")).run("t")
    assert "on_text" not in r.kwargs[0]
    r2 = Recorder()
    Agent(r2, "m", Workspace(tmp_path / "w"), on_text=print).run("t")
    assert "on_text" in r2.kwargs[0]


# ---------------------------------------------------------------- Ctrl-C
def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, **kw):
        import copy
        self.calls.append(copy.deepcopy(kw))
        item = self.script.pop(0)
        if item == "INTERRUPT":
            raise KeyboardInterrupt
        content, tok = item
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                             {"input_tokens": tok, "output_tokens": 5, "cache_hit_tokens": 0})


def valid_history(messages):
    """Every assistant tool call has exactly one tool message, right after it."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            following = [x.get("tool_call_id") for x in messages[i + 1:i + 1 + len(ids)] if x["role"] == "tool"]
            assert sorted(following) == sorted(ids), (ids, following)
    return True


def test_ctrl_c_during_a_model_request_saves_a_resumable_session(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "f.txt").write_text("x")
    a = Agent(Script([([call("1", "read_file", path="f.txt")], 10), "INTERRUPT"]), "m", Workspace(tmp_path / "w"),
              trajectory=Trajectory(tmp_path / "run"), session=True)
    r = a.run("t")
    assert r.status == "interrupted" and r.steps == 1
    st = load_session(tmp_path / "run")
    assert st["step"] == 1 and st["messages"][-1]["role"] == "tool" and valid_history(st["messages"])
    ev = [json.loads(l) for l in (tmp_path / "run" / "trajectory.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "run_end" and ev[-1]["status"] == "interrupted"
    b = Agent(Script([(["carried on"], 10)]), "m", Workspace(tmp_path / "w"), trajectory=Trajectory(tmp_path / "run"), session=True)
    assert b.run("continue", resume=st).status == "done"
    assert valid_history(b.client.calls[0]["messages"])


def test_ctrl_c_in_the_middle_of_a_tool_turn_closes_the_unanswered_calls(tmp_path):
    (tmp_path / "w").mkdir()

    def boom(args): raise KeyboardInterrupt

    tools = {"slow": Tool("slow", "d", {"type": "object", "properties": {}}, boom)}
    script = [([call("1", "slow"), call("2", "slow")], 10)]
    a = Agent(Script(script), "m", Workspace(tmp_path / "w"), trajectory=Trajectory(tmp_path / "run"), session=True, extra_tools=tools)
    r = a.run("t")
    assert r.status == "interrupted"
    st = load_session(tmp_path / "run")
    assert valid_history(st["messages"]) and st["step"] == 1
    notes = [m["content"] for m in st["messages"] if m["role"] == "tool"]
    assert len(notes) == 2 and all("interrupted by the user" in n for n in notes)


def test_ctrl_c_before_anything_happened(tmp_path):
    (tmp_path / "w").mkdir()
    a = Agent(Script(["INTERRUPT"]), "m", Workspace(tmp_path / "w"), trajectory=Trajectory(tmp_path / "run"), session=True)
    r = a.run("t")
    assert r.status == "interrupted" and r.steps == 0
    assert load_session(tmp_path / "run")["messages"] == [{"role": "user", "content": "t"}]


def test_cli_reports_an_interrupted_run_with_a_resume_hint(tmp_path, monkeypatch, capsys):
    from miniagent import cli
    (tmp_path / "w").mkdir()
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([(["x"], 10)] if False else ["INTERRUPT"]))
    code = cli.main(["go", "-w", str(tmp_path / "w"), "-m", "fake", "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert code == 130 and "--resume" in out and "interrupted" in out


def test_cli_stream_prints_text_as_it_arrives(tmp_path, monkeypatch, capsys):
    from miniagent import cli
    (tmp_path / "w").mkdir()

    class Streaming:
        def chat(self, on_text=None, **kw):
            for piece in ("Hel", "lo"):
                on_text(piece)
            return ModelResponse("Hello", [], "stop", {"input_tokens": 1, "output_tokens": 1})

    monkeypatch.setattr(cli, "ChatClient", lambda: Streaming())
    assert cli.main(["go", "-w", str(tmp_path / "w"), "-m", "fake", "--runs-dir", str(tmp_path / "runs"), "--stream"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Hello") and out.count("Hello") == 1                                    # shown once, not again at the end
