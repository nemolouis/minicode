"""The Anthropic Messages API client: message conversion, cache breakpoints, usage, streaming (local fake server, no network)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from miniagent.client import ANTHROPIC_BASE_URL, ANTHROPIC_VERSION, APIError, ChatClient


class Server:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
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


def reply(content, stop="end_turn", usage=None, model="claude-x"):
    body = {"type": "message", "model": model, "role": "assistant", "content": content, "stop_reason": stop,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5}}
    return 200, json.dumps(body).encode(), "application/json"


def client(s, **kw): return ChatClient(api_key="k", base_url=s.url, api_style="anthropic", retries=0, timeout=5, **kw)


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "read",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object", "properties": {}}}}]


def call_tool_id(i, name, args): return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# ---------------------------------------------------------------- request shape
def test_request_headers_path_system_tools_and_temperature(serve):
    s = serve(reply([{"type": "text", "text": "hi"}]))
    r = client(s).chat(model="claude-x", system="be brief", messages=[{"role": "user", "content": "hello"}], tools=TOOLS, max_tokens=99,
                       temperature=0.2)
    req = s.requests[0]
    assert req["path"] == "/v1/messages" and req["headers"]["x-api-key"] == "k" and req["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert "authorization" not in req["headers"]
    b = req["body"]
    assert b["model"] == "claude-x" and b["max_tokens"] == 99 and b["temperature"] == 0.2 and "stream" not in b
    assert b["system"] == [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}]
    assert b["tools"][0]["name"] == "read_file" and b["tools"][0]["input_schema"] == TOOLS[0]["function"]["parameters"]
    assert "cache_control" not in b["tools"][0] and b["tools"][1]["cache_control"] == {"type": "ephemeral"}      # only the LAST tool is marked
    assert b["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}]
    assert r.text == "hi" and r.finish_reason == "stop" and r.model == "claude-x"


def test_prompt_cache_can_be_switched_off(serve):
    s = serve(reply([{"type": "text", "text": "hi"}]))
    client(s, prompt_cache=False).chat(model="m", system="x", messages=[{"role": "user", "content": "hello"}], tools=TOOLS)
    assert "cache_control" not in json.dumps(s.requests[0]["body"])


def test_history_with_tool_calls_is_converted_and_roles_alternate(serve):
    history = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "looking", "tool_calls": [call_tool_id("t1", "read_file", {"path": "a.py"}),
                                                                 call_tool_id("t2", "bash", {"command": "ls"})]},
        {"role": "tool", "tool_call_id": "t1", "content": "file body"},
        {"role": "tool", "tool_call_id": "t2", "content": "error: command blocked"},
        {"role": "assistant", "content": None, "tool_calls": [call_tool_id("t3", "bash", {"command": "pytest"})]},
        {"role": "tool", "tool_call_id": "t3", "content": "3 passed"},
        {"role": "user", "content": "thanks, now summarise"},
    ]
    s = serve(reply([{"type": "text", "text": "ok"}]))
    client(s, prompt_cache=False).chat(model="m", messages=history)
    m = s.requests[0]["body"]["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant", "user"]
    assert m[1]["content"] == [{"type": "text", "text": "looking"},
                               {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
                               {"type": "tool_use", "id": "t2", "name": "bash", "input": {"command": "ls"}}]
    assert m[2]["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "file body"},
                               {"type": "tool_result", "tool_use_id": "t2", "content": "error: command blocked", "is_error": True}]
    assert m[3]["content"] == [{"type": "tool_use", "id": "t3", "name": "bash", "input": {"command": "pytest"}}]      # no empty text block
    assert [b["type"] for b in m[4]["content"]] == ["tool_result", "text"]                                           # result first, then the new text


def test_unparseable_tool_arguments_do_not_break_the_request(serve):
    history = [{"role": "user", "content": "x"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "t", "type": "function", "function": {"name": "bash", "arguments": "{oops"}}]},
               {"role": "tool", "tool_call_id": "t", "content": "error: arguments are not valid JSON"}]
    s = serve(reply([{"type": "text", "text": "ok"}]))
    client(s, prompt_cache=False).chat(model="m", messages=history)
    assert s.requests[0]["body"]["messages"][1]["content"][0]["input"] == {"_raw_arguments": "{oops"}


# ---------------------------------------------------------------- response mapping
def test_tool_use_blocks_become_tool_calls_and_stop_reasons_map(serve):
    blocks = [{"type": "text", "text": "I'll look"}, {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}]
    s = serve(reply(blocks, stop="tool_use"), reply([{"type": "text", "text": "cut"}], stop="max_tokens"),
              reply([{"type": "text", "text": ""}], stop="refusal"))
    c = client(s)
    r = c.chat(model="m", messages=[{"role": "user", "content": "x"}], tools=TOOLS)
    assert r.text == "I'll look" and r.finish_reason == "tool_calls"
    assert [(t.id, t.name, json.loads(t.arguments)) for t in r.tool_calls] == [("toolu_1", "read_file", {"path": "a.py"})]
    assert c.chat(model="m", messages=[{"role": "user", "content": "x"}]).finish_reason == "length"
    assert c.chat(model="m", messages=[{"role": "user", "content": "x"}]).finish_reason == "content_filter"


def test_usage_includes_the_cached_prefix_like_the_other_styles(serve):
    s = serve(reply([{"type": "text", "text": "x"}], usage={"input_tokens": 100, "cache_read_input_tokens": 4000,
                                                          "cache_creation_input_tokens": 500, "output_tokens": 7}))
    r = client(s).chat(model="m", messages=[{"role": "user", "content": "x"}])
    assert r.usage["input_tokens"] == 4600 and r.usage["cache_hit_tokens"] == 4000 and r.usage["output_tokens"] == 7
    assert r.usage["cache_creation_tokens"] == 500


def test_error_responses(serve):
    s = serve((400, b'{"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}', "application/json"),
              (200, b'{"type": "error", "error": {"message": "weird"}}', "application/json"))
    with pytest.raises(APIError, match="HTTP 400"):
        client(s).chat(model="m", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(APIError, match="unexpected response"):
        client(s).chat(model="m", messages=[{"role": "user", "content": "x"}])


# ---------------------------------------------------------------- key and host rules
def test_default_host_and_key_rules(monkeypatch):
    for v in ("LLM_API_KEY", "LLM_BASE_URL", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(APIError, match="no API key"):
        ChatClient(api_style="anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    c = ChatClient(api_style="anthropic")
    assert c.base_url == ANTHROPIC_BASE_URL and c.api_key == "sk-ant"
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")                         # must not leak into the anthropic style
    assert ChatClient(api_style="anthropic").base_url == ANTHROPIC_BASE_URL
    with pytest.raises(APIError, match="no API key"):                                           # a relay does not get the Anthropic key
        ChatClient(api_style="anthropic", base_url="https://relay.example.com")
    with pytest.raises(APIError, match="unknown api_style"):
        ChatClient(api_key="k", api_style="claude")


# ---------------------------------------------------------------- streaming
def sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_streaming_text_and_tool_use(serve):
    body = sse(
        {"type": "message_start", "message": {"model": "claude-x", "usage": {"input_tokens": 12, "cache_read_input_tokens": 100}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "check"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"path\": "}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"a.py\"}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 33}},
        {"type": "message_stop"})
    s = serve((200, body, "text/event-stream"))
    seen = []
    r = client(s).chat(model="m", messages=[{"role": "user", "content": "x"}], tools=TOOLS, on_text=seen.append)
    assert seen == ["Let me ", "check"] and r.text == "Let me check" and r.finish_reason == "tool_calls"
    assert [(t.id, t.name, json.loads(t.arguments)) for t in r.tool_calls] == [("tu1", "read_file", {"path": "a.py"})]
    assert r.usage["input_tokens"] == 112 and r.usage["output_tokens"] == 33 and r.usage["cache_hit_tokens"] == 100 and r.model == "claude-x"
    assert s.requests[0]["body"]["stream"] is True


def test_streaming_error_event_and_empty_stream(serve):
    s = serve((200, sse({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}), "text/event-stream"),
              (200, b"", "text/event-stream"))
    with pytest.raises(APIError, match="stream error"):
        client(s).chat(model="m", messages=[{"role": "user", "content": "x"}], on_text=lambda t: None)
    with pytest.raises(APIError, match="without any data"):
        client(s).chat(model="m", messages=[{"role": "user", "content": "x"}], on_text=lambda t: None)


def test_agent_runs_against_the_anthropic_client(serve, tmp_path):
    from miniagent.agent import Agent
    from miniagent.tools import Workspace
    (tmp_path / "f.txt").write_text("hello\n")
    s = serve(reply([{"type": "tool_use", "id": "tu1", "name": "read_file", "input": {"path": "f.txt"}}], stop="tool_use"),
              reply([{"type": "text", "text": "The file says hello."}]))
    r = Agent(client(s), "claude-x", Workspace(tmp_path)).run("what does f.txt say?")
    assert r.status == "done" and r.final_text == "The file says hello."
    second = s.requests[1]["body"]["messages"]
    assert [x["role"] for x in second] == ["user", "assistant", "user"] and second[2]["content"][0]["type"] == "tool_result"
    assert "hello" in second[2]["content"][0]["content"]
