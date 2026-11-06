"""MCP over Streamable HTTP: the client against a scripted local HTTP server (no network beyond 127.0.0.1)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from miniagent.mcp.client import MAX_HTTP_RESPONSE, McpError, McpHttpClient, McpTimeout, mcp_session

MODERN = "2026-07-28"


class Server:
    """mode: modern | legacy | sse | reject_probe (HTTP 404 on server/discover)."""
    def __init__(self, mode="modern", token=None):
        self.mode, self.token, self.seen, self.deleted, self.sid = mode, token, [], False, "sess-1"
        self.gets: list = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _reply(self, code, body=b"", ctype="application/json", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                outer.gets.append(dict(self.headers))
                self._reply(200, b"{}")

            def do_DELETE(self):
                outer.deleted = True
                self._reply(200)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append({"body": body, "headers": dict(self.headers)})
                if outer.token and self.headers.get("Authorization") != f"Bearer {outer.token}":
                    return self._reply(401, b'{"error":"unauthorized"}')
                if "id" not in body:
                    return self._reply(202)
                m, rid = body["method"], body["id"]
                if "io.modelcontextprotocol/protocolVersion" in (body.get("params") or {}).get("_meta", {}):     # modern request
                    h = {k.lower(): v for k, v in self.headers.items()}
                    want_name = body["params"].get("name") if m in ("tools/call", "prompts/get") else None
                    if h.get("mcp-method") != m or (want_name is not None and h.get("mcp-name") != want_name):
                        return self._json({"jsonrpc": "2.0", "id": rid, "error": {"code": -32020, "message": "mcp-method/mcp-name header mismatch"}})
                ok = lambda result, **kw: self._json({"jsonrpc": "2.0", "id": rid, "result": result}, **kw)
                if m == "server/discover":
                    if outer.mode == "reject_probe":
                        return self._reply(404, b"not found", "text/plain")
                    if outer.mode == "legacy":
                        return self._json({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown method"}})
                    return ok({"supportedVersions": [MODERN], "instructions": "hi",
                               "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "fake"}}})
                if m == "initialize":
                    return ok({"protocolVersion": "2025-11-25", "capabilities": {}, "serverInfo": {"name": "fake-legacy"}},
                              headers={"Mcp-Session-Id": outer.sid})
                if m == "tools/list":
                    return ok({"tools": [{"name": "echo", "description": "d", "inputSchema": {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}}]})
                if m == "tools/call":
                    name = body["params"]["name"]
                    if name == "slow":
                        time.sleep(1.5)
                    if name == "huge":
                        return ok({"content": [{"type": "text", "text": "x" * (MAX_HTTP_RESPONSE + 10)}]})
                    if name == "boom":
                        return ok({"content": [{"type": "text", "text": "kaput"}], "isError": True})
                    return ok({"content": [{"type": "text", "text": "echo:" + body["params"]["arguments"].get("t", "")}]})
                return self._json({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown"}})

            def _json(self, obj, headers=None):
                if outer.mode == "bad_id":
                    obj = {**obj, "id": obj["id"] + 100}
                if outer.mode == "sse":
                    ev = b"event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}).encode() + b"\n\n"
                    ev += b"data: " + json.dumps({"jsonrpc": "2.0", "id": 999, "result": {}}).encode() + b"\n\n"        # someone else's reply
                    ev += b"data: " + json.dumps(obj).encode() + b"\n\n"
                    return self._reply(200, ev, "text/event-stream", headers)
                self._reply(200, json.dumps(obj).encode(), "application/json", headers)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def make(*a, **kw):
        s = Server(*a, **kw)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.stop()


def test_modern_handshake_and_tool_call(serve):
    s = serve()
    c = McpHttpClient(s.url, timeout=5)
    c.start()
    c.initialize()
    probe = next(m for m in s.seen if m["body"].get("method") == "server/discover")
    assert {k.lower(): v for k, v in probe["headers"].items()}["mcp-protocol-version"] == MODERN       # already on the probe
    assert c.era == "modern" and c.instructions == "hi"
    assert [t["name"] for t in c.list_tools()] == ["echo"]
    assert c.call_tool("echo", {"t": "hi"}) == ("echo:hi", False)
    assert c.call_tool("boom", {}) == ("kaput", True)
    call = next(m for m in s.seen if m["body"].get("method") == "tools/call" and m["body"]["params"]["name"] == "boom")
    assert call["body"]["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] == MODERN       # per-request metadata
    h = {k.lower(): v for k, v in call["headers"].items()}
    assert h["mcp-protocol-version"] == MODERN and "text/event-stream" in h["accept"]
    assert h["mcp-method"] == "tools/call" and h["mcp-name"] == "boom"                                   # the modern binding's routing headers
    c.close()


@pytest.mark.parametrize("mode", ["legacy", "reject_probe"])
def test_falls_back_to_the_legacy_handshake_and_keeps_the_session(serve, mode):
    s = serve(mode)
    c = McpHttpClient(s.url, timeout=5)
    c.start()
    c.initialize()
    assert c.era == "legacy" and c.server_info["name"] == "fake-legacy" and c.session_id == "sess-1"
    c.call_tool("echo", {"t": "x"})
    methods = [m["body"].get("method") for m in s.seen]
    assert "notifications/initialized" in methods
    later = [m for m in s.seen if m["body"].get("method") in ("notifications/initialized", "tools/call")]
    assert all({k.lower(): v for k, v in m["headers"].items()}.get("mcp-session-id") == "sess-1" for m in later)
    assert "_meta" not in next(m for m in s.seen if m["body"].get("method") == "tools/call")["body"]["params"]
    c.close()
    assert s.deleted                                                    # the session is ended politely


def test_event_stream_replies_are_matched_by_id(serve):
    s = serve("sse")
    c = McpHttpClient(s.url, timeout=5)
    c.start()
    c.initialize()
    assert c.call_tool("echo", {"t": "via-sse"}) == ("echo:via-sse", False)


def test_a_slow_server_times_out(serve):
    s = serve()
    c = McpHttpClient(s.url, timeout=0.5)
    c.start()
    c.initialize()
    with pytest.raises(McpTimeout):
        c.call_tool("slow", {})


def test_an_oversized_reply_is_refused(serve):
    s = serve()
    c = McpHttpClient(s.url, timeout=20)
    c.start()
    c.initialize()
    with pytest.raises(McpError):
        c.call_tool("huge", {})


def test_unreachable_server_is_an_mcp_error(serve):
    s = serve()
    url = s.url
    s.stop()
    dead = McpHttpClient(url, timeout=1)
    dead.start()
    with pytest.raises(McpError):
        dead.list_tools()


def test_bearer_token_is_sent_and_a_wrong_one_is_an_error(serve):
    s = serve(token="sekret")
    ok = McpHttpClient(s.url, bearer_token="sekret", timeout=5)
    ok.start()
    ok.initialize()
    assert ok.list_tools()
    bad = McpHttpClient(s.url, bearer_token="nope", timeout=5)
    bad.start()
    with pytest.raises(McpError):
        bad.list_tools()


def test_redirects_are_not_followed_so_the_token_cannot_leave(serve):
    target = serve()
    redirector = serve()
    # 302 on a POST: urllib would silently turn it into a GET to the new location and copy the Authorization header along
    redirector.httpd.RequestHandlerClass.do_POST = lambda h: (h.send_response(302), h.send_header("Location", target.url), h.send_header("Content-Length", "0"), h.end_headers())
    c = McpHttpClient(redirector.url, bearer_token="sekret", timeout=5)
    c.start()
    with pytest.raises(McpError):
        c.list_tools()
    assert target.seen == [] and target.gets == []


def test_url_policy():
    for bad in ("ftp://x/mcp", "file:///etc/passwd", "http://example.com/mcp", "//x", "http:///x"):
        with pytest.raises(ValueError):
            McpHttpClient(bad)
    McpHttpClient("https://example.com/mcp")
    McpHttpClient("http://localhost:9/mcp")
    McpHttpClient("http://127.0.0.1:9/mcp")
    McpHttpClient("http://example.com/mcp", allow_insecure=True)


def test_a_session_id_with_odd_characters_is_not_adopted(serve):
    s = serve("legacy")
    s.sid = "has space"                                   # outside the visible-ASCII range the spec allows
    c = McpHttpClient(s.url, timeout=5)
    c.start()
    c.initialize()
    assert c.session_id is None
    c.call_tool("echo", {"t": "x"})
    assert not any("mcp-session-id" in {k.lower() for k in m["headers"]} for m in s.seen)


def test_mcp_session_config_with_url(serve, monkeypatch):
    s = serve(token="sekret")
    monkeypatch.setenv("NOTES_TOKEN", "sekret")
    with mcp_session([{"name": "remote", "url": s.url, "allow": ["echo"], "auth_env": "NOTES_TOKEN"}]) as tools:
        assert list(tools) == ["mcp__remote__echo"]
        assert tools["mcp__remote__echo"].func({"t": "q"}) == "echo:q"
    monkeypatch.delenv("NOTES_TOKEN")
    with pytest.raises(ValueError, match="not set"):
        with mcp_session([{"name": "remote", "url": s.url, "allow": ["echo"], "auth_env": "NOTES_TOKEN"}]):
            pass
    for bad in ({"name": "x", "allow": ["e"]}, {"name": "x", "allow": ["e"], "url": s.url, "command": ["python"]}):
        with pytest.raises(ValueError, match="exactly one"):
            with mcp_session([bad]):
                pass


def test_a_reply_with_the_wrong_id_is_rejected(serve):
    s = serve("bad_id")
    c = McpHttpClient(s.url, timeout=5)
    c.start()
    with pytest.raises(McpError, match="does not match"):
        c.initialize()


def test_an_oversized_event_stream_is_refused(serve):
    s = serve("sse")
    c = McpHttpClient(s.url, timeout=30)
    c.start()
    c.initialize()
    with pytest.raises(McpError, match="too large"):
        c.call_tool("huge", {})
