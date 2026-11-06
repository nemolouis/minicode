"""A minimal MCP client over stdio, and the glue that exposes MCP tools to the agent loop.

Lifecycle (per the MCP spec): the client launches the server as a subprocess and exchanges one JSON-RPC
message per line. It first probes with `server/discover` (modern protocol); a modern error such as
-32022 also identifies a modern server, anything else (or silence) means a legacy server and the client
falls back to the `initialize` handshake. Shutdown: close stdin, wait, then SIGTERM, then SIGKILL.
"""
from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from contextlib import contextmanager

from ..tools import Tool, ToolError

MODERN_VERSION = "2026-07-28"
LEGACY_VERSION = "2025-11-25"
CLIENT_INFO = {"name": "minicode", "version": "0.1.0"}
_EOF = object()


class McpError(Exception):
    """Local failure: server not running, timeout, malformed reply, unsupported result type."""


class McpRpcError(McpError):
    def __init__(self, code: int, message: str, data=None):
        super().__init__(f"MCP error {code}: {message}")
        self.code, self.message, self.data = code, message, data


class McpTimeout(McpError):
    pass


class McpProbeFailed(McpTimeout):
    """The modern-protocol probe got no usable answer (HTTP error, not JSON-RPC): fall back to the legacy handshake."""


def _kill_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class McpClient:
    def __init__(self, command: list[str], *, env: dict | None = None, cwd: str | None = None, timeout: float = 20.0):
        self.command, self.cwd, self.timeout = list(command), cwd, timeout
        # the server gets a minimal environment, not our secrets (API keys live in os.environ)
        self.env = env if env is not None else {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        self.proc: subprocess.Popen | None = None
        self.era: str | None = None
        self.server_info: dict = {}
        self.instructions: str = ""
        self.stderr_tail: deque = deque(maxlen=200)
        self._pending: dict[int, queue.Queue] = {}
        self._next_id = 0
        self._send_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ process
    def start(self) -> None:
        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=self.cwd, env=self.env,
                                     start_new_session=True)
        for target in (self._read_stdout, self._drain_stderr):       # stderr must be drained or a chatty server blocks
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue                                          # not a protocol message: ignore it
                q = self._pending.get(msg.get("id")) if isinstance(msg, dict) else None
                if q is not None and ("result" in msg or "error" in msg):
                    q.put(msg)
        finally:
            for q in list(self._pending.values()):
                q.put(_EOF)                                            # wake every waiter: the server is gone

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip("\n")[:500])

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            proc.stdin.close()                                         # 1. the polite signal: stdin EOF
        except OSError:
            pass
        for sig, wait in ((None, 2.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            if sig is not None:
                _kill_group(proc, sig)
            try:
                proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        _kill_group(proc, signal.SIGKILL)                              # reap anything the server left behind
        for t in self._threads:
            t.join(timeout=1.0)
        self.proc = None

    # ------------------------------------------------------------------ JSON-RPC
    def _send(self, msg: dict) -> None:
        if not self.alive:
            raise McpError(f"MCP server is not running (stderr: {' | '.join(list(self.stderr_tail)[-3:])})")
        data = json.dumps(msg, separators=(",", ":"), ensure_ascii=True) + "\n"
        try:
            with self._send_lock:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise McpError(f"could not write to the MCP server: {e}")

    def _meta(self) -> dict:
        return {"io.modelcontextprotocol/protocolVersion": MODERN_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO}

    def request(self, method: str, params: dict | None = None, timeout: float | None = None, *, modern: bool | None = None) -> dict:
        use_meta = (self.era == "modern") if modern is None else modern
        params = dict(params or {})
        if use_meta:
            params["_meta"] = self._meta()
        with self._id_lock:
            self._next_id += 1
            rid = self._next_id
        q: queue.Queue = queue.Queue(maxsize=1)
        self._pending[rid] = q
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            try:
                resp = q.get(timeout=timeout or self.timeout)
            except queue.Empty:
                try:
                    self._send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                                "params": {"requestId": rid, "reason": "client timeout"}})
                except McpError:
                    pass
                raise McpTimeout(f"{method} timed out after {timeout or self.timeout:.0f}s")
        finally:
            self._pending.pop(rid, None)
        if resp is _EOF:
            raise McpError(f"MCP server exited (stderr: {' | '.join(list(self.stderr_tail)[-3:])})")
        if "error" in resp:
            e = resp["error"] if isinstance(resp["error"], dict) else {}
            raise McpRpcError(e.get("code", -32603), str(e.get("message", "error")), e.get("data"))
        result = resp["result"]
        if not isinstance(result, dict):
            raise McpError("malformed result")
        if result.get("resultType", "complete") != "complete":         # e.g. input_required: not supported here
            raise McpError(f"unsupported resultType {result.get('resultType')!r}")
        return result

    # ------------------------------------------------------------------ handshake
    def initialize(self, probe_timeout: float = 5.0) -> None:
        """Probe the modern protocol; fall back to the legacy handshake. Sets `self.era`."""
        try:
            result = self.request("server/discover", timeout=probe_timeout, modern=True)
            supported = result.get("supportedVersions") or []
            if MODERN_VERSION not in supported:
                raise McpError(f"no common protocol version (server supports {supported})")
            self.era = "modern"
            info = (result.get("_meta") or {}).get("io.modelcontextprotocol/serverInfo")
            self.server_info, self.instructions = info or {}, str(result.get("instructions") or "")
            return
        except McpRpcError as e:
            if e.code == -32022:                                       # modern server, version mismatch: no fallback
                raise McpError(f"server does not support protocol {MODERN_VERSION}: {e.data}")
        except McpTimeout:
            pass
        if not self.alive:
            raise McpError("MCP server exited during startup: " + " | ".join(list(self.stderr_tail)[-3:]))
        result = self.request("initialize", {"protocolVersion": LEGACY_VERSION, "capabilities": {},
                                             "clientInfo": CLIENT_INFO}, modern=False)
        self.era = "legacy"
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.server_info, self.instructions = result.get("serverInfo") or {}, str(result.get("instructions") or "")

    # ------------------------------------------------------------------ tools
    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        for _ in range(50):                                            # bounded pagination
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += [t for t in result.get("tools", []) if isinstance(t, dict) and isinstance(t.get("name"), str)]
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
        raise McpError("tools/list pagination did not terminate")

    def call_tool(self, name: str, arguments: dict, timeout: float | None = None) -> tuple[str, bool]:
        result = self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(f"[{item.get('type', 'unknown')} content omitted]")
        return "\n".join(parts), bool(result.get("isError"))


# ---------------------------------------------------------------------- Streamable HTTP transport
MAX_HTTP_RESPONSE = 5_000_000
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None                                    # a redirect could carry the bearer token to another host: never follow


class McpHttpClient(McpClient):
    """MCP over Streamable HTTP: every JSON-RPC message is a POST; the reply is a JSON body or a text/event-stream containing the
    reply. Same interface as the stdio client (initialize / list_tools / call_tool / close), so the agent glue is shared.

    Safety: only https, or http to localhost (a bearer token over plain http to a remote host is refused); redirects are not
    followed; the reply size is capped; the token goes only to the configured URL; the server never sees our environment.
    """

    def __init__(self, url: str, *, bearer_token: str | None = None, timeout: float = 20.0, allow_insecure: bool = False):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"MCP url must be http(s)://host/path, got {url!r}")
        if u.scheme == "http" and u.hostname not in _LOCAL_HOSTS and not allow_insecure:
            raise ValueError("plain http is only allowed for localhost; use https (or set allow_insecure for a trusted network)")
        super().__init__([], timeout=timeout)
        self.url, self.token = url, bearer_token
        self.session_id: str | None = None
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._started = False

    def start(self) -> None:
        self._started = True

    @property
    def alive(self) -> bool:
        return self._started

    def close(self) -> None:
        if self._started and self.session_id:
            try:
                self._post(None, method="DELETE", timeout=3.0)
            except Exception:
                pass
        self._started = False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "User-Agent": f"{CLIENT_INFO['name']}/{CLIENT_INFO['version']}"}
        if self.era:
            h["MCP-Protocol-Version"] = MODERN_VERSION if self.era == "modern" else LEGACY_VERSION
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _post(self, msg: dict | None, *, method: str = "POST", timeout: float, extra_headers: dict | None = None):
        data = json.dumps(msg, separators=(",", ":")).encode() if msg is not None else None
        req = urllib.request.Request(self.url, data=data, headers={**self._headers(), **(extra_headers or {})}, method=method)
        return self._opener.open(req, timeout=timeout)

    def _send(self, msg: dict) -> None:                 # notifications: the server answers 202 with no body
        try:
            with self._post(msg, timeout=self.timeout) as r:
                r.read(MAX_HTTP_RESPONSE)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise McpError(f"could not send to the MCP server: {e}")

    def request(self, method: str, params: dict | None = None, timeout: float | None = None, *, modern: bool | None = None) -> dict:
        use_meta = (self.era == "modern") if modern is None else modern
        params = dict(params or {})
        if use_meta:
            params["_meta"] = self._meta()
        with self._id_lock:
            self._next_id += 1
            rid = self._next_id
        limit = timeout or self.timeout
        extra = {}
        if use_meta:         # the modern HTTP binding repeats the method (and the tool/prompt name) in headers; servers check them
            extra["MCP-Protocol-Version"] = MODERN_VERSION      # also on the discovery probe, before the era is known
            extra["Mcp-Method"] = method
            if method in ("tools/call", "prompts/get") and isinstance(params.get("name"), str):
                extra["Mcp-Name"] = params["name"]
        try:
            r = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}, timeout=limit, extra_headers=extra)
        except urllib.error.HTTPError as e:
            body = e.read(MAX_HTTP_RESPONSE).decode("utf-8", "replace")
            try:
                resp = json.loads(body)                     # a JSON-RPC error carried by a 4xx/5xx is still an answer
                if isinstance(resp, dict) and "error" in resp:
                    return self._unwrap(resp)
            except ValueError:
                pass
            if modern:
                raise McpProbeFailed(f"{method}: HTTP {e.code}")
            raise McpError(f"{method}: HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if "timed out" in str(e):
                raise McpTimeout(f"{method} timed out after {limit:.0f}s")
            raise McpError(f"{method}: {e}")
        with r:
            sid = r.headers.get("Mcp-Session-Id")
            if sid and not self.session_id and re.fullmatch(r"[\x21-\x7e]{1,256}", sid):
                self.session_id = sid
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            try:
                if ctype == "text/event-stream":
                    resp = self._read_sse(r, rid, limit)
                elif ctype == "application/json":
                    raw = r.read(MAX_HTTP_RESPONSE + 1)
                    if len(raw) > MAX_HTTP_RESPONSE:
                        raise McpError("reply too large")
                    resp = json.loads(raw.decode("utf-8", "replace"))
                else:
                    raise McpProbeFailed(f"{method}: unexpected content type {ctype!r}") if modern else \
                        McpError(f"{method}: unexpected content type {ctype!r}")
            except (TimeoutError, OSError) as e:
                raise McpTimeout(f"{method} timed out after {limit:.0f}s") if "timed out" in str(e) else McpError(f"{method}: {e}")
            except ValueError as e:
                raise McpError(f"{method}: malformed reply ({e})")
        if not isinstance(resp, dict) or resp.get("id") != rid:
            raise McpError(f"{method}: reply does not match the request")
        return self._unwrap(resp)

    def _read_sse(self, r, rid: int, limit: float) -> dict:
        """Read events until the one answering `rid`. Other events (progress, server requests) are ignored."""
        deadline, got, data = time.time() + limit, 0, []
        while time.time() < deadline:
            line = r.readline(MAX_HTTP_RESPONSE)
            if not line:
                break
            got += len(line)
            if got > MAX_HTTP_RESPONSE:
                raise McpError("reply too large")
            line = line.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif line == "" and data:
                try:
                    msg = json.loads("\n".join(data))
                except ValueError:
                    msg = None
                data = []
                if isinstance(msg, dict) and msg.get("id") == rid and ("result" in msg or "error" in msg):
                    return msg
        raise McpTimeout("event stream ended without a reply" if time.time() < deadline else f"no reply within {limit:.0f}s")

    @staticmethod
    def _unwrap(resp: dict) -> dict:
        if "error" in resp:
            e = resp["error"] if isinstance(resp["error"], dict) else {}
            raise McpRpcError(e.get("code", -32603), str(e.get("message", "error")), e.get("data"))
        result = resp.get("result")
        if not isinstance(result, dict):
            raise McpError("malformed result")
        if result.get("resultType", "complete") != "complete":
            raise McpError(f"unsupported resultType {result.get('resultType')!r}")
        return result


# ---------------------------------------------------------------------- glue for the agent
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_NAME_BAD = re.compile(r"[^A-Za-z0-9_-]")


def resolve_command(command: list[str]) -> list[str]:
    """`python` in a config means *this* interpreter (so the venv's packages are found)."""
    if command and command[0] in ("python", "python3"):
        return [sys.executable, *command[1:]]
    return list(command)


def mcp_tools_for(client: McpClient, server: str, allow: list[str], trusted: list[str] = ()) -> dict[str, Tool]:
    """Wrap the server's whitelisted tools as agent tools named `mcp__<server>__<tool>`.

    Only names in `allow` are exposed: the model cannot reach any other tool, so a guessed name never even
    gets as far as the server. Tool annotations are untrusted (the spec says so), so every MCP tool asks for
    user confirmation unless it is listed in `trusted`.
    """
    tools: dict[str, Tool] = {}
    listed = {t["name"]: t for t in client.list_tools()}
    for name in allow:
        t = listed.get(name)
        if t is None:
            continue
        schema = {k: v for k, v in (t.get("inputSchema") or {"type": "object", "properties": {}}).items() if k != "$schema"}
        full = f"mcp__{_NAME_BAD.sub('_', server)}__{_NAME_BAD.sub('_', name)}"
        desc = _CTRL.sub("", str(t.get("description") or ""))[:1000]

        def run(args: dict, _name=name) -> str:
            try:
                text, is_error = client.call_tool(_name, args)
            except McpError as e:
                raise ToolError(str(e))
            if is_error:
                raise ToolError(text)
            return text

        tools[full] = Tool(full, f"{desc} (MCP server '{server}')", schema, run, needs_confirm=name not in trusted)
    return tools


@contextmanager
def mcp_session(configs: list[dict]):
    """Start the configured MCP servers, yield their tools (name -> Tool), and always shut them down.

    Each config: {"name", "command": [...] | "url": "https://...", "allow": [tool names], "trusted": [...], "timeout": seconds}.
    A `url` server speaks Streamable HTTP; its bearer token (if any) is read from the environment variable named by `auth_env`.
    """
    clients: list[McpClient] = []
    try:
        tools: dict[str, Tool] = {}
        for cfg in configs or []:
            if not cfg.get("allow"):
                raise ValueError(f"MCP server {cfg.get('name')!r} needs an explicit non-empty 'allow' list")
            if ("command" in cfg) == ("url" in cfg):
                raise ValueError(f"MCP server {cfg.get('name')!r} needs exactly one of 'command' (stdio) or 'url' (HTTP)")
            if "url" in cfg:
                token = None
                if cfg.get("auth_env"):
                    token = os.environ.get(cfg["auth_env"])
                    if not token:
                        raise ValueError(f"MCP server {cfg.get('name')!r}: environment variable {cfg['auth_env']} is not set")
                c = McpHttpClient(cfg["url"], bearer_token=token, timeout=cfg.get("timeout", 20.0),
                                  allow_insecure=bool(cfg.get("allow_insecure", False)))
            else:
                c = McpClient(resolve_command(cfg["command"]), timeout=cfg.get("timeout", 20.0))
            clients.append(c)
            c.start()
            c.initialize()
            tools.update(mcp_tools_for(c, cfg["name"], cfg["allow"], cfg.get("trusted", [])))
        yield tools
    finally:
        for c in clients:
            c.close()
