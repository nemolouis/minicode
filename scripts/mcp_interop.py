#!/usr/bin/env python3
"""Interoperability check against the official MCP Python SDK (the `mcp` package), in both directions:

  1. our HTTP client     -> SDK server over Streamable HTTP
  2. our stdio client    -> SDK server over stdio
  3. SDK client (stdio)  -> our notes server (miniagent/mcp/notes_server.py)

The SDK is a third-party dependency, so run this in a separate environment (miniagent itself needs nothing):

    python3 -m venv .venv-sdk && .venv-sdk/bin/pip install mcp && .venv-sdk/bin/pip install -e .
    .venv-sdk/bin/python scripts/mcp_interop.py

Exit code 0 only if every check passed. Checked against mcp 2.3.0; the script prints the SDK version it ran with.
"""
from __future__ import annotations

import argparse
import asyncio
import socket
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve()


# ------------------------------------------------------------------ a tiny SDK-based server (run as a subprocess)
def build_server():
    from mcp.server.mcpserver import MCPServer
    srv = MCPServer("interop-sdk-server", instructions="interop test server")

    @srv.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @srv.tool()
    def fail(reason: str) -> str:
        """Always fails."""
        raise ValueError(reason)

    return srv


def serve_http(port: int) -> None:
    import uvicorn
    uvicorn.run(build_server().streamable_http_app(), host="127.0.0.1", port=port, log_level="warning")


def serve_stdio() -> None:
    build_server().run("stdio")


# ------------------------------------------------------------------ the checks
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def drive(client, label: str) -> None:
    from miniagent.mcp.client import McpError
    client.initialize()
    check(f"{label}: handshake", client.era in ("modern", "legacy"), f"era={client.era}, server={client.server_info}")
    tools = {t["name"]: t for t in client.list_tools()}
    check(f"{label}: tools/list", {"add", "fail"} <= set(tools), ", ".join(sorted(tools)))
    text, is_err = client.call_tool("add", {"a": 2, "b": 3})
    check(f"{label}: tools/call add(2,3)", text.strip() == "5" and not is_err, f"text={text!r}")
    try:
        text, is_err = client.call_tool("fail", {"reason": "nope"})
        check(f"{label}: tool failure is reported", is_err, f"text={text!r}")
    except McpError as e:
        check(f"{label}: tool failure is reported", True, f"as protocol error: {e}")
    try:
        text, is_err = client.call_tool("add", {"a": "x"})
        check(f"{label}: bad arguments rejected", is_err, f"text={text!r}")
    except McpError as e:
        check(f"{label}: bad arguments rejected", True, f"as protocol error: {e}")


def wait_port(port: int, proc: subprocess.Popen, seconds: float = 20) -> bool:
    end = time.time() + seconds
    while time.time() < end and proc.poll() is None:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def check_our_http_client() -> None:
    from miniagent.mcp.client import McpHttpClient
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(HERE), "--serve-http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        if not wait_port(port, proc):
            check("http: SDK server started", False, (proc.stderr.read() or b"").decode()[-300:])
            return
        c = McpHttpClient(f"http://127.0.0.1:{port}/mcp", timeout=15)
        c.start()
        try:
            drive(c, "our HTTP client -> SDK server")
        finally:
            c.close()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def check_our_stdio_client() -> None:
    from miniagent.mcp.client import McpClient
    c = McpClient([sys.executable, str(HERE), "--serve-stdio"], timeout=15)
    c.start()
    try:
        drive(c, "our stdio client -> SDK server")
    finally:
        c.close()


def err(result) -> bool:
    """CallToolResult.is_error in mcp 2.x, isError in 1.x."""
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


async def sdk_client_against_our_server() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    notes = HERE.parent.parent / "examples" / "notes"
    params = StdioServerParameters(command=sys.executable, args=["-m", "miniagent.mcp.notes_server", "--root", str(notes)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            check("SDK client -> our notes server: tools/list", names == {"list_notes", "read_note", "search_notes"}, ", ".join(sorted(names)))
            listing = await session.call_tool("list_notes", {})
            text = "".join(getattr(c, "text", "") for c in listing.content)
            check("SDK client -> our notes server: list_notes", not err(listing) and ".md" in text, text[:80].replace("\n", " "))
            first = next((w for w in text.split() if w.endswith(".md")), None)
            if first:
                got = await session.call_tool("read_note", {"path": first})
                check("SDK client -> our notes server: read_note", not err(got) and got.content, first)
            for bad in ("../../etc/passwd", "/etc/passwd", ".hidden.md"):
                r = await session.call_tool("read_note", {"path": bad})
                body = "".join(getattr(c, "text", "") for c in r.content)
                check(f"SDK client -> our notes server: refuses {bad!r}", err(r) and "root:" not in body, body[:60])
            try:                              # an unknown tool is a JSON-RPC error: the SDK raises it, which is the correct behaviour
                r = await session.call_tool("append_note", {"path": "x.md", "text": "y"})
                check("SDK client -> our notes server: write tool absent in read-only mode", err(r), "returned a tool result")
            except Exception as e:
                check("SDK client -> our notes server: write tool absent in read-only mode", True, f"{type(e).__name__}: {str(e)[:60]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-http", type=int)
    ap.add_argument("--serve-stdio", action="store_true")
    a = ap.parse_args()
    if a.serve_http:
        serve_http(a.serve_http)
        return 0
    if a.serve_stdio:
        serve_stdio()
        return 0
    print(f"mcp SDK {version('mcp')}, python {sys.version.split()[0]}")
    check_our_http_client()
    check_our_stdio_client()
    asyncio.run(sdk_client_against_our_server())
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
