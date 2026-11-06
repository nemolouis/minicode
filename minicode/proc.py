"""Run a shell command so that a timeout really stops it.

`subprocess.run(shell=True, timeout=...)` only kills the shell; with pipelines or
`cd x && pytest` the grandchildren keep running as orphans (PPID 1, 100% CPU). Here the
command gets its own process group and the whole group is killed on timeout and after exit,
which also reaps background processes a command left behind.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass
class ShellResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    cancelled: bool = False


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_shell(args, *, cwd=None, env=None, timeout: float = 60, shell: bool = False, cancel=None) -> ShellResult:
    """Run a command. `cancel` (a threading.Event) stops it early: the process group is killed and the result says `cancelled`."""
    p = subprocess.Popen(args, shell=shell, cwd=cwd, env=env, text=True, errors="replace",
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while True:
            wait = min(0.2, max(deadline - time.monotonic(), 0.001)) if cancel is not None else max(deadline - time.monotonic(), 0.001)
            try:
                out, err = p.communicate(timeout=wait)
                return ShellResult(p.returncode, out, err, False)
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    _kill_group(p.pid)
                    out, err = p.communicate()
                    return ShellResult(None, out or "", err or "", False, cancelled=True)
                if time.monotonic() >= deadline:
                    _kill_group(p.pid)
                    out, err = p.communicate()
                    return ShellResult(None, out or "", err or "", True)
    finally:
        _kill_group(p.pid)       # stragglers left behind by a command that did finish
