"""Where a run came from: git commit, whether the tree had uncommitted changes, interpreter. Recorded in meta.json and run_start so
that a number in a report can be tied to the code that produced it. Computed once per process (a batch of runs shares it)."""
from __future__ import annotations

import functools
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def provenance() -> dict:
    """{git_sha, git_dirty, dirty_files, python, platform}. git fields are None outside a git checkout (e.g. a pip-installed copy)."""
    sha = _git("rev-parse", "HEAD")
    out = {"git_sha": sha, "git_dirty": None, "dirty_files": [], "python": sys.version.split()[0], "platform": platform.platform()}
    if sha:
        status = _git("status", "--porcelain", "--", "minicode", "configs", "prompts", "tasks", "pyproject.toml")
        if status is not None:
            files = [line[3:] for line in status.splitlines()]
            out["git_dirty"], out["dirty_files"] = bool(files), files[:20]
    return out
