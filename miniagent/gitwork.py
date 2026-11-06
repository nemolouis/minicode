"""`--git`: do the agent's work on its own branch and commit it as it goes, like the mainstream coding agents.

* the run starts from a clean tree (uncommitted work of yours is never swept into an agent commit); a new branch `miniagent/<name>` is
  cut from the current HEAD, or, if you are already on a `miniagent/` branch (a resumed session), that branch is continued;
* after every tool turn that changed files, one commit (author and committer "miniagent", hooks and signing skipped);
* only paths the AGENT changed are staged: modified/deleted tracked files, and untracked files that did not exist when the run began, so
  an untracked scratch file of yours stays untracked; ignored files are never added; the run directory is excluded when it lives
  inside the repository;
* nothing is pushed, no other branch is touched, no history is rewritten. The original branch is left exactly as it was.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

AUTHOR = ["-c", "user.name=miniagent", "-c", "user.email=miniagent@localhost", "-c", "commit.gpgsign=false"]
BRANCH_PREFIX = "miniagent/"


class GitError(Exception):
    pass


def _git(root: Path, *args: str, check: bool = True, timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise GitError("git is not installed")
    except subprocess.TimeoutExpired:
        raise GitError(f"git {args[0]} timed out")
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])} failed: {(r.stderr or r.stdout).strip()[:300]}")
    return r


def _status(root: Path) -> list[tuple[str, str]]:
    """[(XY, path)] from `git status --porcelain=v1 -z` (renames appear as their new path)."""
    out = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout
    items, parts, i = [], out.split("\0"), 0
    while i < len(parts):
        e = parts[i]
        i += 1
        if len(e) < 4:
            continue
        xy, path = e[:2], e[3:]
        if xy[0] in "RC":
            i += 1                                       # the "from" path follows as its own field
        items.append((xy, path))
    return items


@dataclass
class GitSession:
    root: Path
    name: str                                   # used for the branch name
    exclude: list[str] = field(default_factory=list)       # repo-relative paths never to stage (the run directory)
    branch: str = ""
    base_branch: str = ""
    base_sha: str = ""
    commits: list[str] = field(default_factory=list)
    _preexisting_untracked: set[str] = field(default_factory=set)

    def start(self) -> dict:
        """Check the repository, cut (or reuse) the branch. Returns {branch, base_branch, base_sha, reused}."""
        top = _git(self.root, "rev-parse", "--show-toplevel", check=False)
        if top.returncode != 0:
            raise GitError(f"{self.root} is not inside a git repository")
        self.toplevel = Path(top.stdout.strip()).resolve()
        if _git(self.root, "rev-parse", "--verify", "HEAD", check=False).returncode != 0:
            raise GitError("the repository has no commit yet: commit something first so the agent's branch has a base")
        status = _status(self.root)
        tracked_dirty = [p for xy, p in status if xy != "??" and not self._excluded(p)]
        if tracked_dirty:
            raise GitError("the working tree has uncommitted changes (" + ", ".join(tracked_dirty[:5]) +
                           ("..." if len(tracked_dirty) > 5 else "") + "): commit or stash them first, so that the agent's commits contain only its own work")
        self._preexisting_untracked = {p for xy, p in status if xy == "??"}
        self.base_branch = _git(self.root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        self.base_sha = _git(self.root, "rev-parse", "HEAD").stdout.strip()
        reused = self.base_branch.startswith(BRANCH_PREFIX)
        if reused:
            self.branch = self.base_branch
        else:
            self.branch = f"{BRANCH_PREFIX}{self.name}"
            n = 1
            while _git(self.root, "rev-parse", "--verify", "--quiet", f"refs/heads/{self.branch}", check=False).returncode == 0:
                n += 1
                self.branch = f"{BRANCH_PREFIX}{self.name}-{n}"
            _git(self.root, "checkout", "-b", self.branch)
        return {"branch": self.branch, "base_branch": self.base_branch, "base_sha": self.base_sha, "reused": reused}

    def _excluded(self, path: str) -> bool:
        return any(path == e or path.startswith(e.rstrip("/") + "/") for e in self.exclude)

    def commit_step(self, step: int, summary: str) -> dict | None:
        """Commit what the agent changed in this step; None if nothing changed."""
        to_add = []
        for xy, path in _status(self.root):
            if self._excluded(path):
                continue
            if xy == "??":
                if path not in self._preexisting_untracked:
                    to_add.append(path)
            else:
                to_add.append(path)
        if not to_add:
            return None
        _git(self.root, "add", "-A", "--", *to_add)
        staged = _git(self.root, "diff", "--cached", "--name-only", "-z").stdout.split("\0")
        files = [f for f in staged if f]
        if not files:
            return None
        msg = f"miniagent: step {step}: {summary}"[:200]
        _git(self.root, *AUTHOR, "commit", "--no-verify", "-q", "-m", msg)
        sha = _git(self.root, "rev-parse", "HEAD").stdout.strip()
        self.commits.append(sha)
        return {"sha": sha, "files": files}

    def summary(self) -> str:
        if not self.commits:
            return f"git: branch {self.branch}: the agent made no commits"
        stat = _git(self.root, "diff", "--shortstat", f"{self.base_sha}..HEAD", check=False).stdout.strip()
        back = ""
        if not self.base_branch.startswith(BRANCH_PREFIX):
            back = f"; your branch {self.base_branch!r} is untouched: `git checkout {self.base_branch}` to return"
        return f"git: {len(self.commits)} commit(s) on {self.branch} ({stat or 'no diff'}){back}"
