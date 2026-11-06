"""Sandboxing the agent's shell commands with the operating system's own mechanism, no Docker needed:

* macOS: `sandbox-exec` (Seatbelt) with a generated profile;
* Linux: `bwrap` (bubblewrap) with a read-only view of the filesystem.

The same contract as the Docker sandbox (`miniagent.sandbox.Sandbox`): `exec(command, timeout)` returns a ShellResult, it is a context
manager, and an infrastructure failure is a `SandboxError`, never something shown to the model as a tool result.

What it enforces: writes only inside the workspace and a private temp directory; no network unless asked; a minimal environment (no API
keys); the command gets its own process group, is killed on timeout, and its output is capped. What it does NOT do: hide the rest of the
filesystem. Reads are allowed everywhere (the interpreter, libraries and tools must be readable) except a deny-list of credential
locations (~/.ssh, ~/.aws, keychains, ...), so this is a smaller guarantee than the Docker sandbox, which mounts nothing but the workspace.
There are no CPU or memory limits (macOS has no reliable way): only the timeout. Seatbelt (`sandbox-exec`) is deprecated by Apple but is
what Claude Code and Codex use on macOS.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from .proc import ShellResult
from .sandbox import SandboxError, _run_capped

# credential and secret locations under $HOME that a sandboxed command may not even read
SECRET_DIRS = [".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gcloud", ".config/gh", ".config/op", ".netrc", ".npmrc",
               ".pypirc", ".git-credentials", ".bash_history", ".zsh_history", "Library/Keychains", "Library/Cookies", "Library/Safari",
               "Library/Application Support/Google/Chrome", "Library/Application Support/Firefox", "Library/Messages", "Library/Mail"]
SEATBELT = "/usr/bin/sandbox-exec"


def _q(path: str | Path) -> str:
    """A path as an SBPL string literal."""
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(workdir: Path, tmpdir: Path, *, network: bool = False, home: Path | None = None) -> str:
    """The Seatbelt profile. Pure, so every rule can be tested without running anything."""
    home = home or Path.home()
    rules = [
        "(version 1)", "(deny default)",
        "(allow process-fork)", "(allow process-exec)", "(allow signal (target self))", "(allow sysctl-read)",
        "(allow mach-lookup)", "(allow ipc-posix-shm)", "(allow pseudo-tty)",
        "(allow file-read*)",
        f"(allow file-write* (subpath {_q(workdir)}) (subpath {_q(tmpdir)}) (literal \"/dev/null\") (literal \"/dev/dtracehelper\") "
        "(regex #\"^/dev/tty\") (regex #\"^/dev/fd/\"))",
    ]
    rules += [f"(deny file-read* (subpath {_q(home / d)}))" for d in SECRET_DIRS]        # later rules win: these override the allow above
    if network:
        rules += ["(allow network*)"]
    return "\n".join(rules) + "\n"


def bwrap_argv(command: str, workdir: Path, tmpdir: Path, *, network: bool = False, env: dict[str, str] | None = None) -> list[str]:
    """The `bwrap` command line: the whole filesystem read-only, the workspace and a private /tmp writable, fresh /dev and /proc, new
    namespaces (network included unless asked), killed with its parent. Pure, so it can be tested without bubblewrap."""
    argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--bind", str(tmpdir), "/tmp",
            "--bind", str(workdir), str(workdir), "--chdir", str(workdir), "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    if network:
        argv.append("--share-net")
    for k, v in (env or {}).items():
        argv += ["--setenv", k, v]
    return argv + ["/bin/sh", "-c", command]


def kind() -> str | None:
    """Which mechanism this machine has: "seatbelt", "bwrap" or None. Checked by actually running something inside it."""
    try:
        if platform.system() == "Darwin" and os.path.exists(SEATBELT):
            r = subprocess.run([SEATBELT, "-p", "(version 1)(deny default)(allow process-exec)(allow file-read*)", "/usr/bin/true"],
                               capture_output=True, timeout=10)
            return "seatbelt" if r.returncode == 0 else None
        if platform.system() == "Linux" and shutil.which("bwrap"):
            r = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-all", "--die-with-parent", "/bin/true"], capture_output=True, timeout=10)
            return "bwrap" if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


class LocalSandbox:
    def __init__(self, workdir: str | Path, *, network: bool = False, mechanism: str | None = None):
        self.workdir = Path(workdir).resolve()
        self.network = network
        self.mechanism = mechanism or kind()
        if self.mechanism is None:
            raise SandboxError("no OS-level sandbox available here (needs sandbox-exec on macOS or a working bwrap on Linux)")
        self._tmp: Path | None = None
        self._profile: Path | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> "LocalSandbox":
        self._tmp = Path(tempfile.mkdtemp(prefix="miniagent-sb-")).resolve()
        if self.mechanism == "seatbelt":
            self._profile = self._tmp.parent / (self._tmp.name + ".sb")
            self._profile.write_text(seatbelt_profile(self.workdir, self._tmp, network=self.network))
        return self

    def close(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            if self._profile is not None:
                self._profile.unlink(missing_ok=True)
            self._tmp = self._profile = None

    def __enter__(self) -> "LocalSandbox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- running
    def env(self) -> dict[str, str]:
        """A minimal environment: no API keys, nothing from the user's shell but PATH and the locale."""
        assert self._tmp is not None
        tmp = "/tmp" if self.mechanism == "bwrap" else str(self._tmp)          # bwrap mounts the private temp dir at /tmp
        e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "TMPDIR": tmp, "GIT_PAGER": "cat",
             "PAGER": "cat", "PYTHONDONTWRITEBYTECODE": "1"}
        for k in ("LANG", "LC_ALL", "LC_CTYPE", "TERM", "VIRTUAL_ENV"):
            if k in os.environ:
                e[k] = os.environ[k]
        return e

    def argv(self, command: str) -> list[str]:
        """The full command line that runs `command` inside the sandbox (used for background jobs too)."""
        if self._tmp is None:
            raise SandboxError("the sandbox is not started")
        env = self.env()
        if self.mechanism == "bwrap":
            return bwrap_argv(command, self.workdir, self._tmp, network=self.network, env=env)
        env_args = [f"{k}={v}" for k, v in env.items()]
        return ["/usr/bin/env", "-i", *env_args, SEATBELT, "-f", str(self._profile), "/bin/sh", "-c", f"cd {_sh_quote(str(self.workdir))} && {command}"]

    def exec(self, command: str, timeout: float = 60) -> ShellResult:
        try:
            rc, out, err, timed_out = _run_capped(self.argv(command), timeout)
        except OSError as e:
            raise SandboxError(f"could not start the sandboxed command: {e}")
        return ShellResult(rc, out, err, timed_out)


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
