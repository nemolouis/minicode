"""Long-running shell commands (dev servers, watchers, slow test suites) without blocking the loop: start one, read what it printed
since the last look, kill it. Three tools sit on top (see features.py): bash_background, bash_output, bash_kill.

Safety: the same deny-list as `bash`; the command runs in its own process group with the same minimal environment handling as `bash`
(cwd = workspace root); output goes to a file outside the workspace, read in bounded chunks; at most MAX_JOBS jobs at a time; every
job is killed when the run ends (and at interpreter exit). Not available with the Docker sandbox: a host process would defeat it."""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

MAX_JOBS = 4
MAX_LOG_BYTES = 5_000_000
READ_CHUNK = 20_000


class JobError(Exception):
    pass


class Job:
    def __init__(self, job_id: str, command: str, proc: subprocess.Popen, log: Path):
        self.id, self.command, self.proc, self.log = job_id, command, proc, log
        self.read_pos = 0
        self.started = time.time()

    @property
    def running(self) -> bool:
        return self.proc.poll() is None


class BackgroundJobs:
    def __init__(self, cwd: Path, env: dict | None = None):
        self.cwd, self.env = Path(cwd), env
        self._dir = Path(tempfile.mkdtemp(prefix="miniagent-bg-"))
        self._jobs: dict[str, Job] = {}
        self._n = 0
        self._lock = threading.Lock()
        atexit.register(self.kill_all)

    def start(self, command: str) -> Job:
        with self._lock:
            if sum(j.running for j in self._jobs.values()) >= MAX_JOBS:
                raise JobError(f"{MAX_JOBS} background jobs are already running: kill one with bash_kill first")
            self._n += 1
            job_id = f"job{self._n}"
            log = self._dir / f"{job_id}.log"
            f = log.open("wb")
            proc = subprocess.Popen(command, shell=True, cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL, stdout=f,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            f.close()                                   # the child keeps its own descriptor
            job = Job(job_id, command, proc, log)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobError(f"no such job {job_id!r}; running or finished jobs: {', '.join(self._jobs) or 'none'}")
        return job

    def read(self, job_id: str, tail: int | None = None) -> str:
        """New output since the last read (or the last `tail` lines of everything), plus the job's state."""
        job = self.get(job_id)
        size = job.log.stat().st_size
        with job.log.open("rb") as f:
            if tail:
                f.seek(max(size - MAX_LOG_BYTES, 0))
                text = b"".join(f.readlines()[-tail:]).decode("utf-8", "replace")
            else:
                f.seek(job.read_pos)
                data = f.read(READ_CHUNK)
                job.read_pos += len(data)
                text = data.decode("utf-8", "replace")
                if job.read_pos < size:
                    text += f"\n... ({size - job.read_pos} more bytes: call bash_output again)"
        state = "running" if job.running else f"exited with code {job.proc.returncode}"
        if size > MAX_LOG_BYTES and job.running:
            self.kill(job_id)
            state = f"killed: its output passed {MAX_LOG_BYTES // 10**6} MB"
        return f"[{job.id}: {state}]\n{text}".rstrip()

    def kill(self, job_id: str) -> str:
        job = self.get(job_id)
        if job.running:
            self._kill_group(job, signal.SIGTERM)
            try:
                job.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._kill_group(job, signal.SIGKILL)
                job.proc.wait(timeout=3)
        self._kill_group(job, signal.SIGKILL)           # stragglers in the group
        return f"{job.id} stopped (exit code {job.proc.returncode})"

    @staticmethod
    def _kill_group(job: Job, sig: int) -> None:
        try:
            os.killpg(job.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def kill_all(self) -> None:
        for job in list(self._jobs.values()):
            try:
                self.kill(job.id)
            except Exception:
                pass
