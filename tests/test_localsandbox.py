"""OS-level sandbox: pure profile/argv tests everywhere, real enforcement tests where Seatbelt (macOS) is available."""
import json
import socket
import subprocess
import threading
import time

import pytest

from minicode import localsandbox as L
from minicode.agent import Agent
from minicode.client import ModelResponse, ToolCall
from minicode.sandbox import SandboxError
from minicode.tools import Workspace

seatbelt = pytest.mark.skipif(L.kind() != "seatbelt", reason="needs macOS sandbox-exec")
# the behaviour every OS sandbox must have, whichever mechanism this machine uses. The bwrap path has only been run where bubblewrap
# works (not on the macOS development machine; on GitHub's runners user namespaces are often disabled, then these are skipped).
anysandbox = pytest.mark.skipif(L.kind() is None, reason="needs sandbox-exec (macOS) or a working bwrap (Linux)")


# ---------------------------------------------------------------- pure: what the profile / command line says
def test_seatbelt_profile_rules(tmp_path):
    w, t, home = tmp_path / "work dir", tmp_path / "tmp", tmp_path / "home"
    p = L.seatbelt_profile(w, t, home=home)
    assert p.startswith("(version 1)\n(deny default)")
    assert f'(subpath "{w}")' in p and f'(subpath "{t}")' in p                         # writable: workspace and private temp
    writes = [line for line in p.splitlines() if line.startswith("(allow file-write*")]
    assert len(writes) == 1 and str(home) not in writes[0]
    assert "(allow network" not in p
    assert "(allow network*)" in L.seatbelt_profile(w, t, network=True, home=home)
    denies = [line for line in p.splitlines() if line.startswith("(deny file-read*")]
    assert f'(deny file-read* (subpath "{home}/.ssh"))' in denies and len(denies) == len(L.SECRET_DIRS)
    assert p.index("(allow file-read*)") < p.index("(deny file-read*")                  # the deny comes later, so it wins


def test_seatbelt_profile_escapes_quotes_in_paths(tmp_path):
    p = L.seatbelt_profile(tmp_path / 'a"b', tmp_path / "t", home=tmp_path)
    assert '\\"' in p and 'a"b' not in p.replace('a\\"b', "")


def test_bwrap_command_line(tmp_path):
    w, t = tmp_path / "w", tmp_path / "t"
    a = L.bwrap_argv("pytest -q", w, t, env={"PATH": "/usr/bin", "HOME": "/tmp"})
    assert a[:3] == ["bwrap", "--ro-bind", "/"] and a[-3:] == ["/bin/sh", "-c", "pytest -q"]
    j = " ".join(a)
    assert f"--bind {w} {w}" in j and f"--bind {t} /tmp" in j and "--chdir " + str(w) in j
    for flag in ("--unshare-all", "--die-with-parent", "--clearenv", "--new-session"):
        assert flag in a
    assert "--share-net" not in a and "--setenv PATH /usr/bin" in j
    assert "--share-net" in L.bwrap_argv("x", w, t, network=True)
    assert a.index("--ro-bind") < a.index("--bind")                                      # read-only root first, writable binds layered on top


def test_no_mechanism_means_a_clear_error(monkeypatch):
    monkeypatch.setattr(L, "kind", lambda: None)
    with pytest.raises(SandboxError, match="no OS-level sandbox"):
        L.LocalSandbox("/tmp")


def test_the_secret_list_covers_the_usual_suspects():
    for d in (".ssh", ".aws", ".gnupg", "Library/Keychains", ".netrc", ".git-credentials"):
        assert d in L.SECRET_DIRS


# ---------------------------------------------------------------- real enforcement (Seatbelt)
@pytest.fixture
def box(tmp_path):
    w = tmp_path / "work"
    w.mkdir()
    with L.LocalSandbox(w) as sb:
        yield sb, w


def run(sb, cmd, timeout=20):
    r = sb.exec(cmd, timeout)
    return r.returncode, (r.stdout + r.stderr)


@anysandbox
def test_commands_run_in_the_workspace_and_can_write_there(box):
    sb, w = box
    rc, out = run(sb, "pwd; echo hello > made.txt; cat made.txt")
    assert rc == 0 and str(w.resolve()) in out and "hello" in out and (w / "made.txt").read_text() == "hello\n"


@anysandbox
def test_writes_outside_the_workspace_are_refused(box, tmp_path):
    sb, w = box
    rc, out = run(sb, f"echo x > {tmp_path}/outside.txt; echo rc=$?; echo y > ../sibling.txt; echo rc=$?; touch /private/etc/nope 2>&1; echo rc=$?")
    assert out.count("rc=1") >= 2 and "Operation not permitted" in out
    assert not (tmp_path / "outside.txt").exists() and not (tmp_path / "sibling.txt").exists()


@anysandbox
def test_the_private_temp_dir_is_writable_and_is_removed_afterwards(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    with L.LocalSandbox(w) as sb:
        tmp = sb._tmp
        rc, out = run(sb, "echo scratch > $TMPDIR/s.txt && cat $TMPDIR/s.txt")
        assert rc == 0 and "scratch" in out
    assert not tmp.exists()


@anysandbox
def test_no_network(box):
    sb, w = box
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        rc, out = run(sb, f"python3 -c \"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3); print('CONNECTED')\" 2>&1 | tail -1")
        assert "CONNECTED" not in out and "PermissionError" in out
    finally:
        srv.close()


@anysandbox
def test_network_can_be_allowed_on_request(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: srv.accept(), daemon=True).start()
    try:
        with L.LocalSandbox(w, network=True) as sb:
            rc, out = run(sb, f"python3 -c \"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3); print('CONNECTED')\" 2>&1 | tail -1")
            assert "CONNECTED" in out
    finally:
        srv.close()


@anysandbox
def test_the_environment_is_minimal_and_has_no_secrets(box, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
    monkeypatch.setenv("SOME_TOKEN", "t0ken")
    sb, w = box
    rc, out = run(sb, "env")
    assert "sk-secret" not in out and "t0ken" not in out and "DEEPSEEK" not in out
    assert "PATH=" in out and "HOME=" in out
    names = {line.split("=")[0] for line in out.splitlines() if "=" in line}
    assert names <= {"PATH", "HOME", "TMPDIR", "GIT_PAGER", "PAGER", "PYTHONDONTWRITEBYTECODE", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
                     "VIRTUAL_ENV", "PWD", "SHLVL", "_", "OLDPWD"}


@seatbelt
def test_credential_folders_cannot_be_read(tmp_path):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE KEY")
    (home / "notes.txt").write_text("harmless")
    w = tmp_path / "w"
    w.mkdir()
    profile = tmp_path / "p.sb"
    profile.write_text(L.seatbelt_profile(w.resolve(), tmp_path.resolve(), home=home.resolve()))
    def sandboxed(cmd):
        return subprocess.run([L.SEATBELT, "-f", str(profile), "/bin/sh", "-c", cmd], capture_output=True, text=True, cwd=w)
    assert sandboxed(f"cat {home}/notes.txt").stdout == "harmless"                       # other files stay readable (the tools must work)
    r = sandboxed(f"cat {home}/.ssh/id_rsa")
    assert "PRIVATE KEY" not in r.stdout and r.returncode != 0 and "Operation not permitted" in r.stderr


@anysandbox
def test_timeout_kills_the_command_and_its_children(box, tmp_path):
    sb, w = box
    marker = w / "alive"
    r = sb.exec(f"(while true; do touch {marker}; sleep 0.1; done) & wait", timeout=1.5)
    assert r.timed_out
    marker.unlink(missing_ok=True)
    time.sleep(0.5)
    assert not marker.exists()                                                           # the background loop died with the group


@anysandbox
def test_output_is_capped(box):
    sb, w = box
    rc, out = run(sb, "python3 -c \"print('x' * 3_000_000)\"")
    assert "more bytes dropped" in out and len(out) < 2_100_000


@anysandbox
def test_the_agent_bash_tool_runs_inside_it(tmp_path):
    w = tmp_path / "w"
    w.mkdir()

    class Script:
        def __init__(self, script):
            self.script = list(script)

        def chat(self, **kw):
            content, tok = self.script.pop(0)
            calls = [c for c in content if isinstance(c, ToolCall)]
            return ModelResponse("".join(c for c in content if isinstance(c, str)), calls, "tool_calls" if calls else "stop",
                                 {"input_tokens": tok, "output_tokens": 1})

    c = lambda i, cmd: ToolCall(i, "bash", json.dumps({"command": cmd}))
    with L.LocalSandbox(w) as sb:
        ag = Agent(Script([([c("1", f"echo ok > in.txt; echo bad > {tmp_path}/out.txt; echo rc=$?")], 1), (["done"], 1)]), "m",
                   Workspace(w, sandbox=sb))
        assert ag.run("t").status == "done"
    assert (w / "in.txt").exists() and not (tmp_path / "out.txt").exists()


@anysandbox
def test_background_jobs_run_inside_the_sandbox(tmp_path):
    w = tmp_path / "w"
    w.mkdir()

    class Idle:
        def chat(self, **kw):
            return ModelResponse("", [], "stop", {"input_tokens": 1, "output_tokens": 1})

    with L.LocalSandbox(w) as sb:
        ag = Agent(Idle(), "m", Workspace(w, sandbox=sb), features=("background",))            # allowed: this sandbox can wrap the command
        job = ag.background.start(f"echo from-job > j.txt; echo x > {tmp_path}/escape.txt; echo done")
        deadline = time.time() + 5
        while job.running and time.time() < deadline:
            time.sleep(0.05)
        assert (w / "j.txt").read_text() == "from-job\n" and not (tmp_path / "escape.txt").exists()
        ag.background.kill_all()


def test_background_is_still_refused_with_docker_style_sandboxes(tmp_path):
    class DockerLike:
        pass

    with pytest.raises(ValueError, match="Docker sandbox"):
        Agent(None, "m", Workspace(tmp_path, sandbox=DockerLike()), features=("background",))


# ---------------------------------------------------------------- CLI
def test_cli_sandbox_kinds(tmp_path, monkeypatch, capsys):
    from minicode import cli

    class Quick:
        def chat(self, **kw):
            return ModelResponse("ok", [], "stop", {"input_tokens": 1, "output_tokens": 1})

    (tmp_path / "w").mkdir()
    base = ["go", "-w", str(tmp_path / "w"), "-m", "fake", "--runs-dir", str(tmp_path / "runs")]
    monkeypatch.setattr(cli, "ChatClient", lambda: Quick())
    other = "bwrap" if L.kind() != "bwrap" else "seatbelt"
    assert cli.main(base + ["--sandbox", other]) == 2 and f"--sandbox {other} is not available" in capsys.readouterr().err
    if L.kind():
        assert cli.main(base + ["--sandbox", L.kind()]) == 0
        assert cli.main(base + ["--sandbox", "auto"]) == 0
    monkeypatch.setattr(L, "kind", lambda: None)
    import minicode.sandbox as S
    monkeypatch.setattr(S, "available", lambda image=S.IMAGE: False)
    assert cli.main(base + ["--sandbox", "auto"]) == 2 and "sandbox image is missing" in capsys.readouterr().err
    assert cli.main(base + ["--sandbox"]) == 2                                              # bare --sandbox still means docker
