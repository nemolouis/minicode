"""Resumable sessions and file checkpoints (`Agent(session=True)`, `miniagent --resume DIR`, `miniagent --rewind DIR --to-step N`).

Next to `trajectory.jsonl` the agent keeps
  session.json              the message list the model sees, the step counter, the todo list, and for every step the message-list
                            length before that step's model call (the places a session can be rewound to)
  checkpoints/step-N/       a copy of the workspace taken at the start of step N, only for steps whose tool calls may change files
                            (anything that is not read-only), plus `manifest.json` (every file that existed then)

Rewinding to step N puts the files back to how they were when step N began and cuts the conversation to just before step N's model
call; a later `--resume` continues from there with a new instruction. Limits, on purpose:
  * a compaction rewrites the history, so a session can only be rewound to steps after its latest compaction;
  * a checkpoint skips files over 1 MB and stops (and says so) once a workspace copy would exceed 50 MB: such files are neither
    restored nor deleted;
  * `.git` and the usual vendored/venv/cache directories are not copied. Only the workspace is restored, nothing the commands did elsewhere.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .tools import SKIP_DIRS

MAX_FILE_BYTES = 1_000_000
MAX_SNAPSHOT_BYTES = 50_000_000
VERSION = 1


def exclusions(workspace: Path, run_dir: Path) -> list[str]:
    """Directories inside the workspace that belong to the agent's own bookkeeping and must never be copied or deleted: the run
    directory, and its parent (`runs/`) when that is a proper subdirectory of the workspace."""
    out = []
    for d in (run_dir, run_dir.parent):
        d = d.resolve()
        if d != workspace and workspace in d.parents:
            out.append(d.relative_to(workspace).as_posix())
    return out


def _files(root: Path, exclude=()):
    skip = set(exclude)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and (d if rel_dir == "." else f"{rel_dir}/{d}") not in skip)
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.is_file() and not p.is_symlink():
                yield p


def _dirs(root: Path, exclude=()) -> set[str]:
    skip = set(exclude)
    found = set()
    for dirpath, dirnames, _ in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and (d if rel_dir == "." else f"{rel_dir}/{d}") not in skip]
        found.update((d if rel_dir == "." else f"{rel_dir}/{d}") for d in dirnames)
    return found


def snapshot(workspace: Path, dest: Path, exclude=()) -> dict:
    """Copy the workspace into `dest`. Returns {files, skipped_large, complete}."""
    dest.mkdir(parents=True, exist_ok=True)
    manifest, skipped, total, complete = [], [], 0, True
    for p in _files(workspace, exclude):
        rel = p.relative_to(workspace).as_posix()
        manifest.append(rel)
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped.append(rel)
            continue
        if total + size > MAX_SNAPSHOT_BYTES:
            complete = False
            skipped.append(rel)
            continue
        total += size
        out = dest / "files" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
    (dest / "manifest.json").write_text(json.dumps({"files": manifest, "skipped": skipped, "complete": complete, "exclude": list(exclude),
                                                      "dirs": sorted({str(Path(m).parent.as_posix()) for m in manifest} | _dirs(workspace, exclude))}))
    return {"files": len(manifest), "skipped_large": len(skipped), "complete": complete}


def restore(workspace: Path, src: Path) -> dict:
    """Make the workspace match the snapshot: restored files overwritten, files created since then deleted."""
    meta = json.loads((src / "manifest.json").read_text())
    existed, skipped, exclude = set(meta["files"]), set(meta["skipped"]), meta.get("exclude", [])
    removed = restored = 0
    for p in list(_files(workspace, exclude)):
        rel = p.relative_to(workspace).as_posix()
        if rel not in existed:
            p.unlink()
            removed += 1
    for rel in existed - skipped:
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "files" / rel, target)
        restored += 1
    kept = set(meta.get("dirs", []))
    for d in sorted((p for p in workspace.rglob("*") if p.is_dir() and not (SKIP_DIRS & set(p.relative_to(workspace).parts))),
                    key=lambda p: -len(p.parts)):
        rel = d.relative_to(workspace).as_posix()
        if rel in kept or any(rel == e or rel.startswith(e + "/") for e in exclude):
            continue
        try:
            d.rmdir()                                  # a directory created since the snapshot, now empty
        except OSError:
            pass
    return {"restored": restored, "removed": removed, "skipped_large": len(skipped)}


def save_session(run_dir: Path, state: dict) -> None:
    tmp = run_dir / "session.json.tmp"
    tmp.write_text(json.dumps({"version": VERSION, **state}, ensure_ascii=False))
    os.replace(tmp, run_dir / "session.json")


def load_session(run_dir: str | Path) -> dict:
    p = Path(run_dir) / "session.json"
    if not p.is_file():
        raise FileNotFoundError(f"no session.json in {run_dir}: the run was not started with a session (miniagent writes one by default)")
    state = json.loads(p.read_text())
    if state.get("version") != VERSION:
        raise ValueError(f"unsupported session version {state.get('version')!r}")
    state["step_index"] = {int(k): v for k, v in state.get("step_index", {}).items()}
    state["todos_at"] = {int(k): v for k, v in state.get("todos_at", {}).items()}
    return state


def checkpoint_steps(run_dir: Path) -> list[int]:
    d = run_dir / "checkpoints"
    return sorted(int(p.name.split("-")[1]) for p in d.glob("step-*")) if d.is_dir() else []


def rewind(run_dir: str | Path, step: int, workspace: str | Path) -> dict:
    """Rewind to the start of `step`. Raises ValueError with the reason if that step cannot be reached."""
    run_dir, workspace = Path(run_dir), Path(workspace).resolve()
    state = load_session(run_dir)
    if step not in state["step_index"]:
        raise ValueError(f"cannot rewind to step {step}: rewindable steps are {sorted(state['step_index'])} "
                         "(steps before the latest compaction are gone)")
    if Path(state["workspace"]).resolve() != workspace:
        raise ValueError(f"this session ran in {state['workspace']}, not {workspace}")
    # State at the start of `step` = the checkpoint of the first step >= `step` that could change files; none = nothing changed since.
    later = [s for s in checkpoint_steps(run_dir) if s >= step]
    result = {"restored": 0, "removed": 0, "skipped_large": 0}
    if later:
        snap = run_dir / "checkpoints" / f"step-{later[0]}"
        if not json.loads((snap / "manifest.json").read_text())["complete"]:
            raise ValueError(f"the checkpoint at step {later[0]} is incomplete (workspace over {MAX_SNAPSHOT_BYTES // 10**6} MB); not restoring")
        result = restore(workspace, snap)
    for s in checkpoint_steps(run_dir):
        if s >= step:
            shutil.rmtree(run_dir / "checkpoints" / f"step-{s}")
    state["messages"] = state["messages"][:state["step_index"][step]]
    state["step"] = step - 1
    state["step_index"] = {k: v for k, v in state["step_index"].items() if k < step}
    state["todos"] = state["todos_at"].get(step, [])
    state["todos_at"] = {k: v for k, v in state["todos_at"].items() if k < step}
    state["last_input"] = 0
    save_session(run_dir, state)
    return {"step": step, **result, "messages": len(state["messages"])}
