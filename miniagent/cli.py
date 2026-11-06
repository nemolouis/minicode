import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import json

from .agent import Agent
from .budget import Budget
from .permissions import Permissions
from .mcp.client import mcp_session
from .client import APIError, ChatClient
from .tools import Workspace
from .trajectory import Trajectory

DEFAULT_MODEL = "deepseek-chat"


def _make_confirm(auto: bool):
    approve_all = auto

    def confirm(name: str, args: dict) -> bool:
        nonlocal approve_all
        if approve_all:
            return True
        shown = args.get("command") or args.get("path") or ""
        ans = input(f"\n[confirm] {name}: {shown}\n  allow? [y]es / [n]o / [a]ll: ").strip().lower()
        if ans == "a":
            approve_all = True
        return ans in ("y", "a")

    return confirm


def _printer():
    def show(text: str) -> None:
        print(text, end="", flush=True)
    return show


def _new_run_dir(root: Path) -> Path:
    """`root/<timestamp>`, with a numeric suffix if that name is taken (two runs in the same second must not share a trajectory)."""
    base = time.strftime("%Y%m%d-%H%M%S")
    d, n = root / base, 1
    while d.exists():
        d, n = root / f"{base}-{n}", n + 1
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="miniagent")
    ap.add_argument("task", nargs="?", help="what the agent should do (with --resume: the follow-up instruction)")
    ap.add_argument("-w", "--workspace", default=".", help="repo root the agent may touch")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("-y", "--yes", action="store_true", help="auto-approve write/edit/bash (deny-list still applies)")
    ap.add_argument("--context-limit", type=int, default=100_000,
                    help="compact history when input tokens exceed 70%% of this; set low (e.g. 3000) to exercise compaction")
    ap.add_argument("--keep-recent", type=int, default=8, help="messages kept verbatim when compacting")
    ap.add_argument("--compact-regrowth", type=float, default=1.0,
                    help="after a compaction that cannot get under the threshold, wait for this much input growth before the next one (0 = every step)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--mcp-config", help="JSON file with a list of MCP servers: [{name, command, allow, trusted?}]")
    ap.add_argument("--disable-tools", default="", help="comma-separated built-in tools to hide, e.g. bash,grep,read_file")
    ap.add_argument("--features", default="", help="comma-separated opt-in features: memory (AGENTS.md), todo, explore (read-only sub-agent), glob")
    ap.add_argument("--permissions", help="JSON file with allow / ask / deny rules, e.g. {\"deny\": [\"bash(curl*)\"], \"allow\": [\"bash(pytest*)\"]}")
    ap.add_argument("--require-read", action="store_true", help="edit_file refuses files the agent has not read in this session")
    ap.add_argument("--diagnostics", action="store_true", help="warn when a .py/.json file the agent just wrote no longer parses")
    ap.add_argument("--resume", metavar="RUN_DIR", help="continue the session saved in this run directory (task = follow-up instruction)")
    ap.add_argument("--rewind", metavar="RUN_DIR", help="put the workspace and the conversation back to the start of --to-step, then exit")
    ap.add_argument("--to-step", type=int, help="step to rewind to (with --rewind)")
    ap.add_argument("--no-session", action="store_true", help="do not save session.json / workspace checkpoints")
    ap.add_argument("--stuck-warn", type=int, default=0, metavar="N",
                    help="after N identical calls with identical results in a row, tell the model (0 = off); the run stops at --stuck-stop")
    ap.add_argument("--stuck-stop", type=int, default=6, metavar="N", help="end the run with status 'stuck' after N such repeats")
    ap.add_argument("--stream", action="store_true", help="print the model's reply as it is generated (OpenAI-compatible endpoints)")
    ap.add_argument("--output-budget", action="store_true", help="shrink the size of one tool output as the context fills up")
    ap.add_argument("--max-cost", type=float, help="stop before the request that would exceed this many USD (needs a price in --prices)")
    ap.add_argument("--max-total-tokens", type=int, help="stop before the request that would exceed this many input+output tokens")
    ap.add_argument("--prices", default="configs/prices.json", help="price table used for --max-cost")
    ap.add_argument("--no-parallel", action="store_true", help="run tool calls one at a time")
    ap.add_argument("--no-validate", action="store_true", help="do not check tool arguments against their schemas")
    ap.add_argument("--sandbox", action="store_true", help="run bash in a Docker container (no network, read-only root, only the workspace mounted)")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    if args.rewind:
        from .session import rewind
        if args.to_step is None:
            ap.error("--rewind needs --to-step N")
        try:
            print(rewind(args.rewind, args.to_step, args.workspace))
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        return 0
    if not args.task:
        ap.error("a task is required (except with --rewind)")
    resume = None
    if args.resume:
        from .session import load_session
        try:
            resume = load_session(args.resume)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if Path(resume["workspace"]).resolve() != Path(args.workspace).resolve():
            print(f"error: the session ran in {resume['workspace']}; pass that directory with -w", file=sys.stderr)
            return 2
    try:
        permissions = Permissions.from_dict(json.loads(Path(args.permissions).read_text())) if args.permissions else None
    except (OSError, ValueError) as e:
        print(f"error: permissions: {e}", file=sys.stderr)
        return 2

    try:
        client = ChatClient()
    except APIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    budget = None
    if args.max_cost is not None or args.max_total_tokens is not None:
        price = None
        if args.max_cost is not None:
            try:
                price = (json.loads(Path(args.prices).read_text()).get(args.model) or {}).get("off_peak")
            except (OSError, ValueError) as e:
                print(f"error: cannot read prices: {e}", file=sys.stderr)
                return 2
            if not price:
                print(f"error: no price for model {args.model!r} in {args.prices}; use --max-total-tokens instead", file=sys.stderr)
                return 2
        try:
            budget = Budget(max_cost_usd=args.max_cost, max_total_tokens=args.max_total_tokens, price=price)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    run_dir = Path(args.resume) if args.resume else _new_run_dir(Path(args.runs_dir))
    servers = json.loads(Path(args.mcp_config).read_text()) if args.mcp_config else []
    from .sandbox import Sandbox, available
    if args.sandbox and not available():
        print("error: the sandbox image is missing or Docker is not running (build it with: miniagent-eval sandbox-build)", file=sys.stderr)
        return 2
    with mcp_session(servers) as mcp_tools, (Sandbox(args.workspace) if args.sandbox else nullcontext()) as sb:
        agent = Agent(
            client=client, model=args.model, workspace=Workspace(args.workspace, sandbox=sb, require_read=args.require_read, diagnostics=args.diagnostics),
            confirm=_make_confirm(args.yes), trajectory=Trajectory(run_dir), max_steps=args.max_steps,
            context_limit=args.context_limit, keep_recent=args.keep_recent, compact_regrowth=args.compact_regrowth,
            temperature=args.temperature, extra_tools=mcp_tools,
            features=tuple(f for f in args.features.split(",") if f), parallel_tools=not args.no_parallel,
            validate_args=not args.no_validate, permissions=permissions, session=not args.no_session,
            stuck_warn_after=args.stuck_warn, stuck_stop_after=args.stuck_stop, budget=budget,
            output_budget=args.output_budget, on_text=_printer() if args.stream else None,
            disabled_tools=tuple(t for t in args.disable_tools.split(",") if t),
        )
        res = agent.run(args.task, resume=resume)
    print(f"\n=== {res.status} | {res.steps} steps | {res.input_tokens} in / {res.output_tokens} out tokens ===")
    if not args.stream:                              # with --stream the text has already been printed as it arrived
        print(res.final_text)
    if res.status == "interrupted":
        print(f"interrupted; continue with: miniagent \"<next instruction>\" -w {args.workspace} --resume {run_dir}")
        print(f"trajectory: {run_dir / 'trajectory.jsonl'}")
        return 130
    print(f"trajectory: {run_dir / 'trajectory.jsonl'}")
    return 0 if res.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
