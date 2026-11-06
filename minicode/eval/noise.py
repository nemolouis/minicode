"""The noise floor of an evaluation: how much do two runs of the SAME configuration on the SAME tasks differ?

Every comparison in this project is a difference between two runs, and a model called at temperature 0 is still not deterministic.
Until this was measured the only evidence was indirect (an almost-unused feature differed from the baseline by 4 of 39 passes).
`minicode-eval noise DIR DIR [DIR ...]` takes result directories of replicate runs (same config hash, same tasks) and reports:

* the pass rate of each replicate, and the largest difference between any two;
* how many replicate PAIRS the comparison tests (compare.py) would call "different" although nothing changed: the false-alarm rate;
* the flaky tasks (those that both pass and fail across the replicates);
* a rule of thumb for the smallest difference worth looking at.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from .report import compare, load_records, per_task_pass, valid
from .stats import mean_sd


class NoiseError(ValueError):
    pass


def load_replicate(d: str | Path) -> tuple[dict, list[dict]]:
    d = Path(d)
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError) as e:
        raise NoiseError(f"{d}: cannot read meta.json ({e})")
    return meta, valid(load_records(d))


def noise_report(dirs: list[str | Path]) -> dict:
    if len(dirs) < 2:
        raise NoiseError("need at least two replicate result directories")
    loaded = [load_replicate(d) for d in dirs]
    hashes = {m.get("config_hash") for m, _ in loaded}
    if len(hashes) != 1 or None in hashes:
        raise NoiseError(f"the directories are not replicates of one config (config hashes: {sorted(map(str, hashes))})")
    task_sets = {frozenset(m.get("task_hashes", {}).items()) for m, _ in loaded}
    if len(task_sets) != 1:
        raise NoiseError("the replicates ran different tasks (or the tasks changed between them)")
    recs = [r for _, r in loaded]
    rates = [_mean_rate(r) for r in recs]
    mean, sd = mean_sd(rates)
    pairs = []
    for (i, a), (j, b) in combinations(enumerate(recs), 2):
        c = compare(a, b)
        pairs.append({"a": str(dirs[i]), "b": str(dirs[j]), "diff": c["mean_diff"], "p_paired_tasks": c["p_paired_tasks"],
                      "p_pooled_runs": c["p_pooled_runs"], "verdict": c["verdict"],
                      "bootstrap_excludes_zero": c["diff_ci95"][0] > 0 or c["diff_ci95"][1] < 0})
    per_task = [per_task_pass(r) for r in recs]
    flaky = sorted(t for t in per_task[0] if len({round(p.get(t, float("nan")), 6) for p in per_task}) > 1)
    unstable = sorted(t for t in per_task[0] if 0 < sum(p.get(t, 0) for p in per_task) / len(per_task) < 1)
    largest = max(abs(p["diff"]) for p in pairs)
    return {
        "config": loaded[0][0]["config"].get("name"), "config_hash": next(iter(hashes)), "replicates": len(recs),
        "runs_per_replicate": [len(r) for r in recs], "tasks": len(per_task[0]),
        "pass_rates": rates, "mean": mean, "sd": sd, "largest_pairwise_difference": largest,
        "pairs": pairs,
        "false_alarms_tests": sum(1 for p in pairs if p["verdict"] == "different"),
        "false_alarms_bootstrap": sum(1 for p in pairs if p["bootstrap_excludes_zero"]),
        "flaky_tasks": flaky, "unstable_tasks": unstable,
    }


def _mean_rate(records: list[dict]) -> float:
    pt = per_task_pass(records)
    return sum(pt.values()) / len(pt) if pt else float("nan")


def markdown_noise(r: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    n = len(r["pairs"])
    sd = "n/a (two replicates)" if r["sd"] != r["sd"] else f"{r['sd'] * 100:.1f} points"
    lines = [
        f"# Noise floor: `{r['config']}` (config hash `{r['config_hash']}`)", "",
        f"{r['replicates']} replicates of the same configuration on the same {r['tasks']} tasks "
        f"({', '.join(map(str, r['runs_per_replicate']))} valid runs).", "",
        "| replicate | pass rate (task average) |", "|---|---|",
        *[f"| {i + 1} | {pct(x)} |" for i, x in enumerate(r["pass_rates"])], "",
        f"- mean {pct(r['mean'])}, standard deviation between replicates {sd}",
        f"- **largest difference between two replicates: {r['largest_pairwise_difference'] * 100:.1f} points**: differences up to this size "
        "were produced with nothing changed",
        f"- false alarms among the {n} replicate pair(s): the compare tests said \"different\" {r['false_alarms_tests']}×; "
        f"the task-bootstrap interval alone excluded 0 {r['false_alarms_bootstrap']}×",
        f"- flaky tasks (pass rate differs between replicates): {len(r['flaky_tasks'])}"
        + (f" ({', '.join(r['flaky_tasks'])})" if r["flaky_tasks"] else ""),
        "", "How to use this: a difference between two configurations smaller than the largest difference above cannot be told apart from "
        "re-running the same thing. With few replicates this is itself an estimate: more replicates make it more reliable.", "",
    ]
    return "\n".join(lines)
