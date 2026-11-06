"""Run-level statistics in compare, and the noise-floor report."""
import json
import math

import pytest

from minicode.eval.noise import NoiseError, markdown_noise, noise_report
from minicode.eval.report import compare, markdown_compare
from minicode.eval.stats import fisher_exact, mean_sd, sign_flip_test, verdict, wilson


# ---------------------------------------------------------------- the tests themselves
def test_fisher_exact_known_values():
    assert fisher_exact(35, 4, 39, 0) == pytest.approx(0.1148, abs=1e-3)           # the flash vs todo arm of 2026-11-06
    assert fisher_exact(35, 4, 36, 3) == pytest.approx(1.0, abs=1e-6)
    assert fisher_exact(8, 2, 1, 5) == pytest.approx(0.0350, abs=1e-3)              # textbook table
    assert fisher_exact(0, 0, 0, 0) == 1.0
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)


def test_sign_flip_exact():
    assert sign_flip_test([1 / 3] * 4 + [0] * 9) == pytest.approx(2 / 16)           # 4 tasks improved, none regressed: p = 0.125
    assert sign_flip_test([0, 0, 0]) == 1.0 and sign_flip_test([]) == 1.0
    assert sign_flip_test([1.0] * 6) == pytest.approx(2 / 64)
    assert sign_flip_test([1, -1]) == 1.0                                           # balanced: nothing to see
    assert sign_flip_test([3, -1, -1, -1]) == pytest.approx(1.0)                    # |sum| = 0 is the smallest possible
    assert sign_flip_test([1, 1, 1, -1]) == pytest.approx(10 / 16)                  # |sum| = 2: the 10 of 16 sign patterns with |sum| >= 2


def test_sign_flip_monte_carlo_agrees_with_exact_in_the_region_both_can_handle():
    d = [1.0] * 12 + [-1.0] * 3 + [0.5] * 2 + [0.0] * 5                              # 17 non-zero differences: above EXACT_LIMIT
    p_mc = sign_flip_test(d, samples=40_000)
    import minicode.eval.stats as S
    old = S.EXACT_LIMIT
    S.EXACT_LIMIT = 20
    try:
        p_exact = sign_flip_test(d)
    finally:
        S.EXACT_LIMIT = old
    assert p_mc == pytest.approx(p_exact, abs=0.01)


def test_wilson_interval():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32
    lo, hi = wilson(39, 39)
    assert hi == 1.0 and 0.90 < lo < 0.93
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and math.isnan(wilson(0, 0)[0])


def test_verdict_needs_both_tests():
    assert verdict(0.01, 0.01) == "different"
    assert "disagree" in verdict(0.01, 0.2) and "disagree" in verdict(0.2, 0.01)
    assert "no difference" in verdict(0.2, 0.3)
    assert verdict(0.05, 0.01).startswith("inconclusive")                           # exactly alpha is not below alpha


def test_mean_sd():
    assert mean_sd([1, 2, 3]) == (2, 1.0)
    assert math.isnan(mean_sd([5])[1]) and math.isnan(mean_sd([])[0])


# ---------------------------------------------------------------- compare on realistic records
def rec(task, rep, passed, steps=8):
    return {"task_id": task, "repeat": rep, "passed": passed, "failure_class": "pass" if passed else "tests_failed_after_edit", "steps": steps}


def runs(spec):
    """spec: {task: [bool, ...]} -> records"""
    return [rec(t, i, ok) for t, v in spec.items() for i, ok in enumerate(v)]


FLASH = {f"t{i:02d}": [True, True, True] for i in range(13)}
FLASH.update({"t00": [True, True, False], "t01": [True, False, True], "t02": [False, True, True], "t03": [True, True, False]})
TODO = {f"t{i:02d}": [True, True, True] for i in range(13)}


def test_the_flash_todo_case_is_flagged_as_noise():
    c = compare(runs(FLASH), runs(TODO))
    assert c["runs_a"] == [35, 39] and c["runs_b"] == [39, 39]
    assert c["p_paired_tasks"] == pytest.approx(0.125) and c["p_pooled_runs"] == pytest.approx(0.1148, abs=1e-3)
    assert c["diff_ci95"][0] > 0                                                      # the bootstrap alone would have said "significant"
    assert c["verdict"].startswith("inconclusive")
    text = markdown_compare("flash", "todo", c)
    assert "WARNING" in text and "Do not quote this as an improvement" in text and "35/39 → 39/39" in text and "p = 0.125" in text


def test_a_large_consistent_difference_is_called_different():
    weak = {f"t{i:02d}": [False, False, False] for i in range(12)}
    strong = {f"t{i:02d}": [True, True, True] for i in range(12)}
    c = compare(runs(weak), runs(strong))
    assert c["verdict"] == "different" and "WARNING" not in markdown_compare("a", "b", c)


def test_identical_runs_have_no_warning_and_p_one():
    c = compare(runs(FLASH), runs(FLASH))
    assert c["verdict"].startswith("inconclusive") and c["p_paired_tasks"] == 1.0 and c["p_pooled_runs"] == 1.0
    assert "WARNING" not in markdown_compare("a", "a", c)


def test_only_common_tasks_enter_the_run_counts():
    a = runs({"x": [True, False, True], "only_a": [True, True, True]})                 # passing runs on tasks only one side has
    b = runs({"x": [True, True, True], "only_b": [True, True, True]})
    assert compare(a, b)["runs_a"] == [2, 3] and compare(a, b)["runs_b"] == [3, 3]


# ---------------------------------------------------------------- noise report
def write_dir(root, name, spec, config_hash="abc", tasks=None, config_name="flash"):
    d = root / name
    d.mkdir()
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs(spec)))
    (d / "meta.json").write_text(json.dumps({"config": {"name": config_name}, "config_hash": config_hash,
                                             "task_hashes": tasks or {t: "h" for t in spec}}))
    return d


def test_noise_report_numbers(tmp_path):
    a = write_dir(tmp_path, "r1", FLASH)
    b = write_dir(tmp_path, "r2", {**FLASH, "t00": [True, True, True], "t01": [True, True, True], "t02": [True, True, True]})
    c = write_dir(tmp_path, "r3", FLASH)
    r = noise_report([a, b, c])
    assert r["replicates"] == 3 and r["tasks"] == 13 and r["runs_per_replicate"] == [39, 39, 39]
    assert r["pass_rates"][0] == pytest.approx(r["pass_rates"][2]) and r["pass_rates"][1] > r["pass_rates"][0]
    assert r["largest_pairwise_difference"] == pytest.approx(r["pass_rates"][1] - r["pass_rates"][0])
    assert len(r["pairs"]) == 3 and r["sd"] > 0
    assert set(r["flaky_tasks"]) == {"t00", "t01", "t02"}                              # t03 is the same in all three
    assert "t03" in r["unstable_tasks"]                                                # but it does fail sometimes
    text = markdown_noise(r)
    assert "largest difference between two replicates" in text and "t00" in text


def test_noise_report_counts_false_alarms(tmp_path):
    lo = write_dir(tmp_path, "lo", {f"t{i:02d}": [False] * 3 for i in range(12)})
    hi = write_dir(tmp_path, "hi", {f"t{i:02d}": [True] * 3 for i in range(12)})
    r = noise_report([lo, hi])
    assert r["false_alarms_tests"] == 1 and r["false_alarms_bootstrap"] == 1 and r["largest_pairwise_difference"] == 1.0


def test_noise_report_refuses_things_that_are_not_replicates(tmp_path):
    a = write_dir(tmp_path, "a", FLASH)
    with pytest.raises(NoiseError, match="at least two"):
        noise_report([a])
    other_cfg = write_dir(tmp_path, "b", FLASH, config_hash="zzz")
    with pytest.raises(NoiseError, match="not replicates of one config"):
        noise_report([a, other_cfg])
    other_tasks = write_dir(tmp_path, "c", FLASH, tasks={t: "different-hash" for t in FLASH})
    with pytest.raises(NoiseError, match="different tasks"):
        noise_report([a, other_tasks])
    with pytest.raises(NoiseError, match="cannot read"):
        noise_report([a, tmp_path / "missing"])


def test_noise_command(tmp_path, capsys):
    from minicode.eval.cli import main
    a, b = write_dir(tmp_path, "r1", FLASH), write_dir(tmp_path, "r2", TODO)
    out = tmp_path / "noise.md"
    assert main(["noise", str(a), str(b), "--out", str(out)]) == 0
    assert "Noise floor" in capsys.readouterr().out and out.exists() and json.loads(out.with_suffix(".json").read_text())["replicates"] == 2
    assert main(["noise", str(a)]) == 2
