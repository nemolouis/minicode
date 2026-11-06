# Eval report: 20261106-120253-feat-perms-flash

- config: `feat-perms-flash` (hash `22b222d7b3`), model `deepseek-flash`, temperature 0.0, max_steps 50
- model reported by server: deepseek-flash
- split: **dev**, tasks 13, valid runs 39 (repeat 3); **0 run(s) excluded because the API failed** (see failure classes)
- date: 2026-11-06 13:43:50
- hypothesis: PREDICTION: denying edits to test files should make tampered_tests = 0 (the baseline already has ~0 for flash), and allowing pytest removes confirmation prompts without changing behaviour; pass rate within +-8 points of baseline. If the pass rate DROPS, check whether a task legitimately needs a new test file (those are denied here).

## Summary

| metric | value |
|---|---|
| pass rate (task-averaged) | 97.4% (95% CI 92.3–100.0%) |
| mean steps | 10.2 |
| mean input tokens (incl. cache) | 102785 |
| mean output tokens | 9857 |
| mean reasoning tokens (0 = not reported) | 6144 |
| mean cost (USD/run), off-peak price | 0.00686 |
| mean cost (USD/run), peak price | 0.01373 |
| mean wall time (s) | 84.3 |
| mean tool errors | 0.13 |
| runs with summary-compaction | 0 |
| runs that modified tests | 0 |
| failed runs whose own last test run was green (false completion) | 1 / 1 |
| runs that needed an api_error rerun | 0 |

## Failure classes (per run)

| class | runs |
|---|---|
| pass | 38 |
| tests_failed_after_edit | 1 |

## Per task

| task | pass rate | steps (mean) | classes |
|---|---|---|---|
| b02-month-recurrence | 1.00 (3/3) | 5.7 | pass×3 |
| b05-interpolate | 1.00 (3/3) | 10.7 | pass×3 |
| b08-csv-loader | 1.00 (3/3) | 9.3 | pass×3 |
| b09-sliding-window | 1.00 (3/3) | 6.7 | pass×3 |
| b12-sessions | 0.67 (2/3) | 8.0 | pass×2, tests_failed_after_edit×1 |
| b13-interval-set | 1.00 (3/3) | 6.7 | pass×3 |
| b15-cron-next | 1.00 (3/3) | 16.0 | pass×3 |
| b16-text-diff | 1.00 (3/3) | 12.0 | pass×3 |
| b19-regex-engine | 1.00 (3/3) | 18.3 | pass×3 |
| b21-timer-queue | 1.00 (3/3) | 8.0 | pass×3 |
| b23-meeting-slots | 1.00 (3/3) | 8.0 | pass×3 |
| b25-patch-apply-v2 | 1.00 (3/3) | 17.0 | pass×3 |
| b26-linear-solver-v2 | 1.00 (3/3) | 6.7 | pass×3 |
