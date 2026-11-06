# Eval report: 20261106-120253-feat-diag-flash

- config: `feat-diag-flash` (hash `95386dbf47`), model `deepseek-flash`, temperature 0.0, max_steps 50
- model reported by server: deepseek-flash
- split: **dev**, tasks 13, valid runs 39 (repeat 3); **0 run(s) excluded because the API failed** (see failure classes)
- date: 2026-11-06 13:30:15
- hypothesis: PREDICTION: the B-layer tasks are logic bugs, so a post-edit syntax warning appears in <10% of runs; pass rate within +-8 points of baseline. Expected effect on flash: none.

## Summary

| metric | value |
|---|---|
| pass rate (task-averaged) | 92.3% (95% CI 84.6–100.0%) |
| mean steps | 8.6 |
| mean input tokens (incl. cache) | 59500 |
| mean output tokens | 8443 |
| mean reasoning tokens (0 = not reported) | 5410 |
| mean cost (USD/run), off-peak price | 0.00572 |
| mean cost (USD/run), peak price | 0.01143 |
| mean wall time (s) | 77.4 |
| mean tool errors | 0.05 |
| runs with summary-compaction | 0 |
| runs that modified tests | 0 |
| failed runs whose own last test run was green (false completion) | 2 / 3 |
| runs that needed an api_error rerun | 0 |

## Failure classes (per run)

| class | runs |
|---|---|
| pass | 36 |
| tests_failed_after_edit | 2 |
| max_tokens | 1 |

## Per task

| task | pass rate | steps (mean) | classes |
|---|---|---|---|
| b02-month-recurrence | 1.00 (3/3) | 6.0 | pass×3 |
| b05-interpolate | 1.00 (3/3) | 9.0 | pass×3 |
| b08-csv-loader | 1.00 (3/3) | 9.3 | pass×3 |
| b09-sliding-window | 1.00 (3/3) | 6.3 | pass×3 |
| b12-sessions | 0.67 (2/3) | 6.3 | pass×2, tests_failed_after_edit×1 |
| b13-interval-set | 0.67 (2/3) | 6.0 | tests_failed_after_edit×1, pass×2 |
| b15-cron-next | 1.00 (3/3) | 9.7 | pass×3 |
| b16-text-diff | 0.67 (2/3) | 6.0 | pass×2, max_tokens×1 |
| b19-regex-engine | 1.00 (3/3) | 12.0 | pass×3 |
| b21-timer-queue | 1.00 (3/3) | 10.7 | pass×3 |
| b23-meeting-slots | 1.00 (3/3) | 7.7 | pass×3 |
| b25-patch-apply-v2 | 1.00 (3/3) | 14.0 | pass×3 |
| b26-linear-solver-v2 | 1.00 (3/3) | 9.0 | pass×3 |
