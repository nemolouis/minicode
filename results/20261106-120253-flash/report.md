# Eval report: 20261106-120253-flash

- config: `flash` (hash `818f1f0e69`), model `deepseek-flash`, temperature 0.0, max_steps 50
- model reported by server: deepseek-flash
- split: **dev**, tasks 13, valid runs 39 (repeat 3); **0 run(s) excluded because the API failed** (see failure classes)
- date: 2026-11-06 12:02:53
- hypothesis: model comparison with explicit model id; max_tokens 16384 (4096 truncated thinking), max_steps 50 (30 exhausted by slower explorers)

## Summary

| metric | value |
|---|---|
| pass rate (task-averaged) | 89.7% (95% CI 82.1–97.4%) |
| mean steps | 8.6 |
| mean input tokens (incl. cache) | 59726 |
| mean output tokens | 8214 |
| mean reasoning tokens (0 = not reported) | 5031 |
| mean cost (USD/run), off-peak price | 0.00564 |
| mean cost (USD/run), peak price | 0.01129 |
| mean wall time (s) | 103.6 |
| mean tool errors | 0.08 |
| runs with summary-compaction | 0 |
| runs that modified tests | 0 |
| failed runs whose own last test run was green (false completion) | 4 / 4 |
| runs that needed an api_error rerun | 0 |

## Failure classes (per run)

| class | runs |
|---|---|
| pass | 35 |
| tests_failed_after_edit | 3 |
| max_tokens | 1 |

## Per task

| task | pass rate | steps (mean) | classes |
|---|---|---|---|
| b02-month-recurrence | 0.67 (2/3) | 5.3 | pass×2, tests_failed_after_edit×1 |
| b05-interpolate | 1.00 (3/3) | 6.7 | pass×3 |
| b08-csv-loader | 1.00 (3/3) | 10.0 | pass×3 |
| b09-sliding-window | 1.00 (3/3) | 6.7 | pass×3 |
| b12-sessions | 1.00 (3/3) | 8.7 | pass×3 |
| b13-interval-set | 1.00 (3/3) | 7.7 | pass×3 |
| b15-cron-next | 0.67 (2/3) | 10.0 | pass×2, tests_failed_after_edit×1 |
| b16-text-diff | 0.67 (2/3) | 8.0 | pass×2, max_tokens×1 |
| b19-regex-engine | 1.00 (3/3) | 10.7 | pass×3 |
| b21-timer-queue | 1.00 (3/3) | 9.7 | pass×3 |
| b23-meeting-slots | 1.00 (3/3) | 8.0 | pass×3 |
| b25-patch-apply-v2 | 0.67 (2/3) | 13.7 | pass×2, tests_failed_after_edit×1 |
| b26-linear-solver-v2 | 1.00 (3/3) | 7.0 | pass×3 |
