# Eval report: 20261106-120253-feat-todo-flash

- config: `feat-todo-flash` (hash `dbae570528`), model `deepseek-flash`, temperature 0.0, max_steps 50
- model reported by server: deepseek-flash
- split: **dev**, tasks 13, valid runs 39 (repeat 3); **0 run(s) excluded because the API failed** (see failure classes)
- date: 2026-11-06 12:21:21
- hypothesis: PREDICTION (written before running; baseline = flash, B-layer dev 13 tasks, ~90%): todo_write is used on <50% of runs and the pass rate stays within +-8 points of baseline (flash is near the ceiling, so no effect is detectable at 13 tasks x 3). Mean steps rise by 0-2 (the todo calls themselves).

## Summary

| metric | value |
|---|---|
| pass rate (task-averaged) | 100.0% (95% CI 100.0–100.0%) |
| mean steps | 11.7 |
| mean input tokens (incl. cache) | 126181 |
| mean output tokens | 10479 |
| mean reasoning tokens (0 = not reported) | 6574 |
| mean cost (USD/run), off-peak price | 0.00732 |
| mean cost (USD/run), peak price | 0.01464 |
| mean wall time (s) | 125.8 |
| mean tool errors | 0.05 |
| runs with summary-compaction | 0 |
| runs that modified tests | 0 |
| failed runs whose own last test run was green (false completion) | 0 / 0 |
| runs that needed an api_error rerun | 0 |

## Failure classes (per run)

| class | runs |
|---|---|
| pass | 39 |

## Per task

| task | pass rate | steps (mean) | classes |
|---|---|---|---|
| b02-month-recurrence | 1.00 (3/3) | 6.0 | pass×3 |
| b05-interpolate | 1.00 (3/3) | 8.7 | pass×3 |
| b08-csv-loader | 1.00 (3/3) | 17.0 | pass×3 |
| b09-sliding-window | 1.00 (3/3) | 7.0 | pass×3 |
| b12-sessions | 1.00 (3/3) | 7.0 | pass×3 |
| b13-interval-set | 1.00 (3/3) | 7.7 | pass×3 |
| b15-cron-next | 1.00 (3/3) | 12.0 | pass×3 |
| b16-text-diff | 1.00 (3/3) | 16.0 | pass×3 |
| b19-regex-engine | 1.00 (3/3) | 22.7 | pass×3 |
| b21-timer-queue | 1.00 (3/3) | 14.3 | pass×3 |
| b23-meeting-slots | 1.00 (3/3) | 5.3 | pass×3 |
| b25-patch-apply-v2 | 1.00 (3/3) | 21.0 | pass×3 |
| b26-linear-solver-v2 | 1.00 (3/3) | 8.0 | pass×3 |
