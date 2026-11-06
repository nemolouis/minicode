# Eval report: 20261106-120253-feat-explore-flash

- config: `feat-explore-flash` (hash `51740c3248`), model `deepseek-flash`, temperature 0.0, max_steps 50
- model reported by server: deepseek-flash
- split: **dev**, tasks 13, valid runs 39 (repeat 3); **0 run(s) excluded because the API failed** (see failure classes)
- date: 2026-11-06 12:44:15
- hypothesis: PREDICTION: flash calls explore in <40% of runs. Pass rate within +-8 points of baseline. Parent input tokens per run fall 0-25% on tasks where it is used, but total tokens (parent + sub-agent) rise, because the sub-agent re-reads files; I expect cost per run to go UP by 10-40%, not down.

## Summary

| metric | value |
|---|---|
| pass rate (task-averaged) | 100.0% (95% CI 100.0–100.0%) |
| mean steps | 10.5 |
| mean input tokens (incl. cache) | 110342 |
| mean output tokens | 10051 |
| mean reasoning tokens (0 = not reported) | 6411 |
| mean cost (USD/run), off-peak price | 0.00699 |
| mean cost (USD/run), peak price | 0.01398 |
| mean wall time (s) | 115.1 |
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
| b02-month-recurrence | 1.00 (3/3) | 5.3 | pass×3 |
| b05-interpolate | 1.00 (3/3) | 9.0 | pass×3 |
| b08-csv-loader | 1.00 (3/3) | 11.7 | pass×3 |
| b09-sliding-window | 1.00 (3/3) | 10.7 | pass×3 |
| b12-sessions | 1.00 (3/3) | 7.0 | pass×3 |
| b13-interval-set | 1.00 (3/3) | 7.3 | pass×3 |
| b15-cron-next | 1.00 (3/3) | 14.3 | pass×3 |
| b16-text-diff | 1.00 (3/3) | 8.3 | pass×3 |
| b19-regex-engine | 1.00 (3/3) | 18.0 | pass×3 |
| b21-timer-queue | 1.00 (3/3) | 11.3 | pass×3 |
| b23-meeting-slots | 1.00 (3/3) | 6.7 | pass×3 |
| b25-patch-apply-v2 | 1.00 (3/3) | 21.0 | pass×3 |
| b26-linear-solver-v2 | 1.00 (3/3) | 6.0 | pass×3 |
