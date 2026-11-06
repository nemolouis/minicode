# Eval report: 20261106-120253-feat-readfirst-flash

- config: `feat-readfirst-flash` (hash `fa73012efc`), model `deepseek-flash`, temperature 0.0, max_steps 50
- model reported by server: deepseek-flash
- split: **dev**, tasks 13, valid runs 39 (repeat 3); **0 run(s) excluded because the API failed** (see failure classes)
- date: 2026-11-06 13:06:58
- hypothesis: PREDICTION (baseline = flash, B-layer dev 13 tasks x 3, ~90%): flash reads a file before editing it in nearly every run, so require_read fires in <10% of runs and the pass rate stays within +-8 points of baseline; mean steps change by <1. The guard is aimed at weak models (their edits copy text that was never shown: 76/94 for the fine-tuned student) and cannot be evaluated on flash.

## Summary

| metric | value |
|---|---|
| pass rate (task-averaged) | 87.2% (95% CI 71.8–100.0%) |
| mean steps | 9.7 |
| mean input tokens (incl. cache) | 107423 |
| mean output tokens | 9961 |
| mean reasoning tokens (0 = not reported) | 6320 |
| mean cost (USD/run), off-peak price | 0.00708 |
| mean cost (USD/run), peak price | 0.01417 |
| mean wall time (s) | 113.0 |
| mean tool errors | 0.18 |
| runs with summary-compaction | 0 |
| runs that modified tests | 0 |
| failed runs whose own last test run was green (false completion) | 4 / 5 |
| runs that needed an api_error rerun | 0 |

## Failure classes (per run)

| class | runs |
|---|---|
| pass | 34 |
| tests_failed_after_edit | 4 |
| max_tokens | 1 |

## Per task

| task | pass rate | steps (mean) | classes |
|---|---|---|---|
| b02-month-recurrence | 1.00 (3/3) | 5.3 | pass×3 |
| b05-interpolate | 1.00 (3/3) | 7.7 | pass×3 |
| b08-csv-loader | 1.00 (3/3) | 11.7 | pass×3 |
| b09-sliding-window | 1.00 (3/3) | 8.0 | pass×3 |
| b12-sessions | 1.00 (3/3) | 7.7 | pass×3 |
| b13-interval-set | 0.33 (1/3) | 6.7 | tests_failed_after_edit×2, pass×1 |
| b15-cron-next | 0.33 (1/3) | 9.0 | tests_failed_after_edit×2, pass×1 |
| b16-text-diff | 0.67 (2/3) | 7.3 | pass×2, max_tokens×1 |
| b19-regex-engine | 1.00 (3/3) | 27.0 | pass×3 |
| b21-timer-queue | 1.00 (3/3) | 7.7 | pass×3 |
| b23-meeting-slots | 1.00 (3/3) | 7.3 | pass×3 |
| b25-patch-apply-v2 | 1.00 (3/3) | 15.3 | pass×3 |
| b26-linear-solver-v2 | 1.00 (3/3) | 5.7 | pass×3 |
