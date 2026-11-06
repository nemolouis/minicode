"""Small, dependency-free statistics for comparing evaluation runs.

Why not only the bootstrap over tasks that `report.py` has always used: with 13 tasks it ignores how noisy one task is from run to
run, and (the reason this module exists) it called a +10 point difference "significant" (CI 2.6 to 17.9) when two arms in which the
feature was hardly used differed by that much through run-to-run noise alone. Two tests are reported next to it:

* a paired sign-flip test over tasks (exact for up to `EXACT_LIMIT` tasks with a difference, Monte Carlo above): under "no effect" the
  sign of each task's difference in pass rate is a coin flip. The statistical unit is the task, as it should be.
* Fisher's exact test over pooled runs: treats runs as independent, which they are not (runs of one task share its difficulty), so it is
  anti-conservative; it is shown because it is the number people usually quote, and as a cross-check.

The verdict needs both to agree: "inconclusive" is the default, "different" requires p < alpha in the paired test AND the pooled test.
"""
from __future__ import annotations

import math
import random
from itertools import product

EXACT_LIMIT = 16
ALPHA = 0.05


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p of Fisher's exact test for the table [[a, b], [c, d]] (sum of all tables no more likely than the observed one)."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0:
        return 1.0

    def p(x: int) -> float:
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    observed = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= observed * (1 + 1e-9)))


def sign_flip_test(diffs: list[float], *, seed: int = 0, samples: int = 20000) -> float:
    """Two-sided p that the mean of `diffs` is this far from 0 if each difference's sign were a fair coin. Zeros carry no information
    and are dropped. Exact when at most EXACT_LIMIT differences are non-zero, Monte Carlo otherwise."""
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    observed = abs(sum(d))
    eps = 1e-12
    if len(d) <= EXACT_LIMIT:
        hits = sum(1 for signs in product((1, -1), repeat=len(d)) if abs(sum(s * x for s, x in zip(signs, d))) >= observed - eps)
        return hits / 2 ** len(d)
    rng = random.Random(seed)
    hits = sum(1 for _ in range(samples) if abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= observed - eps)
    return (hits + 1) / (samples + 1)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; sensible at 0/n and n/n, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def verdict(p_paired: float, p_pooled: float, alpha: float = ALPHA) -> str:
    if p_paired < alpha and p_pooled < alpha:
        return "different"
    if p_paired < alpha or p_pooled < alpha:
        return "inconclusive (the two tests disagree)"
    return "inconclusive (no difference beyond noise)"


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, sd
