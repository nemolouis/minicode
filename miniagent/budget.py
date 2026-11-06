"""Hard limits on what one run may spend. Checked before every model request, so a run overshoots by at most one step. Sub-agent
tokens count. A run that hits a limit ends with status "budget" and keeps its trajectory (the 402 that used to end evaluation batches
after the account ran dry becomes a clean, recorded stop instead)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    max_cost_usd: float | None = None
    max_total_tokens: int | None = None            # input + output tokens over all model calls of the run
    price: dict | None = None                      # {"input", "cache_read", "output"} USD per 1M tokens (one tier of configs/prices.json)

    def __post_init__(self):
        if self.max_cost_usd is not None:
            if self.max_cost_usd <= 0:
                raise ValueError("max_cost_usd must be > 0")
            if not self.price or any(self.price.get(k) is None for k in ("input", "cache_read", "output")):
                raise ValueError("max_cost_usd needs a complete price (input, cache_read, output) for the model")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be > 0")

    def cost(self, uncached_in: int, cached_in: int, out: int) -> float | None:
        if not self.price or any(self.price.get(k) is None for k in ("input", "cache_read", "output")):
            return None
        return (uncached_in * self.price["input"] + cached_in * self.price["cache_read"] + out * self.price["output"]) / 1e6

    def exceeded(self, tokens: int, cost: float | None) -> str | None:
        """The reason a limit is hit, or None."""
        if self.max_total_tokens is not None and tokens >= self.max_total_tokens:
            return f"token limit {self.max_total_tokens} reached ({tokens} used)"
        if self.max_cost_usd is not None and cost is not None and cost >= self.max_cost_usd:
            return f"cost limit ${self.max_cost_usd:g} reached (${cost:.4f} spent)"
        return None
