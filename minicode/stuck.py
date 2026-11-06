"""Detecting an agent that is going in circles.

Measured on the fine-tuned student: 54% of its tool calls were identical to the previous one (the same pytest run, over and over).
A repeat is only suspicious when *nothing changed*: re-running the tests after an edit is normal. So the detector keys on the call
together with its result:

  repeat  the same tool call with the same result N times in a row (N = warn_after: a notice is appended to the tool result;
          N = stop_after: the run ends with status "stuck")
  errors  consecutive tool errors of the same kind (same tool, same first line of the message), same thresholds

Both counters reset as soon as something different happens. Disabled unless warn_after > 0.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass
class StuckEvent:
    kind: str        # "repeat" | "errors"
    count: int
    action: str      # "warn" | "stop"


def _fingerprint(call: dict, result: dict) -> tuple:
    inp = json.dumps(call["input"], sort_keys=True, ensure_ascii=False, default=str)
    return (call["name"], inp, hashlib.sha1(str(result["output"]).encode()).hexdigest())


def _error_kind(call: dict, result: dict) -> tuple | None:
    if not result["is_error"]:
        return None
    return (call["name"], str(result["output"]).strip().splitlines()[0][:80] if str(result["output"]).strip() else "")


class StuckDetector:
    def __init__(self, warn_after: int = 3, stop_after: int = 6):
        if warn_after < 2 or stop_after <= warn_after:
            raise ValueError("need 2 <= warn_after < stop_after")
        self.warn_after, self.stop_after = warn_after, stop_after
        self._repeat_key: tuple | None = None
        self._repeat_n = 0
        self._error_key: tuple | None = None
        self._error_n = 0

    def observe(self, call: dict, result: dict) -> StuckEvent | None:
        """Feed one finished call (in the order they ran). Returns the most serious event it triggers, if any."""
        fp = _fingerprint(call, result)
        self._repeat_n = self._repeat_n + 1 if fp == self._repeat_key else 1
        self._repeat_key = fp
        ek = _error_kind(call, result)
        self._error_n = self._error_n + 1 if (ek is not None and ek == self._error_key) else (1 if ek is not None else 0)
        self._error_key = ek
        events = [StuckEvent(kind, n, "stop" if n >= self.stop_after else "warn")
                  for kind, n in (("repeat", self._repeat_n), ("errors", self._error_n)) if n >= self.warn_after]
        return max(events, key=lambda e: (e.action == "stop", e.count), default=None)


def notice(ev: StuckEvent) -> str:
    if ev.action == "stop":
        return f"\n\n[notice] The same {'call with the same result' if ev.kind == 'repeat' else 'error'} {ev.count} times in a row: stopping the run."
    if ev.kind == "repeat":
        return (f"\n\n[notice] This is the same call with the same result {ev.count} times in a row. Repeating it will not change "
                "anything: change your approach (read the relevant code, make an edit, or finish with what you know).")
    return (f"\n\n[notice] {ev.count} consecutive errors of the same kind. Re-read the error message and the file before trying again; "
            "do not repeat the call unchanged.")
