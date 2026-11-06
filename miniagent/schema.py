"""A small JSON-schema checker for tool arguments.

Supports the subset tool schemas actually use: `type` (a name or a list), `required`, `properties`, `enum`, `items`.
Unknown keywords (anyOf, pattern, ...) and unknown properties are ignored on purpose: the goal is to catch the
mistakes weak models make (missing field, string where an integer is expected) with a message they can act on,
not to reject every deviation. Returns a list of error strings; empty means valid.
"""
from __future__ import annotations

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _short(v) -> str:
    s = repr(v)
    return s if len(s) <= 40 else s[:37] + "..."


def validate(value, schema: dict, path: str = "") -> list[str]:
    errs: list[str] = []
    where = path or "arguments"
    t = schema.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        checks = [_TYPES[n] for n in names if n in _TYPES]
        if checks and not any(c(value) for c in checks):
            return [f"{where}: expected {'/'.join(names)}, got {type(value).__name__} {_short(value)}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{where}: must be one of {schema['enum']}, got {_short(value)}")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path + '.' if path else ''}{k}: required")
        for k, sub in (schema.get("properties") or {}).items():
            if k in value and isinstance(sub, dict):
                errs += validate(value[k], sub, f"{path + '.' if path else ''}{k}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errs += validate(item, schema["items"], f"{where}[{i}]")
    return errs


def signature(schema: dict) -> str:
    """One-line description of the expected arguments, e.g. `path: string, offset?: integer`."""
    req = set(schema.get("required", []))
    parts = []
    for k, sub in (schema.get("properties") or {}).items():
        t = sub.get("type", "any") if isinstance(sub, dict) else "any"
        t = "/".join(t) if isinstance(t, list) else t
        parts.append(f"{k}{'' if k in req else '?'}: {t}")
    return "{" + ", ".join(parts) + "}"
