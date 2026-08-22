"""
JSON-boundary sanitizer.

lstm/models/run_summary.json contains literal `Infinity`/`NaN` tokens (MAPE
on exact-zero-return days -- see lstm/README.md). Python's own json.load()
parses these into float('inf')/float('nan') without error, and Python's
json.dumps() will happily re-emit them too -- but `Infinity`/`NaN` are NOT
valid strict JSON, and a browser's JSON.parse() will reject them. This module
is the one place that boundary gets crossed safely: every artifact-serving
router must pass its loaded dict through sanitize_json() before returning it.

null is used as the sanitized value, documented in each affected response's
schema as: "null means the underlying metric's denominator was exactly zero
(e.g. MAPE on a zero daily_return) -- see lstm/models/run_summary.json's
known caveat," so a client can't mistake the substitution for "no data".
"""

from __future__ import annotations

import math
from typing import Any


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if (math.isinf(value) or math.isnan(value)) else value
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    return value
