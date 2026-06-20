"""Custom promptfoo assertion: score model output against gold THROUGH the rule layer.

promptfoo sends each prompt variant to the model; this scorer parses the raw JSON, applies
the same deterministic rule layer the production pipeline uses (history-flag merge, enum
clamping, consistency), then computes a composite metric against the gold labels carried in
the test vars. The returned ``score`` (0..1) is what promptfoo averages per prompt variant to
produce the leaderboard; ``componentResults`` expose the per-field breakdown in the web UI.
"""

from __future__ import annotations

import json
import re

from pf_common import make_claim
from data import load_user_history
from rules import apply_rules
from evaluation.metrics import SCALAR_FIELDS, SET_FIELDS, set_prf

_DEFAULT_STRATEGY = {"merge_history": True, "enforce_consistency": True}


def _parse_json(output: str) -> dict:
    if isinstance(output, dict):
        return output
    try:
        return json.loads(output)
    except Exception:  # noqa: BLE001 - try to salvage a JSON object embedded in text
        m = re.search(r"\{.*\}", output, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {}


def get_assert(output, context):
    vars = context["vars"]
    claim = make_claim(vars)
    gold = claim.gold

    pred_raw = _parse_json(output)
    history = load_user_history().get(claim.user_id)
    pred = apply_rules(pred_raw, claim, history, **_DEFAULT_STRATEGY)

    components = []
    total = 0.0
    n = 0

    for f in SCALAR_FIELDS:
        ok = str(pred.get(f, "")).strip().lower() == str(gold.get(f, "")).strip().lower()
        components.append({"pass": ok, "score": 1.0 if ok else 0.0,
                           "reason": f"{f}: pred={pred.get(f)!r} gold={gold.get(f)!r}"})
        total += 1.0 if ok else 0.0
        n += 1

    for f in SET_FIELDS:
        _, _, f1 = set_prf(gold.get(f, ""), pred.get(f, ""))
        components.append({"pass": f1 >= 0.5, "score": f1,
                           "reason": f"{f} F1={f1:.2f}: pred={pred.get(f)!r} gold={gold.get(f)!r}"})
        total += f1
        n += 1

    score = total / n if n else 0.0
    return {
        "pass": score >= 0.6,
        "score": score,
        "reason": f"composite={score:.3f} over {n} fields (rule layer applied)",
        "componentResults": components,
    }
