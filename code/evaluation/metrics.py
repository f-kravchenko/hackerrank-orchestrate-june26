"""Scoring utilities for evaluating predictions against gold labels.

- Exact-match accuracy for scalar categorical fields.
- Set-based precision/recall/F1 for semicolon-separated multi-value fields
  (risk_flags, supporting_image_ids).
"""

from __future__ import annotations

SCALAR_FIELDS = [
    "evidence_standard_met", "valid_image", "issue_type", "object_part",
    "claim_status", "severity",
]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]


def _as_set(value: str) -> set[str]:
    if value is None:
        return set()
    items = {v.strip() for v in str(value).split(";") if v.strip()}
    # Treat "none" as the empty set so F1 isn't gamed by matching the literal token.
    items.discard("none")
    return items


def set_prf(gold: str, pred: str) -> tuple[float, float, float]:
    g, p = _as_set(gold), _as_set(pred)
    if not g and not p:
        return 1.0, 1.0, 1.0  # both empty -> perfect agreement
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def score(preds: list[dict], golds: list[dict]) -> dict:
    """Compute per-field metrics over aligned prediction/gold rows."""
    n = len(golds)
    assert len(preds) == n, "preds/golds length mismatch"

    scalar_correct = {f: 0 for f in SCALAR_FIELDS}
    set_f1_sum = {f: 0.0 for f in SET_FIELDS}
    set_p_sum = {f: 0.0 for f in SET_FIELDS}
    set_r_sum = {f: 0.0 for f in SET_FIELDS}

    for pred, gold in zip(preds, golds):
        for f in SCALAR_FIELDS:
            if str(pred.get(f, "")).strip().lower() == str(gold.get(f, "")).strip().lower():
                scalar_correct[f] += 1
        for f in SET_FIELDS:
            p, r, f1 = set_prf(gold.get(f, ""), pred.get(f, ""))
            set_p_sum[f] += p
            set_r_sum[f] += r
            set_f1_sum[f] += f1

    result = {
        "n": n,
        "scalar_accuracy": {f: scalar_correct[f] / n for f in SCALAR_FIELDS},
        "set_metrics": {
            f: {"precision": set_p_sum[f] / n, "recall": set_r_sum[f] / n,
                "f1": set_f1_sum[f] / n}
            for f in SET_FIELDS
        },
    }
    result["macro_scalar_accuracy"] = sum(result["scalar_accuracy"].values()) / len(SCALAR_FIELDS)
    return result


def confusion(preds: list[dict], golds: list[dict], field: str) -> dict:
    """Nested dict gold->pred->count for a categorical field."""
    table: dict[str, dict[str, int]] = {}
    for pred, gold in zip(preds, golds):
        g = str(gold.get(field, ""))
        p = str(pred.get(field, ""))
        table.setdefault(g, {}).setdefault(p, 0)
        table[g][p] += 1
    return table


def format_report(name: str, result: dict) -> str:
    lines = [f"### {name} (n={result['n']})", ""]
    lines.append(f"**Macro scalar accuracy:** {result['macro_scalar_accuracy']:.3f}")
    lines.append("")
    lines.append("| Field | Metric | Score |")
    lines.append("|---|---|---|")
    for f, acc in result["scalar_accuracy"].items():
        lines.append(f"| {f} | accuracy | {acc:.3f} |")
    for f, m in result["set_metrics"].items():
        lines.append(f"| {f} | precision | {m['precision']:.3f} |")
        lines.append(f"| {f} | recall | {m['recall']:.3f} |")
        lines.append(f"| {f} | F1 | {m['f1']:.3f} |")
    return "\n".join(lines)
