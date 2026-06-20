"""Deterministic post-processing of the model's structured output.

Responsibilities:
1. Merge history-derived risk flags from user_history.csv (NOT trusted to the model).
2. Clamp every categorical field to its allowed enum (defensive, even with json_schema).
3. Enforce internal consistency observed in the labeled sample data.
4. Constrain supporting_image_ids to the actually-submitted image IDs.

History-flag rule (derived from sample_claims.csv):
- history_flags contains `user_history_risk`  -> output gets user_history_risk AND
  manual_review_required.
- history_flags contains `manual_review_required` -> output gets manual_review_required.
"""

from __future__ import annotations

from schema import (
    ISSUE_TYPE, OBJECT_PART, CLAIM_STATUS, SEVERITY, RISK_FLAGS, ALL_OBJECT_PARTS,
)


def _clamp(value, allowed, default):
    return value if value in allowed else default


def _order_flags(flags: set[str]) -> str:
    ordered = [f for f in RISK_FLAGS if f in flags and f != "none"]
    return ";".join(ordered) if ordered else "none"


def _quality_gate(flags: set[str], images) -> set[str]:
    """Drop model-emitted quality flags that the deterministic CV metrics do not corroborate.

    The model over-emits `low_light_or_glare` (and sometimes `blurry_image`) on clean images.
    Only keep them when at least one submitted image actually looks dark/glary/low-detail.
    This raises precision and never adds a flag.
    """
    metr = [im for im in images if im.exists and im.quality_ok]
    if not metr:
        return flags  # no metrics -> don't touch anything
    min_sharp = min(im.sharpness for im in metr)
    min_bright = min(im.brightness for im in metr)
    max_glare = max(im.glare for im in metr)
    max_dark = max(im.dark for im in metr)

    out = set(flags)
    if "low_light_or_glare" in out and not (min_bright < 60 or max_glare > 0.20 or max_dark > 0.40):
        out.discard("low_light_or_glare")
    if "blurry_image" in out and not (min_sharp < 130):
        out.discard("blurry_image")
    return out


def apply_rules(pred: dict, claim, history: dict | None,
                *, merge_history: bool = True, enforce_consistency: bool = True,
                quality_gate: bool = True) -> dict:
    """Return a normalized output dict (model fields + consistency + history flags).

    Enum clamping and supporting-id validation always run (they keep output in-spec).
    ``merge_history``, ``enforce_consistency``, and ``quality_gate`` can be disabled for
    ablation studies.
    """
    obj = claim.claim_object
    valid_ids = {img.image_id for img in claim.images}

    # --- categorical clamping -------------------------------------------------
    issue_type = _clamp(pred.get("issue_type"), ISSUE_TYPE, "unknown")
    part_allowed = OBJECT_PART.get(obj, ALL_OBJECT_PARTS)
    object_part = _clamp(pred.get("object_part"), part_allowed, "unknown")
    claim_status = _clamp(pred.get("claim_status"), CLAIM_STATUS, "not_enough_information")
    severity = _clamp(pred.get("severity"), SEVERITY, "unknown")
    evidence_met = bool(pred.get("evidence_standard_met", False))
    valid_image = bool(pred.get("valid_image", False))

    # --- risk flags: visual (from model) + history (deterministic) ------------
    flags: set[str] = set()
    for f in pred.get("risk_flags", []) or []:
        if f in RISK_FLAGS and f != "none":
            flags.add(f)

    # Deterministic quality gate on the model's visual flags (before history merge).
    if quality_gate:
        flags = _quality_gate(flags, claim.images)

    if merge_history:
        hist_flags = set((history or {}).get("history_flags", "").split(";"))
        if "user_history_risk" in hist_flags:
            flags.add("user_history_risk")
            flags.add("manual_review_required")
        if "manual_review_required" in hist_flags:
            flags.add("manual_review_required")

    # --- supporting image ids: must be real, submitted IDs --------------------
    sup_raw = pred.get("supporting_image_ids", []) or []
    supporting = [i for i in sup_raw if i in valid_ids]

    # --- consistency rules (mirror labeled sample behavior) -------------------
    # In the labeled data, evidence_standard_met == false  <=>  claim_status == NEI.
    # claim_status is the more reliable model signal, so derive evidence_met from it
    # rather than forcing status from a (less reliable) sufficiency flag.
    if enforce_consistency:
        if claim_status == "not_enough_information":
            evidence_met = False
            supporting = []
            issue_type = "unknown"
            severity = "unknown"
        else:
            evidence_met = True
            if issue_type == "none":  # part visible, no issue -> no severity
                severity = "none"

    supporting_str = ";".join(supporting) if supporting else "none"

    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": obj,
        "evidence_standard_met": "true" if evidence_met else "false",
        "evidence_standard_met_reason": (pred.get("evidence_standard_met_reason") or "").strip(),
        "risk_flags": _order_flags(flags),
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": (pred.get("claim_status_justification") or "").strip(),
        "supporting_image_ids": supporting_str,
        "valid_image": "true" if valid_image else "false",
        "severity": severity,
    }
