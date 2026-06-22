"""Few-shot exemplars for the evidence-review prompt.

Text-only solved examples drawn from the labeled sample set. They teach output calibration
(the supported/contradicted/not_enough_information decision boundary and the severity scale)
without sending exemplar images, so the extra cost is small. History-derived risk flags are
stripped from exemplar outputs (the model is not responsible for those; the rule layer adds
them).

Leakage control: ``few_shot_messages(exclude_user_id=...)`` drops an exemplar if it is the
claim currently being predicted, so held-out evaluation on the sample set stays honest. The
test set (claims.csv) is disjoint from these sample exemplars, so the final run is unaffected.
"""

from __future__ import annotations

import csv
import json
import os

from data import DATASET_DIR

# Chosen to span the decision space: clear-supported, severity-exaggeration -> contradicted,
# wrong-object -> contradicted, not-enough-info, multi-image best-image pick, injection.
EXEMPLAR_IDS = ["user_001", "user_005", "user_033", "user_006", "user_003", "user_034"]

_HISTORY_FLAGS = {"user_history_risk", "manual_review_required"}


def _sample_rows() -> dict:
    path = os.path.join(DATASET_DIR, "sample_claims.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["user_id"]: r for r in csv.DictReader(fh)}


def _exemplar_output(row: dict) -> dict:
    flags = [f for f in row["risk_flags"].split(";") if f not in _HISTORY_FLAGS and f != "none"]
    sup = [] if row["supporting_image_ids"] == "none" else row["supporting_image_ids"].split(";")
    return {
        "evidence_standard_met": row["evidence_standard_met"] == "true",
        "evidence_standard_met_reason": row["evidence_standard_met_reason"],
        "risk_flags": flags,
        "issue_type": row["issue_type"],
        "object_part": row["object_part"],
        "claim_status": row["claim_status"],
        "claim_status_justification": row["claim_status_justification"],
        "supporting_image_ids": sup,
        "valid_image": row["valid_image"] == "true",
        "severity": row["severity"],
    }


def few_shot_messages(exclude_user_id: str | None = None) -> list[dict]:
    """Return alternating user/assistant exemplar turns (text-only)."""
    rows = _sample_rows()
    msgs: list[dict] = []
    for uid in EXEMPLAR_IDS:
        if uid == exclude_user_id or uid not in rows:
            continue
        r = rows[uid]
        image_ids = [os.path.splitext(os.path.basename(p))[0] for p in r["image_paths"].split(";")]
        user_txt = (
            "SOLVED EXAMPLE (text-only; learn the decision boundary and severity calibration).\n"
            f"CLAIM OBJECT: {r['claim_object']}\n"
            f"CONVERSATION:\n{r['user_claim']}\n"
            f"SUBMITTED IMAGES: {', '.join(image_ids)}\n"
            f"WHAT THE IMAGES SHOW: {r['evidence_standard_met_reason']} {r['claim_status_justification']}"
        )
        msgs.append({"role": "user", "content": user_txt})
        msgs.append({"role": "assistant", "content": json.dumps(_exemplar_output(r), ensure_ascii=False)})
    return msgs
