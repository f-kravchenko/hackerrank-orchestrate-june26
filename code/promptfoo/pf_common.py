"""Shared helpers for the promptfoo integration.

Reconstructs a ``Claim`` (with loaded/preprocessed images) from promptfoo test vars so the
prompt builder and the scorer both operate on the exact same objects the real pipeline uses.
"""

from __future__ import annotations

import os
import sys

# Make the code/ package importable regardless of promptfoo's CWD.
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data import Claim, load_image  # noqa: E402

GOLD_FIELDS = [
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type",
    "object_part", "claim_status", "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]


def make_claim(vars: dict) -> Claim:
    """Build a Claim with images from promptfoo vars (which mirror the CSV columns)."""
    claim = Claim(
        user_id=vars["user_id"],
        image_paths=vars["image_paths"],
        user_claim=vars["user_claim"],
        claim_object=vars["claim_object"],
        gold={f: vars.get(f"gold_{f}", "") for f in GOLD_FIELDS},
    )
    claim.images = [load_image(p) for p in vars["image_paths"].split(";") if p.strip()]
    return claim
