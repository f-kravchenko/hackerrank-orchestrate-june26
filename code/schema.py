"""Canonical output schema, allowed enum values, and the JSON schema sent to the model.

Single source of truth for the 14 output columns and every allowed categorical value
(mirrors ``problem_statement.md`` -> "Allowed values"). The rule layer (``rules.py``)
clamps model output to these sets so ``output.csv`` never contains an out-of-spec value.
"""

# Columns required in output.csv, in the exact required order (4 input + 10 derived).
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_OBJECTS = ["car", "laptop", "package"]

CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

SEVERITY = ["none", "low", "medium", "high", "unknown"]

# object_part allowed values are object-specific.
OBJECT_PART = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
        "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label", "contents", "item",
        "unknown",
    ],
}

# Every allowed object_part across objects (used for loose validation / fallback).
ALL_OBJECT_PARTS = sorted({p for parts in OBJECT_PART.values() for p in parts})

RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]

# Flags the deterministic rule layer owns (derived from user_history.csv), not the model.
HISTORY_RISK_FLAGS = ["user_history_risk", "manual_review_required"]


def model_json_schema(claim_object: str) -> dict:
    """JSON schema for the model's structured output for a single claim.

    ``object_part`` is constrained to the per-object allowed list. History-derived flags
    are intentionally excluded from the model's risk_flags enum; the rule layer adds them.
    """
    visual_risk_flags = [f for f in RISK_FLAGS if f not in HISTORY_RISK_FLAGS]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_standard_met": {"type": "boolean"},
            "evidence_standard_met_reason": {"type": "string"},
            "risk_flags": {
                "type": "array",
                "items": {"type": "string", "enum": visual_risk_flags},
            },
            "issue_type": {"type": "string", "enum": ISSUE_TYPE},
            "object_part": {"type": "string", "enum": OBJECT_PART[claim_object]},
            "claim_status": {"type": "string", "enum": CLAIM_STATUS},
            "claim_status_justification": {"type": "string"},
            "supporting_image_ids": {"type": "array", "items": {"type": "string"}},
            "valid_image": {"type": "boolean"},
            "severity": {"type": "string", "enum": SEVERITY},
        },
        "required": [
            "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
            "issue_type", "object_part", "claim_status", "claim_status_justification",
            "supporting_image_ids", "valid_image", "severity",
        ],
    }
