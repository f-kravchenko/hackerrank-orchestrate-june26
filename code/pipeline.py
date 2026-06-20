"""Per-claim orchestration: load context -> VLM call -> deterministic rule layer.

Exposes ``process_claims`` which runs a list of claims (optionally concurrently) and
returns normalized output rows ready to write to CSV.

A ``strategy`` dict controls ablations used by the evaluation harness:
    include_requirements : inject minimum-evidence requirements into the prompt (default True)
    merge_history        : merge history-derived risk flags in the rule layer (default True)
    enforce_consistency  : apply consistency rules in the rule layer (default True)
    verify               : run the adversarial second-opinion pass on supported claims (default False)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from data import load_user_history, requirements_for
from prompts import build_messages
from rules import apply_rules
from schema import model_json_schema
from verify import apply_verification
from vlm_client import VLMClient

DEFAULT_STRATEGY = {
    "include_requirements": True,
    "merge_history": True,
    "enforce_consistency": True,
    "verify": False,
}


def process_claim(claim, client: VLMClient, history_by_user: dict, strategy: dict | None = None) -> dict:
    strategy = {**DEFAULT_STRATEGY, **(strategy or {})}
    requirements = requirements_for(claim.claim_object) if strategy["include_requirements"] else []
    messages = build_messages(claim, requirements)
    schema = model_json_schema(claim.claim_object)
    pred = client.complete(messages, schema)
    if strategy["verify"]:
        pred = apply_verification(pred, claim, client)
    history = history_by_user.get(claim.user_id)
    return apply_rules(
        pred, claim, history,
        merge_history=strategy["merge_history"],
        enforce_consistency=strategy["enforce_consistency"],
    )


def process_claims(claims, client: VLMClient, *, max_workers: int = 6,
                   strategy: dict | None = None) -> list[dict]:
    """Process claims concurrently, preserving input order in the output."""
    history_by_user = load_user_history()
    results: list[dict | None] = [None] * len(claims)

    if max_workers <= 1:
        for i, claim in enumerate(claims):
            results[i] = process_claim(claim, client, history_by_user, strategy)
        return results  # type: ignore[return-value]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_claim, c, client, history_by_user, strategy): i
                   for i, c in enumerate(claims)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return results  # type: ignore[return-value]
