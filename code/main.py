"""Entry point: run the evidence-review system on dataset/claims.csv -> output.csv.

Usage:
    python code/main.py                     # full test set -> output.csv (repo root)
    python code/main.py --input dataset/sample_claims.csv --output sample_output.csv
    python code/main.py --limit 3 --no-cache
    python code/main.py --model deepseek-ai/DeepSeek-V4-Flash --workers 6

Requires DEEPINFRA_TOKEN (or DEEPINFRA_API_KEY) in the environment for live calls.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# Allow running as `python code/main.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import REPO_ROOT, load_claims  # noqa: E402
from pipeline import process_claims  # noqa: E402
from schema import OUTPUT_COLUMNS  # noqa: E402
from vlm_client import USAGE, VLMClient, DEFAULT_MODEL  # noqa: E402

try:
    from dotenv import load_dotenv  # optional convenience
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except Exception:  # noqa: BLE001
    pass


def write_output(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in OUTPUT_COLUMNS})


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-modal evidence review")
    ap.add_argument("--input", default=os.path.join(REPO_ROOT, "dataset", "claims.csv"))
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "output.csv"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N claims")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    claims = load_claims(args.input, with_images=True)
    if args.limit:
        claims = claims[: args.limit]

    print(f"Loaded {len(claims)} claims from {args.input}")
    client = VLMClient(model=args.model, use_cache=not args.no_cache)

    start = time.time()
    rows = process_claims(claims, client, max_workers=args.workers)
    elapsed = time.time() - start

    write_output(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output} in {elapsed:.1f}s")
    print(f"Usage: api_calls={USAGE['api_calls']} cache_hits={USAGE['cache_hits']} "
          f"prompt_tokens={USAGE['prompt_tokens']} completion_tokens={USAGE['completion_tokens']}")

    # Persist test-run stats so the evaluation report can quote actual (not extrapolated) usage.
    if os.path.basename(args.input) == "claims.csv":
        import json
        stats_path = os.path.join(os.path.dirname(__file__), "cache", "test_run_stats.json")
        # Keep wall-clock from the last LIVE run; a fully-cached re-run finishes in ~0s and
        # would otherwise overwrite the meaningful latency.
        live_elapsed = round(elapsed, 1)
        if USAGE["api_calls"] == 0 and os.path.isfile(stats_path):
            try:
                with open(stats_path, encoding="utf-8") as fh:
                    live_elapsed = json.load(fh).get("elapsed_seconds", live_elapsed)
            except Exception:  # noqa: BLE001
                pass
        stats = {
            "model": args.model, "n_claims": len(rows),
            "n_images": sum(c.count(";") + 1 for c in (r["image_paths"] for r in rows)),
            "prompt_tokens": USAGE["prompt_tokens"], "completion_tokens": USAGE["completion_tokens"],
            "api_calls": USAGE["api_calls"], "cache_hits": USAGE["cache_hits"],
            "elapsed_seconds": live_elapsed,
        }
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
