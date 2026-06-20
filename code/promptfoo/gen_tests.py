"""Generate tests.json for promptfoo from dataset/sample_claims.csv.

Each test carries the input columns as vars plus the gold labels as gold_* vars (read by
pf_assert.py). Run once before `promptfoo eval`:  python code/promptfoo/gen_tests.py
"""

from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SAMPLE = os.path.join(REPO_ROOT, "dataset", "sample_claims.csv")
OUT = os.path.join(HERE, "tests.json")

INPUT_COLS = ["user_id", "image_paths", "user_claim", "claim_object"]


def main() -> None:
    tests = []
    with open(SAMPLE, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vars = {c: row[c] for c in INPUT_COLS}
            for k, v in row.items():
                if k not in INPUT_COLS:
                    vars[f"gold_{k}"] = v
            tests.append({"vars": vars, "description": f"{row['user_id']} / {row['claim_object']}"})
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(tests, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(tests)} tests to {OUT}")


if __name__ == "__main__":
    main()
