# Prompt comparison with promptfoo

Compares evidence-review **prompt variants** on the labeled sample set, scoring each variant
against gold **through the deterministic rule layer** (so the numbers reflect the real
pipeline, not raw model output).

## Files
- `promptfooconfig.yaml` — prompts (variants), provider (DeepInfra), tests, assertion.
- `pf_prompts.py` — the prompt variants being compared (`prompt_full`, `prompt_no_rubric`, `prompt_terse`).
- `pf_assert.py` — custom Python scorer: parse JSON → apply rule layer → composite metric vs gold.
- `pf_common.py` — rebuilds a `Claim` (with preprocessed images) from test vars.
- `gen_tests.py` — writes `tests.json` from `dataset/sample_claims.csv`.

## Run
```bash
# from repo root
export DEEPINFRA_TOKEN=...                          # or: set -a; . code/.env; set +a
export PROMPTFOO_PYTHON="$(pwd)/.venv/bin/python"   # venv with openai + Pillow
python code/promptfoo/gen_tests.py                  # -> tests.json (20 rows)

cd code/promptfoo
promptfoo eval -o results.json                      # 3 prompts x 20 tests
promptfoo view                                      # web UI: side-by-side + per-field drilldown
```

## Scoring
Each test's score is the mean of 8 components: exact-match on the 6 scalar fields
(`evidence_standard_met`, `valid_image`, `issue_type`, `object_part`, `claim_status`,
`severity`) + set-F1 on `risk_flags` and `supporting_image_ids`. promptfoo averages the
score per prompt variant to produce the leaderboard; `componentResults` show the per-field
breakdown in the UI.

## Latest result (Qwen3-VL-30B-A3B)
| Prompt variant | Avg composite |
|---|---|
| full (rubric + requirements) | 0.527 |
| terse | 0.522 |
| no_rubric | 0.391 |

The production `full` prompt wins, confirming the choice in `code/evaluation/`.
