# Multi-Modal Evidence Review — Solution

A VLM-based system that verifies damage claims (car / laptop / package) from submitted
images, the claim conversation, user history, and minimum-evidence requirements. For each
row in `dataset/claims.csv` it produces one row in `output.csv` with the required 14-column
schema.

## Approach

For each claim the system makes **one vision-language model call** that receives:
- the claim object type and the full conversation (English / Hindi / Hinglish),
- the applicable **minimum-evidence requirements**,
- the allowed enum values for each output field,
- **all of the claim's images** (base64) in a single request.

The model returns structured JSON (via `response_format` json_schema). A **deterministic
rule layer** then:
- merges **history-derived risk flags** from `user_history.csv` (`user_history_risk` implies
  `manual_review_required`) — these are not visible in images, so they are not trusted to the model,
- clamps every categorical field to its allowed enum (`schema.py`),
- enforces the consistency the labels exhibit (`not_enough_information` ⇒ no supporting images,
  unknown issue/severity; no visible issue ⇒ severity `none`),
- constrains `supporting_image_ids` to the actually-submitted image IDs.

Image preprocessing (`data.py`): every image is normalized to **RGB JPEG with the long edge
capped at 1,536px** before upload. This fixes mislabeled/RGBA/grayscale `.jpg` files (correct
MIME) and cut test-set input tokens ~46% and image payload ~82% with no accuracy loss (sample
macro 0.742 → 0.758). Tunable via `IMG_PREPROCESS`, `IMG_MAX_EDGE`, `IMG_JPEG_QUALITY` env vars.

Design highlights:
- **Prompt-injection defense:** text inside an image is described, never obeyed; instruction-like
  text triggers the `text_instruction_present` flag.
- **Authenticity:** screenshots / edited / reused images trigger `non_original_image` /
  `possible_manipulation`.
- **Caching:** every response is cached on disk (`code/cache/`) keyed by
  `sha256(model + messages + schema)`, so re-runs cost nothing.
- **Concurrency + retry:** a thread pool with exponential backoff handles rate limits / 5xx.

## Model & provider

- Provider: **DeepInfra** (OpenAI-compatible API, `https://api.deepinfra.com/v1/openai`).
- Model: **`Qwen/Qwen3-VL-30B-A3B-Instruct`** (vision, MoE — fast/cheap, strong structured output).
  - Note: `deepseek-ai/DeepSeek-V4-Flash` is **text-only** on DeepInfra (its API rejects image
    input with HTTP 405), so it cannot be used for this image-centric task. Any vision model can
    be selected via `--model` (e.g. `Qwen/Qwen3-VL-235B-A22B-Instruct`,
    `meta-llama/Llama-4-Scout-17B-16E-Instruct`).
- Auth: set `DEEPINFRA_TOKEN` (or `DEEPINFRA_API_KEY`). Never hardcode secrets.

## Layout

```
code/
├── main.py              # claims.csv -> output.csv
├── pipeline.py          # per-claim orchestration (+ strategy ablations)
├── vlm_client.py        # DeepInfra client: structured output, cache, retry
├── prompts.py           # system + user prompt construction
├── rules.py             # deterministic post-layer
├── schema.py            # output columns, enums, JSON schema
├── data.py              # CSV + image loading, base64, requirement lookup
├── cache/               # on-disk response cache (gitignored)
└── evaluation/
    ├── main.py          # score vs gold on sample_claims.csv; compare 2 strategies
    ├── metrics.py       # accuracy + set-F1
    └── evaluation_report.md
```

## Setup

```bash
pip install -r code/requirements.txt
cp code/.env.example code/.env   # then put your DeepInfra token in it
# or: export DEEPINFRA_TOKEN=...
```

## Run

```bash
# Final predictions for the test set -> output.csv (repo root)
python code/main.py

# Evaluate on the labeled sample set + write evaluation_report.md
python code/evaluation/main.py

# Helpful flags
python code/main.py --limit 3 --no-cache        # quick smoke run
python code/main.py --workers 6                  # concurrency
python code/main.py --input dataset/sample_claims.csv --output sample_output.csv
```

## Tried and rejected: adversarial verification pass

`verify.py` implements a skeptical second-opinion pass that re-audits each `supported`
decision for severity exaggeration / part-type-object mismatch (toggle with the `verify`
strategy flag). It was **measured on the sample and rejected**: the model overcorrects when
told to be skeptical (flipping ~7/13 correct decisions to `contradicted`), dropping macro
accuracy 0.742 → 0.633. It is kept behind the flag (OFF by default) and documented in
`evaluation/evaluation_report.md` as strategy `C_verify`.

## Notes

- All inputs are read from `dataset/`; paths resolve relative to the repo root, so the
  commands work from anywhere.
- No test labels or file-specific answers are hardcoded; the rule layer only encodes
  general, documented behavior.
- `output.csv` is written with the exact 14 required columns, fully quoted, one row per
  input claim.
