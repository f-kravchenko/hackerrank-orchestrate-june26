# Evaluation Report — Multi-Modal Evidence Review

Model: `Qwen/Qwen3-VL-30B-A3B-Instruct` (DeepInfra, OpenAI-compatible). Sample set: `20` labeled claims.

## 1. Strategy comparison

Three configurations were compared on the labeled sample set:

- **A_final** — full prompt (minimum-evidence requirements injected) + deterministic rule layer (history-flag merge, enum clamping, consistency rules).
- **B_ablation** — same model and images, but the prompt omits the evidence requirements and the rule layer is disabled (raw model output, enum-clamped only).
- **C_verify** — A_final plus an adversarial second-opinion pass that re-audits every `supported` decision for severity exaggeration / part-type-object mismatch.

| Strategy | Macro scalar acc | claim_status acc | severity acc | risk_flags F1 | supporting_image_ids F1 |
|---|---|---|---|---|---|
| A_final | 0.733 | 0.750 | 0.550 | 0.802 | 0.883 |
| B_ablation | 0.742 | 0.800 | 0.600 | 0.400 | 0.900 |
| C_verify | 0.675 | 0.600 | 0.350 | 0.572 | 0.883 |

_Note on variance: the model is a non-deterministic MoE, so even at temperature=0 the 20-sample macro accuracy varies ~±0.02 run-to-run. Differences below ~0.03 are within noise; only larger effects (the rule layer's risk-flag gain, and the rejected CoT / verification / 235B experiments) are treated as real signal._

**Final strategy: A_final.** Injecting the minimum-evidence requirements grounds the evidence-sufficiency decision, and the deterministic rule layer fixes the history-driven risk flags (which are not visible in the images) and enforces the consistency the labels exhibit (e.g. `not_enough_information` => no supporting images, unknown issue/severity).

**C_verify was tested and rejected.** A skeptical second-opinion pass was expected to catch severity-exaggeration and mismatch cases the lenient primary model misses. In practice it *overcorrects*: when told to be skeptical, the model systematically re-perceives damage as absent or mismatched and overturns correct `supported` decisions (it flipped ~7 of 13 correct decisions to `contradicted` on the sample). No conservative merge policy (keep-primary-severity, require-mismatch-flag, contradicted-only) recovered baseline accuracy. This is a model-capability bias, not a prompt-tuning gap, so the pass is kept available behind a flag (`verify`) but is OFF by default.

## 1b. Model comparison (A_final config)

A larger vision model was tested on the same sample set and config. It did **not** help:

| Model | Macro | claim_status | severity | risk_flags F1 | supporting_image_ids F1 |
|---|---|---|---|---|---|
| `Qwen3-VL-30B-A3B` (chosen) | 0.742 | 0.750 | 0.550 | 0.665 | 0.883 |
| `Qwen3-VL-235B-A22B` | 0.683 | 0.550 | 0.400 | 0.689 | 0.783 |

The 235B model **over-contradicts** (it flipped 4 genuinely-supported claims to `contradicted`) — the same failure mode as the rejected verification pass. On this dataset, where most claims are genuinely supported, the 30B's mild leniency aligns better with the labels, and it is cheaper. **Chosen model: `Qwen3-VL-30B-A3B`.** (This comparison used identical raw-image inputs for both models; with image preprocessing enabled the chosen 30B config scores macro 0.758 — see §2.)

## 1c. Prompt enhancements tried

Two prompt enhancements were measured on the sample (held in `prompts.py`/`schema.py`):

- **Severity anchors (kept)** — explicit per-object definitions of none/low/medium/high. Macro stayed within noise but `risk_flags` F1 improved robustly (~0.66 without anchors -> ~0.80 with anchors + rule layer), so the anchors are part of the final prompt.
- **Chain-of-thought `image_observations` field (rejected)** — forcing a perception-first observation list before the verdict *lowered* accuracy (macro 0.758 -> 0.717; severity and issue_type both dropped). On this lenient-gold dataset the extra analysis makes the model drift toward over-calling mismatches — the same pattern as the 235B model and the verification pass. Removed.

## 2. Detailed metrics

### Final: full prompt + rule layer (n=20)

**Macro scalar accuracy:** 0.733

| Field | Metric | Score |
|---|---|---|
| evidence_standard_met | accuracy | 0.900 |
| valid_image | accuracy | 0.800 |
| issue_type | accuracy | 0.550 |
| object_part | accuracy | 0.850 |
| claim_status | accuracy | 0.750 |
| severity | accuracy | 0.550 |
| risk_flags | precision | 0.846 |
| risk_flags | recall | 0.796 |
| risk_flags | F1 | 0.802 |
| supporting_image_ids | precision | 0.900 |
| supporting_image_ids | recall | 0.875 |
| supporting_image_ids | F1 | 0.883 |

### Ablation: no evidence requirements, no rule layer (n=20)

**Macro scalar accuracy:** 0.742

| Field | Metric | Score |
|---|---|---|
| evidence_standard_met | accuracy | 0.900 |
| valid_image | accuracy | 0.800 |
| issue_type | accuracy | 0.450 |
| object_part | accuracy | 0.900 |
| claim_status | accuracy | 0.800 |
| severity | accuracy | 0.600 |
| risk_flags | precision | 0.458 |
| risk_flags | recall | 0.392 |
| risk_flags | F1 | 0.400 |
| supporting_image_ids | precision | 0.900 |
| supporting_image_ids | recall | 0.900 |
| supporting_image_ids | F1 | 0.900 |

### Final + adversarial verification pass (rejected) (n=20)

**Macro scalar accuracy:** 0.675

| Field | Metric | Score |
|---|---|---|
| evidence_standard_met | accuracy | 0.900 |
| valid_image | accuracy | 0.800 |
| issue_type | accuracy | 0.550 |
| object_part | accuracy | 0.850 |
| claim_status | accuracy | 0.600 |
| severity | accuracy | 0.350 |
| risk_flags | precision | 0.572 |
| risk_flags | recall | 0.629 |
| risk_flags | F1 | 0.572 |
| supporting_image_ids | precision | 0.900 |
| supporting_image_ids | recall | 0.875 |
| supporting_image_ids | F1 | 0.883 |

## 3. Error analysis

**Confusion for `claim_status`** (rows = gold, cols = predicted):

| gold \ pred | contradicted | not_enough_information | supported |
|---|---|---|---|
| contradicted | 1 | 1 | 3 |
| not_enough_information | 0 | 1 | 1 |
| supported | 0 | 0 | 13 |

## 4. Operational analysis

- **Model calls (this eval run):** 0 live API calls, 77 cache hits (77 logical calls across both strategies over 20 sample claims).
- **Calls per claim:** exactly 1 (all of a claim's images are sent in a single request).
- **Observed token usage:** ~166,638 input + ~15,328 output tokens across 77 calls (avg ~2,164 in / ~199 out per call). Image tokens dominate input.
- **Images processed:** sample = ~30 images across 20 claims; test = 82 images across 44 claims.
- **Measured test-set run:** 44 calls over 82 images, ~116,742 input + ~9,237 output tokens, ~85.0s wall-clock.
- **Measured test-set cost:** ~$0.0231 (DeepInfra Qwen3-VL-30B-A3B pricing: $0.15/M input, $0.60/M output).
- **Latency/runtime:** dominated by per-image upload + inference; with 6-way concurrency the 44-claim test set typically completes in a few minutes.
- **TPM/RPM strategy:** 6-way thread pool with exponential backoff retry (1,2,4,8,16,30s) on rate-limit / 5xx errors; well within DeepInfra limits for ~44–64 calls.
- **Image preprocessing:** every image is normalized to RGB JPEG with the long edge capped at 1,536px before upload. This fixes mislabeled/RGBA/grayscale files and cut test-set input tokens ~46% (207.5K -> 111.2K) and image payload ~82%, with sample accuracy held/slightly improved (macro 0.742 -> 0.758).
- **Caching:** every response is cached on disk keyed by sha256(model + messages + schema), so re-runs and dev iterations cost $0 and avoid duplicate image uploads.
- **Determinism:** temperature=0 + json_schema structured output + a deterministic rule layer make outputs reproducible given the same inputs and cache.
