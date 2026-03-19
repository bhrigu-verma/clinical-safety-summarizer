# Clinical Safety Summarization Improvement Plan

Date: 13 March 2026

## 1. Goal

Upgrade the platform from strong safety-first performance to consistently high fluency plus reproducible production behavior, while preserving strict factual guarantees.

Primary success condition:

- No regression in numeric safety.
- Higher narrative quality and consistency.
- Cleaner operations across hardware tiers.

## 2. Current strengths and gaps

### Strengths

- Deterministic ML backbone with strong guardrails.
- DL polish path with fallback on verification failure.
- Dual backend architecture with compare mode.
- Working extraction + summarization flow for text and uploaded files.

### Gaps to close

- Runtime behavior differs by hardware, causing quality variance.
- Documentation and real runtime defaults are not fully synchronized.
- Adapter lifecycle and benchmark evidence are not yet fully version-locked.
- Continuous regression and release gating are not strict enough.

## 3. North-star metrics

Track these every release:

1. Safety metrics
- Numeric accuracy mean and p10.
- Hallucination rate mean and p90.
- Arm-attribution error rate.

2. Quality metrics
- ROUGE-1, ROUGE-2, ROUGE-L.
- Human reviewer quality score (clarity, tone, regulatory fit).

3. Reliability and cost metrics
- P50/P95 latency by mode.
- Failure and fallback rates.
- Throughput and GPU memory headroom.

4. Product metrics
- Percentage of summaries accepted without manual rewrite.
- Average edit distance between generated text and final writer version.

## 4. 30/60/90 day execution roadmap

## Phase A (Days 1-30): Stabilize production behavior

Objective: make outputs predictable and reproducible.

1. Define runtime profiles by hardware tier
- CPU profile.
- MPS profile.
- Single CUDA GPU profile.
- H100 profile.

Deliverable:
- One configuration matrix with explicit model, quantization, token limits, beam settings, and fallback policy per tier.

2. Pin model and adapter provenance
- Add adapter fingerprinting and metadata manifest.
- Record training dataset hash, training config hash, and eval snapshot.

Deliverable:
- Versioned adapter registry document and load-time validation.

3. Formal release gate for safety
- Block release if any safety metric breaches threshold.
- Include compare-mode regression on fixed benchmark set.

Deliverable:
- CI quality gate report artifact per candidate release.

4. Align docs with runtime truth
- Update architecture docs to exactly match active defaults.
- Keep one canonical runtime spec.

Deliverable:
- Single source-of-truth architecture and operations document.

## Phase B (Days 31-60): Improve quality without safety drift

Objective: improve prose quality while preserving guardrails.

1. Tune DL polish policy
- Optimize prompt templates for rewrite style.
- Calibrate beam/length/penalty settings by mode and table type.
- Introduce guarded selective polish only when confidence is high.

Deliverable:
- New inference policy pack with measurable ROUGE and edit-distance gains.

2. Upgrade style routing on ML side
- Strengthen cluster/template routing logic for better baseline fluency.
- Expand deterministic template diversity by table profile.

Deliverable:
- Extended template family and routing tests.

3. Improve content selector calibration
- Tune LightGBM threshold by target recall/precision profile.
- Retrain with updated labels from real writer edits.

Deliverable:
- New selector model version with calibration report.

4. Expand benchmark coverage
- Add stratified benchmark slices: severe SAE tables, sparse tables, single-arm cases, OCR-noisy inputs.

Deliverable:
- Tiered benchmark dashboard with per-slice performance.

## Phase C (Days 61-90): Scale quality operations and governance

Objective: institutionalize quality and release confidence.

1. Human-in-the-loop quality loop
- Collect reviewer edits and structured failure tags.
- Feed back into selector training and DL rewrite tuning.

Deliverable:
- Monthly retraining package and quality drift report.

2. Observability and incident playbook
- Add per-request trace IDs and structured event logs.
- Alerting on fallback spikes, attribution errors, and extraction failures.

Deliverable:
- Monitoring dashboard and on-call runbook.

3. Regulatory-readiness artifacts
- Produce auditable model card, data card, and verification protocol.
- Document deterministic fallback and blocked-output behavior.

Deliverable:
- Audit bundle for internal QA and external review.

## 5. Prioritized work backlog

Priority P0 (start now)

1. Runtime profile matrix and config lock.
2. Adapter fingerprint/version manifest.
3. CI release gate for safety metrics.
4. Canonical runtime doc update.

Priority P1

1. DL prompt and decoding policy tuning.
2. Template/routing expansion.
3. Selector threshold calibration and retraining.

Priority P2

1. HITL feedback ingestion tooling.
2. Full observability and incident automation.
3. Audit bundle generation pipeline.

## 6. Recommended thresholds for release decisions

Use hard pass/fail for deployment:

1. Hallucination rate mean less than 0.01 in candidate benchmark.
2. Arm attribution error rate equals 0 for critical benchmark slices.
3. Numeric accuracy mean at least 0.98.
4. No P95 latency regression above agreed budget (mode-specific).
5. Compare-mode shows no safety regression versus previous stable release.

## 7. Team execution structure

1. ML owner
- Selector model retraining, threshold calibration, routing quality.

2. DL owner
- Adapter lifecycle, decoding policy tuning, rewrite quality experiments.

3. Platform owner
- Runtime profile matrix, CI gates, observability, release controls.

4. Product/clinical owner
- Human review rubric, acceptance criteria, failure taxonomy.

## 8. Immediate next 10 actions

1. Freeze current baseline metrics on fixed benchmark.
2. Create hardware-tier config file set and enforce in startup.
3. Add adapter metadata validation at load time.
4. Implement release gate script with threshold checks.
5. Add benchmark slices for severe and OCR-noisy inputs.
6. Tune finetuned mode decoding on one controlled branch.
7. Expand template set for high-toxicity and low-event profiles.
8. Recalibrate LightGBM threshold with recent outputs.
9. Add trace IDs and structured warnings in both APIs.
10. Publish weekly quality report with trend lines.

## 9. Expected outcomes after 90 days

1. More consistent output quality across all hardware.
2. Higher fluency and reduced writer rewrite effort.
3. Stronger operational confidence through hard release gates.
4. Cleaner auditability and regulatory-readiness posture.
