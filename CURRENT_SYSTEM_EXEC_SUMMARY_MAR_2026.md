# Clinical Safety Summarization System — Executive Summary

Date: 13 March 2026

## 1) What this system does

This platform converts clinical safety tables into regulatory-style narrative summaries.

It is built to solve one core problem:

- Standard AI models can write fluent text, but they may invent numbers.
- In medical/regulatory writing, invented numbers are unacceptable.

So the system is designed to prioritize factual safety first, then writing quality.

## 2) How it works (simple view)

The system runs two coordinated engines:

- ML Engine (deterministic): Extracts and writes summaries with strict numeric grounding.
- DL Engine (LLM-based): Improves fluency, but only if output passes safety checks.

If the DL output fails numeric verification, the system automatically falls back to the ML output.

## 3) Current operating model

The production behavior is effectively:

1. Read table data from text/PDF/DOCX/image.
2. Convert to structured internal format.
3. Generate a factual baseline summary via ML.
4. Optionally rewrite via DL for better flow.
5. Verify all numbers and arm attribution.
6. Return only validated output.

## 4) What model types are being used

### ML side

- LightGBM classifier for selecting clinically important rows.
- KNN retrieval over historical examples for style/context matching.
- Rule/template generation for deterministic narrative construction.
- Hallucination and attribution guards to block unsafe output.

### DL side

- Flan-T5 family model path (XL target, smaller fallback when needed).
- LoRA adapter support for domain-tuned rewriting.
- Quantized runtime on compatible GPU for efficient serving.

## 5) Why this architecture is strong

This is not a single-model black box.

It is a controlled hybrid design:

- Deterministic backbone for reliability.
- Neural rewrite layer for readability.
- Hard verification gates before output release.

This gives a better balance between regulatory safety and human-like prose.

## 6) System status (current)

- Dual backends are implemented and integrated with frontend.
- Multi-mode generation is available (ML, base DL, fine-tuned DL, compare).
- Upload-based extraction and summarization flow is active.
- Evaluation metrics (ROUGE, numeric accuracy, hallucination rate) are in place.

## 7) Business takeaway

The platform is already in a strong pre-production/production-ready posture for safety-first draft generation.

Main value:

- Faster first-draft generation for clinical writers.
- Reduced manual extraction burden.
- Controlled risk profile via explicit verification and fallback.

## 8) Immediate focus areas

1. Standardize one deployment profile per hardware tier (CPU/MPS/GPU/H100).
2. Lock and version the best fine-tuned adapter artifact.
3. Continue compare-mode benchmarking to improve fluency without safety regressions.
4. Keep technical docs aligned with active runtime defaults.
