# Clinical Safety Summarization System Report

Date: 13 March 2026
Project: Clinical Safety Table to Regulatory Narrative Engine

## 1) What we are doing right now

The system is currently operating as a dual-backend hybrid platform for clinical safety table summarization:

- ML Backend (deterministic pipeline) for high factual safety and stable outputs.
- DL Backend (LLM-based polish and comparison) for fluency improvements with hard factual guardrails.
- Frontend workspace that can call ML, DL base, DL fine-tuned mode, or side-by-side comparison.

Current practical goal appears to be:

- Keep numeric hallucination risk near zero.
- Improve narrative quality and ROUGE via fine-tuned LoRA adapters.
- Preserve regulatory style and arm-accurate attribution for all percentages.

## 2) High-level architecture

The platform is built around two serving paths:

- Path A: Deterministic ML path (primary safety-first generation).
- Path B: DL path (rewrite/polish and model comparison), protected by hallucination verification and fallback behavior.

Operationally, both paths accept table text and file uploads, and both return:

- summary text
- verification status
- numeric accuracy score
- warnings
- timing metadata

## 3) ML model: what it is and how it operates

### 3.1 Core ML strategy

The ML system is a staged pipeline with deterministic generation logic:

1. Parse and normalize table structure and arm context.
2. Extract statistical and row-level features.
3. Retrieve nearest historical pattern from indexed corpus (KNN).
4. Fill clinical templates and render individual AE sentences.
5. Verify every numeric and arm-label attribution before output.

### 3.2 ML models and components in use

- LightGBM classifier for row-level content selection.
- KNN retrieval engine (cosine similarity over engineered features).
- Rule-based clinical slot extractor and template router.
- Deterministic text rendering plus post-processing.
- Hallucination guardian with numeric and arm-attribution checks.

### 3.3 Feature systems

Two feature families are active:

- Table-level statistical feature vector (32 features) for retrieval/routing behavior.
- Row-level AE feature vector (12 features) for content selection.

Row-level selection includes clinical override logic:

- SAE/fatal and severe-grade rows can be forced into output regardless of probability.

### 3.4 Why this ML path is important

This path is designed to be robust for regulated writing contexts because:

- number generation is not free-form
- arm attribution is explicit and checked
- deterministic fallback exists when uncertain

## 4) DL model: what it is and how it operates

### 4.1 Base DL architecture

The DL backend loads sequence-to-sequence transformer models and supports multiple runtime modes:

- ml: pure deterministic ML output
- base_dl: direct generation from base model prompt
- finetuned: ML-first output rewritten by DL with hard post-verification

### 4.2 Models currently configured

- Primary intended base model: google/flan-t5-xl (3B)
- Local fallback model for constrained environments: google/flan-t5-small
- Fine-tuned adapter support via PEFT/LoRA from local adapter directory

### 4.3 Runtime loading behavior

- On CUDA: quantized loading in 4-bit (BitsAndBytes, nf4-style config) with auto device map.
- On CPU/MPS: safer full-precision fallback behavior is used to avoid memory crashes.
- If fine-tuned adapter is unavailable or cannot load, system falls back to base behavior.

### 4.4 DL generation pattern

Finetuned mode is not unconstrained generation. It is:

1. ML deterministic summary first.
2. DL rewrite prompt to improve flow/style.
3. Hallucination guardian checks rewritten result.
4. If verification fails, revert to ML output.

This design keeps DL benefits while preserving factual safety requirements.

## 5) End-to-end process under the hood

### 5.1 Input ingestion

Accepted sources include linearized table text and document/image uploads.

Typical flow:

1. Receive file (PDF/DOCX/image).
2. Extract tables and convert to linearized table text.
3. For each extracted table, run selected mode (ML, base DL, finetuned DL).
4. Return structured result list and per-table metrics.

### 5.2 Generation internals (safety-critical path)

For each table:

1. Arm parser identifies control vs experimental columns.
2. Slot extractor maps clinically relevant rows (TEAE, Grade 3-4, SAE, discontinuation, drug-related signals).
3. Row selector picks significant individual events (LightGBM + overrides).
4. Template cluster routing chooses summary scaffold.
5. Optional KNN style adaptation injects de-lexicalized historical narrative pattern.
6. Surface realization produces narrative text.
7. Hallucination and arm-attribution guardian validates output.
8. Strict-mode behavior can block output on failed verification.

### 5.3 Comparison mode

The system can run ML, base DL, and finetuned DL in one request for direct quality/safety comparison.

## 6) Evaluation stack

Current evaluation pipeline includes:

- ROUGE-1, ROUGE-2, ROUGE-L
- Numeric accuracy checks
- Hallucination rate checks
- Leave-one-out style evaluation support

Threshold-oriented reporting is already present in evaluation logic for production readiness decisions.

## 7) What is implemented vs what is still evolving

### Implemented now

- Dual backend serving design (ML + DL)
- Multi-mode generation (ml, base_dl, finetuned, compare)
- Upload + extraction + summarization workflow
- LightGBM row selector and KNN retrieval
- Hallucination and arm attribution guardrails

### Still evolving

- Full deployment and consistent use of the latest high-capacity fine-tuned adapter in all target environments
- Alignment between documents/specs and active runtime defaults (some docs mention XXL plans while runtime paths favor XL/small fallback depending on hardware)
- Ongoing optimization for best fluency without guardrail regression

## 8) Current operating picture in one paragraph

The system is a safety-first clinical summarization engine that uses deterministic ML as the factual backbone and optional DL rewriting as a controlled enhancement layer. The ML path handles extraction, salience selection, retrieval, templating, and strict verification; the DL path improves narrative flow but is always checked and can be rejected in favor of ML output. This hybrid design is intentionally conservative for regulatory contexts where hallucinated numbers are unacceptable.

## 9) Recommended next actions

1. Finalize one canonical production profile for model loading by hardware tier (CPU, MPS, single-GPU, H100).
2. Version and lock adapter artifacts with explicit provenance and benchmark snapshots.
3. Run regular compare-mode regression packs to track fluency gains vs numeric safety.
4. Keep spec docs synchronized with actual runtime defaults to avoid operator confusion.
