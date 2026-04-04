# Clinical Safety Summarization Platform — Project Overview (as of 30 Mar 2026)

## What problem we are solving
- Regulatory writers need fast first-draft narratives from clinical safety tables without invented numbers.
- Generic LLMs improve fluency but hallucinate; deterministic rules stay factual but are rigid. We need both safety and readability.

## Solution in one view
- **Hybrid ML + DL pipeline**: deterministic backbone for factuality plus a controllable LLM rewrite step for prose.
- **Hard verification gates**: numeric/arm attribution checks block any hallucinated number; if DL fails checks, we fall back to the ML output.
- **Multi-mode serving**: ML-only, base DL, fine-tuned DL, and compare modes to pick the best draft per case.

## How the system works (stages)
1. **Ingestion & feature extraction**: parse tables from PDF/DOCX/image into structured rows; compute 32 statistical features for signal strength.
2. **Retrieval & template grounding**: KNN over 187 human reports to pick a stylistic/template neighbor; templates clustered into 13 reporting patterns.
3. **Deterministic generation**: slot-fill counts/percentages with semantic aliasing (e.g., "Death" vs "Fatal Event") and study-ID injection.
4. **LLM rewrite (optional)**: Flan-T5 family with LoRA adapters for clinical tone; quantized paths for efficient serving.
5. **Verification & safety gates**: numeric grounding, hallucination guard, and arm attribution checks; automatic fallback to ML output on any failure.

## Methods and assets
- **ML components**: LightGBM importance classifier; KNN retrieval; templated slot-filling generator; post-processing artifact eraser.
- **DL components**: Flan-T5 XL adapters, LoRA fine-tuning, quantized runtime; compare-mode selector.
- **Evaluation suite**: ROUGE-L, Numeric Accuracy Rate (NAR), Hallucination Rate (HR), Overall Error (OR), and Arm-Correctness Rate (ACR); latency and cost tracked per mode.
- **Data foundation**: 187 curated gold summaries; structured extraction and de-lexicalization to avoid leakage.

## What we have built so far
- End-to-end pipeline wired into frontend upload flow (table → structured rows → ML draft → optional DL rewrite → safety verification).
- Multi-mode generation and comparison harness with benchmarking scripts under `scripts/` and `src/evaluation/`.
- Fine-tuned adapter artifact (XL) and quantized runtimes for efficient serving.
- Figure set and reporting assets for professor/leadership updates.

## Figure package (collected)
All generated PNGs have been copied to [data/figures/collection_20260330](data/figures/collection_20260330).
Key visuals include:
- Architecture and gate flow: figure1_architecture.png, figure5_gate.png, figure5_gate_breakdown.png.
- Faithfulness and safety metrics across modes: figure16_4mode_faithfulness_*.png, figure17_4mode_safety_fluency_*.png, figure19_4mode_latency_*.png, figure18_4mode_frontier_*.png.
- Core evaluation plots: figure2_nar*.png, figure3_hallucination*.png, figure4_safety_fluency_scatter.png, figure6_correlation.png, figure6_rouge_vs_complexity.png, figure7_loo_learning_curve.png, figure8_metric_correlation.png, figure9_error_composition.png, figure10_system_profile.png, figure11_ablation_safety_delta.png, figure12_ablation_metric_deltas.png.
- Aggregated snapshots: combined_core_results.png, combined_advanced_results.png.
- Latency and frontier comparisons: figure18_4mode_frontier_*.png, figure19_4mode_latency_*.png.

## Current status
- Architecture and safety gates implemented and validated on curated 10k-tier benchmarks.
- Evaluation artifacts and reports prepared for stakeholders (professor package, DOCX/MD reports, figure checklist).
- Backend and frontend are integrated; serving supports ML-only and DL-enhanced flows with automatic fallback for safety.

## What is next (suggested)
- Lock per-hardware deployment profiles (CPU/MPS/GPU/H100) and version the preferred adapter.
- Continue compare-mode benchmarking to maximize fluency while preserving zero hallucination.
- Keep documentation in sync with runtime defaults and refresh figures as new benchmarks land.
