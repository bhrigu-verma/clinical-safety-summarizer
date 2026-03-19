# Clinical Safety Summarization Project - Full Implementation Deep Dive

Date: 16 March 2026  
Workspace: /Users/bhriguverma/medicalproject

## Layman Explanation (17 Human Lines)
This project takes difficult medical safety tables and turns them into readable clinical summaries.
It is built for pharmaceutical trial reporting where numbers must be exact.
A normal AI model can write well, but it can also invent numbers.
In this domain, invented numbers are dangerous and unacceptable.
So the system first uses a strict rule-based and machine learning pipeline for factual safety.
Then it optionally uses deep learning to improve writing quality.
If the deep learning rewrite changes numbers incorrectly, the system rejects it.
When that happens, it falls back to the safe deterministic summary.
The result is a balance of safety and readability.
The platform can process plain table text, PDFs, DOCX files, and images.
It extracts table data, identifies treatment arms, and builds structured facts.
It checks if every output number came from the source table.
It also checks if percentages are attached to the correct treatment arm.
Users can compare ML output, base DL output, and fine-tuned DL output side by side.
A web app provides an interface for submitting inputs and reviewing warnings.
Benchmark scripts measure ROUGE, numeric accuracy, and hallucination rate.
Overall, this is a safety-first clinical writing assistant, not an uncontrolled text generator.

---

## 1. Project Objective and Product Scope

The implemented system converts structured clinical safety tables (primarily adverse-event tables from CSRs) into regulatory-style narrative summaries.

Primary implementation objective:
- Maximize factual correctness (numeric grounding + arm attribution)
- Preserve practical readability for medical writing workflows
- Support both deterministic and neural generation paths with guardrails

Current architecture is intentionally hybrid:
- Deterministic ML pipeline for high reliability
- DL rewrite/generation modes for fluency improvements
- Verification gate between generation and returned output

---

## 2. Top-Level Repository Implementation Map

### 2.1 Core documents and reports
- PROJECT_SPECIFICATION.md
- BACKEND_TECHNICAL_SPEC.md
- CURRENT_SYSTEM_REPORT_MAR_2026.md
- CURRENT_SYSTEM_EXEC_SUMMARY_MAR_2026.md
- CURRENT_SYSTEM_ENGINEERING_DEEP_DIVE_MAR_2026.md
- ML_SYSTEM_IMPLEMENTATION_MAR_2026.md
- DL_SYSTEM_IMPLEMENTATION_MAR_2026.md
- SUMMARY_GENERATION_PROCESS.md
- ROADMAP_TO_PERFECTION.md

These provide architecture intent, operational summaries, and roadmap direction. Some roadmap docs include aspirational targets, while runtime behavior is governed by the Python services described below.

### 2.2 Runtime services
- src/api/main_ml.py (FastAPI, port 8000)
- src/api/main_dl.py (FastAPI, port 8001)

### 2.3 Frontend app
- frontend/ (Next.js 16 + React 19 + TypeScript + Tailwind v4 + Framer Motion)

### 2.4 Data, model, and training assets
- data/ (processed pairs, retrieval index, augmentation, benchmark tiers/results)
- models/ (content selector and adapter assets)
- scripts/ (data prep, training, benchmarking, evaluation, orchestration)
- notebooks/ (extraction and training notebooks/scripts)

---

## 3. Implemented Data Assets and Corpus State

### 3.1 Confirmed local data artifacts
- clinical_10k_final.jsonl: 9,630 lines
- data/augmented/synthetic_1000.jsonl: 437 lines
- data/benchmark/tier1_gold.jsonl: 41 lines
- data/benchmark/tier2_stress.jsonl: 78 lines
- data/retrieval_index.pkl present
- data/processed/raw_pairs.json present
- data/processed/knn_pairs.json present

### 3.2 CSR source corpus
- pfizer_csrs/: 147 PDF files present
- pfizer_csrs/trial_index.json present

### 3.3 Model artifacts present
- models/content_selector.pkl
- models/best_clinical_adapter_xl/
  - adapter_config.json
  - tokenizer assets
  - README.md

Note: The adapter directory currently contains tokenizer and config artifacts. Production LoRA behavior on CUDA depends on runtime loading in src/api/main_dl.py and availability of full adapter weight files expected by PEFT workflows.

---

## 4. Input and Output Contract (Implemented)

### 4.1 Canonical input format
The primary contract is a linearized table string:
- start_table ... end_table delimiters
- [TABLE_TITLE: ...]
- [HEADERS: | arm columns | ...]
- repeated [ROW] sections

### 4.2 Supported file ingestion
Implemented in APIs:
- PDF
- DOCX/DOC
- JPEG/JPG
- PNG
- BMP
- TIFF

### 4.3 Output structure
Both backends return structured JSON including:
- summary text
- verification status
- numeric accuracy estimate
- warnings list
- latency metadata

ML backend additionally returns cluster metadata and optional feature payloads.

---

## 5. Backend Service A: ML Engine (src/api/main_ml.py)

### 5.1 API surface implemented
- GET /health
- POST /summarize
- POST /upload
- POST /summarize/batch
- GET /clusters
- GET /features/names
- POST /evaluate/single

### 5.2 Runtime initialization
- StatisticalFeatureExtractor instantiated
- SlotFillGenerator instantiated
- Attempts loading KNNRetrievalEngine from data/retrieval_index.pkl

### 5.3 /summarize behavior
Flow:
1. Optional nearest-neighbor retrieval (if index exists)
2. Deterministic generation via SlotFillGenerator
3. Optional 32-feature extraction when return_features=true
4. Structured response with verification and cluster metadata

### 5.4 /upload behavior
Flow:
1. Detect file type
2. Route extraction:
   - image_extractor for images
   - pdf_extractor for PDF/DOCX path
3. For each detected table, run same summarization pipeline
4. Return per-table result list

### 5.5 Batch and evaluation support
- Batch summarization limit enforced (50 tables)
- Single prediction evaluation bridged to src/evaluation/eval_suite.py

---

## 6. Deterministic Generation Core (src/generation/slot_fill_generator.py)

This is the central safety-first generation module.

### 6.1 Arm parsing and mapping
Implemented in ArmParser:
- Parses headers for arm labels and N counts
- Uses CONTROL_KEYWORDS to infer control vs experimental
- Outputs ArmContext with experimental_arm_index
- Includes fallback behavior when headers are weak

This directly addresses common column-swap and arm-labeling failures.

### 6.2 Overview slot extraction
Implemented in OverviewSlotExtractor:
- Regex-driven row typing for TEAE, SAE, Grade 3-4, discontinuation, fatal, and drug-related variants
- Extracts arm-specific n and % values
- Populates slot keys consumed by template clusters

### 6.3 Individual AE sentence generation path
Implemented in IndividualAERenderer + ContentSelector + RowFeatureExtractor:
- Parses individual AE rows
- Selects salient rows
- Generates explicit per-row comparative sentences with arm names

### 6.4 Cluster routing and template fill
- select_cluster() from template_clusters.py routes table to cluster family
- required slots checked
- template filled and merged with AE sentence block
- optional KNN adaptation can inject de-lexicalized retrieved structure

### 6.5 Hallucination and attribution guard
Implemented in HallucinationGuardian:
- Number-in-source verification
- Arm attribution verification for percentages linked to arm labels
- Numeric accuracy estimate
- Warnings emitted for unsupported values/attribution mismatches

### 6.6 Strict mode behavior
- strict_mode can raise HallucinationError when verification fails
- default runtime in APIs is generally non-blocking with warnings

---

## 7. Feature Engineering (Implemented)

### 7.1 Table-level features (32) - statistical_features.py
Implemented categories:
- numeric distribution stats
- percentage stats
- arm and cohort features
- structural/table shape features
- clinical signal flags
- additional roadmap-aligned features such as discontinuation_delta, grade_3_4_ratio, data completeness, imbalance

These power retrieval and routing behavior.

### 7.2 Row-level features (12) - row_feature_extractor.py
Implemented features include:
- drug/control percentages
- absolute risk difference and direction
- relative risk
- within-table ranking features
- SAE/severe flags
- practical thresholds (>5%, >10%, >5pp)

Overview rows are filtered from individual AE selection.

---

## 8. Content Selection Model (Implemented)

### File
- src/models/content_selector.py

### 8.1 Modes
- Trained LightGBM inference (if model file exists)
- Rule-based fallback when model unavailable

### 8.2 Selection logic
Regulatory-first override order:
1. Always include SAE/fatal
2. Always include severe/Grade 3-4
3. Always include high absolute risk difference
4. Then apply LightGBM threshold logic
5. Safety net on drug_pct threshold even when model score is below threshold

### 8.3 Model persistence
- load()/save() around models/content_selector.pkl
- threshold tuning method exposed

### 8.4 Training interface present
- fit() method with LightGBM + class balancing
- evaluate() helper for holdout metrics

---

## 9. Retrieval Layer (Implemented)

### File
- src/retrieval/knn_retrieval_engine.py

### 9.1 Capabilities
- Build retrieval index from table-writeup pairs
- Standardize features
- KNN search with cosine metric (brute-force exact search)
- retrieve(), retrieve_best(), persistence save/load
- LOO support (build_loo) for cross-validation workflows

### 9.2 Runtime integration
- ML backend attempts to load data/retrieval_index.pkl
- On successful load, retrieve_best influences template adaptation in SlotFillGenerator

---

## 10. Backend Service B: DL Engine (src/api/main_dl.py)

### 10.1 API surface implemented
- GET /health
- POST /summarize
- POST /summarize-compare
- POST /upload
- POST /model/load

### 10.2 Supported generation modes
- ml
- base_dl
- finetuned

### 10.3 Model loading policy
Configured IDs:
- BASE_MODEL_ID = google/flan-t5-xl
- FALLBACK_MODEL_ID = google/flan-t5-small
- ADAPTER_PATH = models/best_clinical_adapter_xl

Device policy:
- CUDA: attempts 4-bit quantized loading with bitsandbytes
- CPU/MPS: safer fallback behavior; avoids high-memory XL+adapter load

### 10.4 Finetuned runtime design (important)
Production finetuned path is hybrid rewrite, not pure end-to-end generation:
1. Generate strict ML summary
2. Rewrite via DL prompt
3. Verify rewritten output
4. Accept rewrite only when verified and accuracy threshold is met
5. Otherwise fall back to ML summary

### 10.5 Comparison mode
/summarize-compare computes and returns all three paths in one response:
- ML result
- base DL result
- finetuned DL result

### 10.6 Upload in DL backend
Supports same file classes as ML backend and applies selected mode per extracted table.

---

## 11. Data Processing and Extraction Stack (Implemented)

### 11.1 PDF and DOCX extractor
File: src/data_processing/pdf_extractor.py

Implemented behavior:
- PDF text extraction via pdfplumber
- Scanned PDF fallback via PyMuPDF image rendering
- DOCX table parsing via python-docx
- Routes image input to image extractor
- Outputs ClinicalTablePair objects with linearized table text and metadata

### 11.2 Image extractor
File: src/data_processing/image_extractor.py

Implemented behavior:
- Preprocess (denoise, deskew, threshold, upscale)
- Primary extraction via img2table + EasyOCR
- Fallback OCR reconstruction path
- Produces table HTML, JSON, headers, linearized string, and confidence metadata

### 11.3 Additional processing modules present
- src/data_processing/augmenter.py
- src/data_processing/splitter.py

---

## 12. Training and Fine-Tuning Implementation

### 12.1 DL training script
File: src/training/finetune_dl.py

Implemented training stack:
- Base model target: google/flan-t5-xl
- QLoRA path with 4-bit NF4 (CUDA)
- LoRA config and target modules declared
- Dataset loading from data/augmented/synthetic_1000.jsonl
- Train/val split via Hugging Face datasets
- Seq2SeqTrainer with early stopping and ROUGE evaluation

### 12.2 Practical note
The script includes a trailing execution note and should be validated before production runs in a clean environment.

---

## 13. Evaluation and Benchmarking Toolchain

### 13.1 Core evaluation module
File: src/evaluation/eval_suite.py

Implemented metrics:
- ROUGE-1, ROUGE-2, ROUGE-L
- numeric accuracy
- hallucination rate
- aggregate stats (mean/std/p10/p50/p90)
- threshold checks
- single example evaluation
- LOO evaluation flow
- baseline template evaluation flow

### 13.2 Benchmark pipeline scripts
- scripts/benchmark_extract.py (tier construction)
- scripts/benchmark_run.py (multi-model execution)
- scripts/benchmark_report.py (aggregation + report generation)
- scripts/benchmark_10k.py
- scripts/benchmark_10k_dl.py
- scripts/benchmark_run.py

### 13.3 Additional QA/eval helpers
- scripts/measure_all_metrics.py
- scripts/run_loo_eval.py
- scripts/baseline_rouge.py
- scripts/testinference.py

---

## 14. Frontend Implementation (Next.js)

### 14.1 Stack
- Next.js 16.1.6
- React 19.2.3
- TypeScript 5
- Tailwind CSS v4
- Framer Motion
- Radix primitives
- Lucide icons

### 14.2 Pages implemented
- frontend/app/page.tsx (landing + architecture framing)
- frontend/app/summarizer/page.tsx (main execution workspace)
- frontend/app/research/page.tsx (methodology and tradeoffs)
- frontend/app/about/page.tsx (project context and roadmap)
- frontend/app/layout.tsx (shared nav/footer and typography)
- frontend/app/globals.css (design tokens and styling)

### 14.3 Summarizer capabilities implemented
- mode selection: ML, DL Base, DL Fine-Tuned, Compare
- backend health checks (ports 8000/8001)
- table input, result rendering, and warnings
- local run history in localStorage
- markdown export/copy/download for run outputs

### 14.4 Frontend component groups
- landing components: hero, metrics-grid, pipeline-visual, tech-stack
- summarizer components: table-input, result-card, approach-selector
- shared components: section-header, loaders, ecg-line

---

## 15. Integration Contracts and Live Coupling

### 15.1 Frontend to backend route map (implemented intent)
- ML mode -> http://localhost:8000/summarize
- DL Base mode -> http://localhost:8001/summarize
- DL Fine-Tuned mode -> http://localhost:8001/summarize
- Compare mode -> http://localhost:8001/summarize-compare

### 15.2 Observed integration drift to monitor
Based on current code shapes:
- compare response parsing in frontend expects a nested results object, while DL backend compare endpoint returns top-level ml/base_dl/finetuned_dl fields
- warning key naming differs between frontend expectation (hallucination_flags) and API payload (warnings)

These are integration alignment items, not architectural blockers.

---

## 16. Scripts and Operational Utilities

Notable implemented script groups:

### 16.1 Data and preprocessing
- scripts/parse_dataset.py
- scripts/scrub_dataset.py
- scripts/generate_synthetic_data.py
- scripts/datasynt.py
- scripts/scraping/fetch_pfizer.py
- pfizzer_scraper.py

### 16.2 Training
- scripts/train_content_selector.py
- src/training/finetune_dl.py
- notebooks/02_train_t5xxl_optimized.py

### 16.3 Benchmark and reporting
- scripts/benchmark_extract.py
- scripts/benchmark_run.py
- scripts/benchmark_report.py
- scripts/benchmark_10k.py
- scripts/benchmark_10k_dl.py

### 16.4 Orchestration
- scripts/setup_and_run.py

---

## 17. Deployment and Environment Assets

### 17.1 Python dependencies
Two dependency manifests exist:
- root requirements.txt (broad stack)
- src/requirements.txt (more pinned backend-centric stack)

### 17.2 Containerization
- src/docker-compose.yml defines:
  - ml-backend service (port 8000)
  - dl-backend service (port 8001)
  - mounted data/models volumes
  - healthcheck and GPU reservation configuration patterns

### 17.3 Frontend runtime
- frontend/package.json with scripts: dev, build, start, lint

---

## 18. End-to-End Execution Flows (What Is Actually Running)

### 18.1 Text table summarization flow
1. User submits linearized table text
2. Backend mode selected (ML or DL)
3. ML path:
   - parse arms
   - extract slots/features
   - select rows
   - choose/fill template
   - verify and return
4. DL path:
   - mode-specific generation
   - verify
   - fallback if unsafe
5. UI displays result cards with metadata

### 18.2 File upload summarization flow
1. User uploads PDF/DOCX/image
2. Extractor parses table(s)
3. Each extracted table converted to canonical linearized format
4. Chosen mode runs per table
5. Aggregated results returned with timing and warnings

---

## 19. Current Strengths (Implemented)

- Hybrid architecture with deterministic safety core
- Multi-format ingestion pipeline (text, PDF, DOCX, image)
- Explicit arm parsing and attribution checks
- LightGBM + rule override content selection
- Retrieval-assisted adaptation support
- DL polish path with guard and fallback
- Multi-mode compare endpoint and UI mode switching
- Established benchmarking/evaluation scripts and report outputs

---

## 20. Current Gaps or Risks (Implementation-Level)

- Runtime docs and roadmap docs are not perfectly synchronized
- Adapter loading behavior differs by hardware tier; reproducibility depends on environment
- Some frontend/API response shape mismatches should be aligned
- Multiple dependency manifests can drift without lock-step maintenance
- Several scripts/notebooks appear experimental and need per-file validation before production automation

---

## 21. Implementation Completeness Summary

This repository is not a prototype skeleton; it contains a full working system with:
- two serving backends,
- integrated extraction pipelines,
- deterministic and neural summarization modes,
- frontend execution interface,
- benchmark/evaluation infrastructure,
- model and data artifacts for continued iteration.

The system is best characterized as a safety-first clinical summarization platform in active optimization, with core architecture already implemented and operational for local and staged deployment workflows.
