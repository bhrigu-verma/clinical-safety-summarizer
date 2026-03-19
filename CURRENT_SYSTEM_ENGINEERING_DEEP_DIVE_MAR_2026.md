# Clinical Safety Summarization System — Engineering Deep Dive

Date: 13 March 2026

## 1) Runtime topology

Two FastAPI services are used:

- ML backend on port 8000
- DL backend on port 8001

Frontend health checks and execution modes call both services.

## 2) ML backend internals (port 8000)

Primary file: src/api/main_ml.py

### 2.1 Pipeline stages

1. Input preparation
- Accepts text summarization requests and file uploads.
- Supported uploads include PDF, DOCX, JPEG, PNG, BMP, TIFF.

2. Retrieval augmentation (optional when index exists)
- KNN index loaded from data/retrieval_index.pkl.
- Retrieves nearest historical pair for style/template adaptation.

3. Deterministic generation
- SlotFillGenerator composes summary from parsed rows + template cluster + selected AE rows.
- Returns structured summary metadata including cluster and verification output.

4. Safety checks
- Hallucination guard validates numeric grounding.
- Arm attribution checks validate that percentages match the correct arm column.

### 2.2 Important core modules

- src/generation/slot_fill_generator.py
  - ArmParser
  - OverviewSlotExtractor
  - IndividualAERenderer
  - HallucinationGuardian
- src/models/content_selector.py (LightGBM + rule fallback)
- src/retrieval/knn_retrieval_engine.py
- src/feature_engineering/statistical_features.py (32 table-level features)
- src/feature_engineering/row_feature_extractor.py (12 row-level features)

### 2.3 ML endpoint examples

#### POST /summarize (text input)
Request body:
```json
{
  "table_text": "start_table [TABLE_TITLE: ...] [HEADERS: ...] [ROW] ... end_table",
  "return_features": true,
  "strict_mode": false
}
```

Typical response shape:
```json
{
  "summary": "...",
  "verified": true,
  "cluster_id": "C01",
  "cluster_description": "...",
  "numeric_accuracy": 1.0,
  "hallucination_free": true,
  "inference_time_ms": 42.3,
  "warnings": [],
  "features": {"num_count": 123, "...": 0},
  "slots_filled": {"TEAE_PCT_B": "92.6", "...": "..."}
}
```

#### POST /upload (document/image input)
Form fields:
- file (required)
- use_gpu_ocr (optional bool)

Typical response shape:
```json
{
  "filename": "sample.pdf",
  "file_type": "pdf",
  "tables_found": 2,
  "results": [
    {
      "table_id": "table_1",
      "table_text": "...",
      "summary": "...",
      "verified": true,
      "numeric_accuracy": 1.0,
      "warnings": []
    }
  ],
  "total_time_ms": 812.7
}
```

#### Other ML utility endpoints
- GET /health
- POST /summarize/batch
- GET /clusters
- GET /features/names
- POST /evaluate/single

## 3) DL backend internals (port 8001)

Primary file: src/api/main_dl.py

### 3.1 Model loading behavior

Configured model path:

- Base target: google/flan-t5-xl
- Fallback: google/flan-t5-small
- Adapter path: models/best_clinical_adapter_xl

Hardware-aware load policy:

- CUDA: 4-bit quantized path with bitsandbytes config.
- CPU/MPS: safer fallback path to avoid OOM on large model + adapter loads.

### 3.2 DL generation modes

- ml: deterministic ML output only.
- base_dl: direct DL summarization prompt on base model.
- finetuned: ML summary first, then DL rewrite polish, then guard validation.

The finetuned mode is guarded:

1. Generate strict ML summary.
2. Rewrite with DL model.
3. Verify rewritten text numbers/attribution.
4. If fails threshold, revert to ML summary.

### 3.3 DL endpoint examples

#### POST /summarize
Request body:
```json
{
  "table_text": "start_table ... end_table",
  "mode": "finetuned",
  "max_new_tokens": 512,
  "num_beams": 4,
  "temperature": 0.0,
  "verify_numbers": true
}
```

Typical response shape:
```json
{
  "summary": "...",
  "model_used": "dl_polished/finetuned_xl",
  "verified": true,
  "numeric_accuracy": 1.0,
  "inference_time_ms": 210.6,
  "warnings": [],
  "tokens_generated": 148
}
```

#### POST /summarize-compare
Request body:
```json
{
  "table_text": "start_table ... end_table",
  "max_new_tokens": 512,
  "num_beams": 4,
  "temperature": 0.0
}
```

Typical response shape:
```json
{
  "ml": {"summary": "...", "model_used": "ml/lgbm+jinja2", "verified": true, "numeric_accuracy": 1.0},
  "base_dl": {"summary": "...", "model_used": "dl_base/flan-t5", "verified": true, "numeric_accuracy": 0.97},
  "finetuned_dl": {"summary": "...", "model_used": "dl_polished/finetuned_xl", "verified": true, "numeric_accuracy": 1.0}
}
```

#### POST /upload (DL mode)
Form fields:
- file
- use_gpu_ocr
- verify_numbers
- num_beams
- mode (ml, base_dl, finetuned)

#### Other DL utility endpoints
- GET /health
- POST /model/load

## 4) Feature engineering and selection details

### 4.1 Table-level feature extractor (32 dims)

Used by KNN and broader routing signals, includes:
- numeric distribution stats
- percentage stats
- arm structure features
- severity/SAE/discontinuation flags
- completeness and imbalance indicators

### 4.2 Row-level feature extractor (12 dims)

Used by content selection, includes:
- drug/control percentages
- absolute risk difference
- risk direction
- relative risk
- table-relative rank features
- severity/SAE binary flags
- threshold indicators

### 4.3 Content selector logic

Selection policy is layered:
- hard overrides (SAE/severe/high diff)
- LightGBM probability thresholding when model exists
- rule-based fallback when model not loaded

## 5) Verification and quality control

Primary evaluation code: src/evaluation/eval_suite.py

Implemented metrics:
- ROUGE-1/2/L
- numeric accuracy
- hallucination rate

Additional support:
- single-example scoring
- aggregate dataset scoring
- threshold checks
- leave-one-out style evaluation routines

## 6) Why hallucination control is robust

Safety is enforced in multiple layers:

1. Deterministic numeric extraction and templating.
2. Post-generation numeric consistency checks.
3. Arm-attribution consistency checks by column.
4. DL rewrite rejection and fallback to ML when unsafe.

This makes DL an optional enhancer, not a single point of factual failure.

## 7) Current technical risks / watch items

1. Hardware-dependent model behavior can change effective DL quality path.
2. Documentation may reference XXL plans while runtime defaults may use XL/small fallback depending on environment.
3. Adapter availability and versioning should be strictly pinned for reproducibility.
4. Continuous regression tests are needed to ensure no safety drift from fluency tuning.

## 8) Suggested engineering roadmap (next sprint)

1. Lock one deployment matrix and exact model policy per hardware class.
2. Add adapter/version fingerprints in health endpoint payloads.
3. Add automatic compare-mode benchmark pipeline in CI for quality drift.
4. Publish a single source-of-truth runtime architecture document aligned to actual code paths.
