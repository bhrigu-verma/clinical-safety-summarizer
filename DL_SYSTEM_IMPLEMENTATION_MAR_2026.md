# Deep Learning System Implementation

Date: 13 March 2026

## Purpose

This document explains how the deep-learning summarization stack is implemented today in this repository, how it interacts with the deterministic ML pipeline, what parameters are used in serving and fine-tuning, and what happens when a new table enters the system.

It is based on the current code in:

- `src/api/main_dl.py`
- `src/training/finetune_dl.py`
- `src/data_processing/pdf_extractor.py`
- `src/data_processing/image_extractor.py`
- `src/generation/slot_fill_generator.py`
- `src/models/content_selector.py`
- `src/feature_engineering/row_feature_extractor.py`
- `scripts/benchmark_run.py`
- `scripts/benchmark_10k.py`

## 1. System Summary

The DL system is not a pure end-to-end table-to-text generator in production mode. It is a hybrid architecture:

1. A deterministic ML pipeline first parses the table and produces a numerically grounded summary.
2. The DL layer then rewrites that summary into smoother regulatory prose.
3. A verification gate checks whether the DL rewrite introduced unsupported numbers or wrong arm attribution.
4. If the rewrite is unsafe, the system falls back to the deterministic ML summary.

This means the deployed "fine-tuned" path is primarily a style-polishing layer on top of a hard-structured generator, not a direct free-form generator from raw table input.

## 2. Main Runtime Components

### 2.1 ML backend on port 8000

File: `src/api/main_ml.py`

This backend runs the deterministic 3-stage pipeline:

1. Feature extraction
2. Optional KNN retrieval
3. Slot-fill generation with hallucination verification

It is the source of truth for numerically grounded narrative generation.

### 2.2 DL backend on port 8001

File: `src/api/main_dl.py`

This backend serves three modes:

- `ml`: deterministic ML summary through an adapter wrapper
- `base_dl`: direct transformer summarization from table text
- `finetuned`: ML summary first, then DL rewrite, then hallucination guard

### 2.3 Frontend summarizer

File: `frontend/app/summarizer/page.tsx`

The frontend exposes four user-facing options:

- `ML`
- `DL Base`
- `DL Fine-Tuned`
- `Compare`

It expects:

- ML backend on `localhost:8000`
- DL backend on `localhost:8001`

## 3. DL Runtime Modes

## 3.1 `ml` mode

This is effectively the deterministic generator exposed through the DL service.

Implementation:

- `main_dl.py` creates `MLAdapter`
- `MLAdapter` wraps `SlotFillGenerator(strict_mode=False)`
- `generate(table_text)` returns the deterministic narrative

Used when:

- user explicitly requests `ml`
- DL model is unavailable and fallback occurs
- fine-tuned rewrite is blocked by verification

Output label:

- `ml/lgbm+jinja2`

## 3.2 `base_dl` mode

This is a direct transformer generation path.

Prompt pattern:

- role: medical writer for regulatory submission
- input: raw linearized safety table
- task: summarize TEAE, Grade 3-4, SAE, and discontinuation statistics

Generation path:

1. Build `DIRECT_PROMPT`
2. Tokenize prompt
3. Run `model.generate(...)`
4. Decode to narrative
5. Verify numbers against the source table

Output label:

- `dl_base/flan-t5`

If generation fails, runtime falls back to the ML engine.

## 3.3 `finetuned` mode

This is the production hybrid path.

Pipeline:

1. Generate strict deterministic summary with ML
2. Build `REWRITE_PROMPT`
3. Send the ML paragraph to the fine-tuned LoRA model, or base model if adapter is unavailable
4. Verify the rewritten summary against the original table
5. Accept rewrite only if:
   - verification passes
   - numeric accuracy is at least `0.95`
6. Otherwise revert to the ML summary

Accepted output label:

- `dl_polished/finetuned_xl` when adapter is loaded
- `dl_polished/base_polish` when rewrite uses base model only

Fallback output label:

- `ml_fallback`

This design is central to the current system behavior. The fine-tuned model is not trusted to generate raw summaries directly in production mode.

## 4. Model Loading and Serving Parameters

## 4.1 Model identifiers

Declared in `src/api/main_dl.py`:

- `ADAPTER_PATH = models/best_clinical_adapter_xl`
- `BASE_MODEL_ID = google/flan-t5-xl`
- `FALLBACK_MODEL_ID = google/flan-t5-small`

## 4.2 Device behavior

Runtime chooses device in this order:

1. `cuda`
2. `mps`
3. `cpu`

Behavior by device:

- On `cuda`: loads `google/flan-t5-xl` and quantizes in 4-bit using bitsandbytes
- On `mps` or `cpu`: loads `google/flan-t5-small` in FP32 as the base model
- On `mps` or `cpu`: skips loading the full XL base + LoRA adapter to avoid out-of-memory failures

Operational consequence:

- Full intended DL behavior exists on NVIDIA GPU
- Local CPU or Apple Silicon testing uses a reduced-capacity base model and simulates fine-tuned mode by polishing with the smaller base model or falling back to ML

## 4.3 Quantization parameters for serving

Used when `device == cuda`:

- `load_in_4bit = True`
- `bnb_4bit_use_double_quant = True`
- `bnb_4bit_quant_type = nf4`
- `bnb_4bit_compute_dtype = torch.bfloat16`

## 4.4 Inference request parameters

Request schema in `main_dl.py`:

- `table_text: str`
- `max_new_tokens: int = 512`
- `num_beams: int = 4`
- `temperature: float = 0.0`
- `verify_numbers: bool = True`
- `mode: str = finetuned`

Current generation kwargs:

- `max_new_tokens = request value`
- `num_beams = request value`
- `early_stopping = True`
- `no_repeat_ngram_size = 3`
- `length_penalty = 1.5`
- `do_sample = False` when `temperature == 0`
- `do_sample = True` when `temperature > 0`

Tokenization parameters:

- `max_length = 2048`
- `truncation = True`
- `padding = True`

## 4.5 Verification thresholds

DL output is checked by `_verify_output(...)` in `main_dl.py`.

Verification logic:

- extract all numbers from source table
- extract all numbers from generated narrative
- ignore table reference integers and tiny context integers (`0`, `1`, `2`)
- mark any generated number as suspicious if it is not present in the source table within tolerance

Tolerance rules:

- serving verification uses approximately `<= 1.01` absolute tolerance in `main_dl.py`
- benchmark scripts use tighter thresholds for hallucination scoring, typically `<= 0.11`

Fine-tuned rewrite acceptance rule:

- `verified == True`
- `numeric_accuracy >= 0.95`

If either condition fails, the DL rewrite is rejected.

## 5. Fine-Tuning Implementation

File: `src/training/finetune_dl.py`

## 5.1 Training objective

The training script fine-tunes `google/flan-t5-xl` on table-to-summary pairs so that the model learns:

- regulatory clinical tone
- better sentence flow
- domain-specific phrasing
- safer structured rewriting around safety statistics

## 5.2 Training data

Configured training input:

- `DATA_PATH = data/augmented/synthetic_1000.jsonl`

Training examples are normalized into:

- `table_text`
- `writeup`

Prompt format during training:

- a system-style instruction
- the raw safety table
- a `Clinical Narrative:` prefix

That means the model is trained as a table-to-text summarizer, even though the production `finetuned` path currently uses it mostly as a rewriter.

## 5.3 Training hyperparameters in code

From `src/training/finetune_dl.py`:

- `BASE_MODEL = google/flan-t5-xl`
- `OUTPUT_DIR = models/flan_t5_xl_clinical`
- `MAX_INPUT_LEN = 2048`
- `MAX_TARGET_LEN = 512`
- `BATCH_SIZE = 4`
- `GRAD_ACCUM = 8`
- `effective batch size = 32`
- `EPOCHS = 20`
- `LR = 2e-4`
- `WARMUP_RATIO = 0.10`
- `SAVE_TOTAL = 3`
- scheduler: `cosine`
- `gradient_checkpointing = True`
- optimizer: `paged_adamw_32bit`
- `bf16 = True` on CUDA
- `fp16 = False`
- early stopping patience: `5`

## 5.4 QLoRA configuration in training code

Training script LoRA settings:

- `LORA_R = 32`
- `LORA_ALPHA = 64`
- `LORA_DROPOUT = 0.05`
- `LORA_TARGETS = [q, k, v, o, wi_0, wi_1, wo]`
- `task_type = SEQ_2_SEQ_LM`
- `bias = none`

## 5.5 Actual adapter artifact currently stored

The deployed adapter config in `models/best_clinical_adapter_xl/adapter_config.json` differs from the defaults in the training script.

Observed deployed adapter settings:

- `base_model_name_or_path = google/flan-t5-xl`
- `r = 64`
- `lora_alpha = 128`
- `lora_dropout = 0.05`
- `target_modules = [k, lm_head, o, wo, v, wi_0, q, wi_1]`
- `modules_to_save = [lm_head]`
- `inference_mode = true`
- `peft_version = 0.18.1`

Important note:

The serving artifact is stronger than the visible default training script configuration. That implies either:

1. the final adapter was trained with a later configuration than `finetune_dl.py`, or
2. the script was not updated after the final training run.

For future reproducibility, this discrepancy should be treated as an implementation note, not ignored.

## 6. What Happens When a Table Comes In

This depends on how the table enters the system.

## 6.1 Case A: user pastes a linearized table

Example entry point:

- frontend summarizer text box
- POST to `/summarize`

Flow:

1. Frontend sends `table_text`
2. Backend chooses mode
3. If mode is `ml`, deterministic summary is produced directly
4. If mode is `base_dl`, raw table is summarized by transformer
5. If mode is `finetuned`, deterministic summary is generated first, then rewritten by DL
6. Verification runs on final output
7. JSON response is returned with summary, timing, numeric accuracy, warnings, and model label

## 6.2 Case B: user uploads PDF or DOCX

Entry point:

- `/upload` endpoint on ML or DL backend

Flow on DL backend:

1. File type is identified from extension or content type
2. File is written to a temporary file
3. `ClinicalPDFExtractor` processes the file
4. For each extracted table pair:
   - use `pair.table_text` as model input
   - summarize according to selected mode
   - package result as `TableResult`
5. Temporary file is deleted
6. Backend returns all tables found plus one summary per table

PDF extraction behavior:

- text-based PDFs use `pdfplumber`
- scanned PDFs fall back to `PyMuPDF` page rendering plus image OCR extraction
- DOCX tables are parsed via `python-docx`

## 6.3 Case C: user uploads image, screenshot, or scanned table

Entry point:

- `/upload` endpoint with image content

Flow:

1. `ClinicalImageExtractor` loads image
2. image pre-processing runs:
   - grayscale conversion
   - upscaling when needed
   - denoising
   - deskew
   - adaptive thresholding
3. primary extraction uses `img2table` with EasyOCR
4. fallback extraction uses raw EasyOCR plus heuristic row grouping
5. extracted table becomes:
   - HTML
   - JSON row data
   - linearized string for summarization
6. summarization runs on the linearized form
7. response returns both table structure and narrative output

## 7. Table Linearization Format

All summarization logic expects a normalized string format like:

```text
start_table [TABLE_TITLE: Table 1: Overview of Treatment-Emergent Adverse Events]
[HEADERS: | Event | Drug Arm (N=120) | Placebo (N=118)]
[ROW] Any TEAE | 110 (91.7) | 96 (81.4)
[ROW] Grade 3-4 TEAE | 28 (23.3) | 16 (13.6)
[ROW] SAE | 14 (11.7) | 9 (7.6)
[ROW] Discontinuation due to AE | 10 (8.3) | 4 (3.4)
end_table
```

This representation is the central interface between extraction, ML generation, and DL serving.

## 8. How the Deterministic ML Layer Parses the Table

Even though this document focuses on DL, the ML layer is structurally required for the fine-tuned path.

## 8.1 Arm parsing

`SlotFillGenerator` uses `ArmParser` to read the `HEADERS` token and determine:

- actual arm names
- sample sizes `N`
- which arm is experimental
- which arm is control/comparator

Control arm detection is keyword driven, using labels such as:

- placebo
- standard of care
- observation
- chemotherapy
- comparator
- control

Default heuristic:

- if no strong signal exists, the rightmost column is treated as experimental

## 8.2 Overview slot extraction

The generator scans `[ROW]` entries and maps row labels into semantic slots such as:

- `TEAE_*`
- `G34_*`
- `SAE_*`
- `DISC_*`
- drug-related variants
- fatal event variants

It extracts both counts and percentages per arm.

## 8.3 Individual adverse-event selection

For rows that represent named adverse events rather than overall rollups:

1. `RowFeatureExtractor` creates 12 features per row
2. `ContentSelector` decides whether the event should be mentioned
3. selected rows are rendered into explicit comparative sentences

Row-level features include:

- drug incidence percent
- control incidence percent
- absolute risk difference
- risk direction
- relative risk
- within-table rank features
- severity flags
- SAE flags
- thresholds such as `drug_above_5pct`

Selection rules:

- always include SAE/fatal rows
- always include severe rows
- always include large absolute differences
- optionally include rows selected by LightGBM probability

## 8.4 Hallucination guard in ML generation

The ML generator also uses a stricter verification layer that checks:

- unsupported numbers
- incorrect arm attribution for percentages

This is why the ML narrative is used as the safe base paragraph for the fine-tuned mode.

## 9. How the Fine-Tuned DL Layer Uses the ML Output

In production mode, the fine-tuned model sees a paragraph, not the raw table.

The rewrite prompt says, in effect:

- rewrite for regulatory tone
- improve flow
- do not add, remove, or change any numbers or clinical events

The intended behavior is:

- preserve content
- improve prose quality
- avoid factual drift

The actual enforcement comes from the verification gate after generation.

This design is conservative by intention. It trades some generative freedom for numerical safety.

## 10. Upload and Extraction Details

## 10.1 PDF and DOCX extractor behavior

`ClinicalPDFExtractor` supports three paths:

- `.pdf` through `pdfplumber`
- scanned `.pdf` through `PyMuPDF` plus OCR image extraction
- `.docx` through `python-docx`

For text PDFs:

- tables are found by `page.extract_tables()`
- tables are linearized into the common format
- nearby narrative may be extracted if present

For scanned PDFs:

- each page is rendered to PNG at roughly 200 DPI
- the rendered page is passed to `ClinicalImageExtractor`
- writeup is usually unavailable, but table text still becomes usable model input

## 10.2 Image extractor behavior

Primary extraction stack:

- OpenCV pre-processing
- EasyOCR OCR engine
- img2table structure reconstruction

Fallback stack:

- raw EasyOCR bounding boxes
- heuristic grouping into rows by Y-coordinate proximity
- DataFrame reconstruction

Outputs per extracted table:

- `table_id`
- `html`
- `json_data`
- `headers`
- `linearized`
- `confidence`
- `n_rows`
- `n_cols`

## 11. Evaluation and Benchmarking

## 11.1 Benchmark datasets

The repo uses two explicit validation tiers:

- `data/benchmark/tier1_gold.jsonl`: table plus human reference writeup
- `data/benchmark/tier2_stress.jsonl`: table only

There is also a large-scale benchmark file:

- `clinical_10k_final.jsonl`

## 11.2 Benchmark scripts

Main scripts:

- `scripts/benchmark_extract.py`: builds tiered benchmark datasets from source PDFs
- `scripts/benchmark_run.py`: runs ML, DL base, and DL fine-tuned on benchmark tiers
- `scripts/benchmark_report.py`: aggregates metrics and writes report outputs
- `scripts/benchmark_10k.py`: runs large-scale evaluation on the full 10k-style dataset

## 11.3 Metrics used

Per-table metrics include:

- hallucination rate
- numeric accuracy versus source or reference
- ROUGE-1
- ROUGE-2
- ROUGE-L
- inference time
- verification pass rate

Roadmap checks in `benchmark_report.py` include thresholds such as:

- `p10 ROUGE-1 > 0.75`
- `mean ROUGE-1 > 0.85`
- `mean numeric accuracy > 0.96`
- `mean hallucination < 4%`

## 12. Current Implementation Notes and Caveats

## 12.1 The deployed system is hybrid-first, not DL-first

The most important implementation truth is this:

- the stable path is deterministic ML generation first
- DL is applied as a controlled rewrite layer

This is why the system achieves better safety than a pure transformer summarizer.

## 12.2 CPU and Apple Silicon behavior is not equivalent to GPU behavior

On non-CUDA devices:

- base model becomes `flan-t5-small`
- LoRA adapter over XL is skipped
- "fine-tuned" mode may effectively be base-polish or ML fallback

So local development behavior is not a faithful representation of the intended H100-class deployment behavior.

## 12.3 Training script and deployed adapter are not perfectly aligned

The checked-in training script says LoRA rank 32 and alpha 64.
The deployed adapter artifact says rank 64 and alpha 128, and also includes `lm_head` handling.

That discrepancy should be resolved if exact reproducibility matters.

## 12.4 Frontend single-mode DL calls should be treated carefully

The frontend currently appends `mode` in the query string for single DL runs, while the FastAPI handler reads `mode` from the request body model. The compare path sends the correct JSON body structure.

In other words, the compare path is aligned with backend expectations, but single-run DL mode selection should be verified against live behavior whenever debugging frontend/backend mismatches.

## 13. End-to-End Flow in One View

### Raw text table

1. User pastes linearized table
2. Frontend sends `table_text`
3. Backend chooses mode
4. ML summary is generated directly, or transformer runs, or hybrid rewrite runs
5. Verification checks numbers
6. Summary and metadata are returned

### Uploaded PDF/DOCX/image

1. User uploads file
2. Extractor converts detected tables into normalized linearized strings
3. Each table is summarized independently
4. Optional HTML/JSON table structure is returned for rendering
5. Final summaries are returned with timing and verification metadata

## 14. Bottom Line

The DL system in this repo is best understood as a controlled language layer on top of a structured clinical summarization engine.

The real architecture is:

- extraction normalizes tables
- deterministic logic anchors the facts
- LoRA-enhanced Flan-T5 improves fluency
- hallucination checks keep the final output safe

That design choice is why the system is workable for clinical safety narratives, where a fluent sentence is useful, but unsupported numbers are unacceptable.