# Clinical Safety Summarization ML System Implementation Guide

Date: 13 March 2026
Project: medicalproject
Scope: Deterministic ML backend and supporting components used for clinical safety table to narrative generation.

---

## 1. What the ML system is

The ML system is a hybrid deterministic pipeline designed for regulatory safety writing.

It combines:

1. Rule-aware parsing and feature engineering.
2. LightGBM-based row selection for individual adverse events.
3. KNN retrieval over engineered table features for style/context adaptation.
4. Template-cluster routing and slot filling for controlled generation.
5. Numeric and arm-attribution verification as hard safety checks.

Primary goal:

- Preserve factual correctness and arm attribution.
- Generate consistent regulatory-style summaries.
- Minimize hallucination risk.

---

## 2. Runtime architecture (ML path)

Main service:

- `src/api/main_ml.py` (FastAPI, port 8000)

Core components wired at startup:

1. `StatisticalFeatureExtractor` (table-level features).
2. `SlotFillGenerator` (main deterministic generator).
3. Optional `KNNRetrievalEngine` loaded from `data/retrieval_index.pkl`.

High-level flow for `/summarize`:

1. Receive `table_text` (linearized table string).
2. Optional nearest-neighbor retrieval from indexed corpus.
3. Deterministic generation through `SlotFillGenerator.generate(...)`.
4. Return summary + verification + metadata.

High-level flow for `/upload`:

1. Receive file (PDF, DOCX, JPEG, PNG, BMP, TIFF).
2. Extract one or more tables.
3. Convert each table to linearized format.
4. Run the same deterministic generation per table.
5. Return result list with per-table diagnostics.

---

## 3. Input contract

Canonical table format (single string):

- `start_table ... end_table`
- `[TABLE_TITLE: ...]`
- `[HEADERS: | arm1 | arm2 | ...]`
- Repeating `[ROW] label | value_arm1 | value_arm2`

Typical cell formats accepted:

1. `N (pct%)`, example `35 (8.5%)`.
2. `pct%`, example `8.5%`.
3. Numeric fallback forms (with reduced confidence in semantics).

---

## 4. What happens when a table arrives (step-by-step)

### Stage A: Arm parsing and column alignment

Component:

- `ArmParser` in `src/generation/slot_fill_generator.py`

Actions:

1. Parse arm labels and N counts from `[HEADERS: ...]`.
2. Infer control vs experimental arm using control keywords.
3. Set `experimental_arm_index` and derive control index.
4. Produce `ArmContext` used by all downstream extraction.

Why this matters:

- Prevents wrong-column percentage attribution.
- Ensures all slot filling and row-level features use the correct arm mapping.

### Stage B: Overview slot extraction

Component:

- `OverviewSlotExtractor` in `src/generation/slot_fill_generator.py`

Actions:

1. Scan rows for clinically significant row types (`TEAE`, `SAE`, `Grade 3-4`, discontinuation, fatal, drug-related variants).
2. Parse per-arm values from each matched row.
3. Populate slot map used by templates.

Output examples:

- `TEAE_N_B`, `TEAE_PCT_B`, `TEAE_N_A`, `TEAE_PCT_A`
- `SAE_PCT_B`, `G34_PCT_A`, `DISC_PCT_B`
- `ARM_A`, `ARM_B`, `N_A`, `N_B`

### Stage C: Individual AE row feature extraction

Component:

- `RowFeatureExtractor` in `src/feature_engineering/row_feature_extractor.py`

Actions:

1. Parse non-overview rows as candidate individual AEs.
2. Extract 12 features per row.
3. Return structured vectors for content selection.

### Stage D: Content selection (LightGBM + overrides)

Component:

- `ContentSelector` in `src/models/content_selector.py`

Actions:

1. Apply regulatory override rules first (always include critical rows).
2. If LightGBM model exists, score each row and compare against threshold.
3. If model does not exist, use rule-based fallback scoring and selection.

### Stage E: Cluster routing and template selection

Components:

- `StatisticalFeatureExtractor` (`src/feature_engineering/statistical_features.py`)
- `select_cluster(...)` and template registry (`src/generation/template_clusters.py`)

Actions:

1. Extract table-level feature vector.
2. Route to the most appropriate template cluster (C01+ family).
3. Identify required slots for that cluster.

### Stage F: Optional KNN adaptation

Component:

- `KNNRetrievalEngine` (`src/retrieval/knn_retrieval_engine.py`)

Actions:

1. Retrieve nearest historical pair based on cosine distance.
2. Attempt de-lexicalization/adaptation of retrieved narrative into slotized template.
3. Fall back to built-in template if adaptation fails.

### Stage G: Surface realization

Component:

- `SlotFillGenerator`

Actions:

1. Fill overview template slots.
2. Render selected individual AE sentences.
3. Merge overview and AE details.
4. Run text polish pass.

### Stage H: Hallucination and attribution verification

Component:

- `HallucinationGuardian` in `src/generation/slot_fill_generator.py`

Actions:

1. Verify generated numbers appear in source table (with tolerance).
2. Verify arm-labeled percentages map to the correct source column.
3. Compute numeric accuracy.
4. Emit warnings.
5. Optionally raise in strict mode.

Final output fields include:

- `summary`, `verified`, `cluster_id`, `cluster_description`, `numeric_accuracy`, `warnings`, `slots_filled`, `inference_time_ms`.

---

## 5. ML techniques and exact parameters

## 5.1 Content selection (LightGBM)

Inference-time selector constants (`src/models/content_selector.py`):

1. `DIFF_THRESHOLD = 5.0` percentage points.
2. `DRUG_THRESHOLD = 5.0` percent incidence.
3. `LGBM_THRESHOLD = 0.40` probability cutoff.

Selection logic:

1. Always include if `is_sae`.
2. Always include if `is_severe`.
3. Always include if `abs_risk_diff >= 5.0`.
4. Else LightGBM probability >= 0.40 includes.
5. If LightGBM is below threshold but `drug_pct >= 5.0`, include as safety net.
6. If no model exists, use rules:
   - Include when `drug_pct >= 5.0` or `abs_risk_diff >= 3.0` or SAE/severe.

Training-time LightGBM params in `ContentSelector.fit(...)`:

1. `n_estimators=200`
2. `max_depth=6`
3. `learning_rate=0.05`
4. `num_leaves=31`
5. `min_child_samples=5`
6. `subsample=0.8`
7. `colsample_bytree=0.8`
8. `random_state=42`
9. `n_jobs=-1`

Training pipeline script:

- `scripts/train_content_selector.py`

Notes:

- Uses `StandardScaler`.
- Uses class-balanced sample weights.
- Labels built by matching event names in reference writeups.

Additional synthetic-data training script in repo:

- `scripts/datasynt.py`

That script demonstrates an alternate approach with:

1. Rule-based synthetic negative generation.
2. SMOTE oversampling.
3. LightGBM training with early stopping.

## 5.2 Row-level feature engineering (12 features)

From `src/feature_engineering/row_feature_extractor.py`:

1. `drug_pct`
2. `ctrl_pct`
3. `abs_risk_diff`
4. `risk_dir`
5. `relative_risk`
6. `drug_pct_rank`
7. `abs_diff_rank`
8. `is_sae`
9. `is_severe`
10. `drug_above_5pct`
11. `drug_above_10pct`
12. `diff_above_5pp`

Rules embedded in parsing:

1. Overview rows are filtered out from individual-AE selection.
2. SAE/severe detection uses keyword/pattern rules.
3. Relative risk capped for stability in zero-denominator scenarios.

## 5.3 Table-level feature engineering (32 features)

From `src/feature_engineering/statistical_features.py`:

Feature groups:

1. Numeric distribution features.
2. Percentage distribution features.
3. Arm-size and cohort features.
4. Safety signal flags (`has_sae`, `has_grade34`, `has_discontinuation`, `has_death`).
5. Structural/table-shape features.
6. Additional clinical signal features:
   - `arm_count`
   - `has_dual_phase`
   - `discontinuation_delta`
   - `grade_3_4_ratio`
   - `sae_drug_related_flag`
   - `data_completeness_score`
   - `footnote_count`
   - `n_subjects_imbalance`

## 5.4 KNN retrieval

From `src/retrieval/knn_retrieval_engine.py`:

1. Distance metric: cosine.
2. Default `k=5`.
3. `NearestNeighbors(..., algorithm='brute')`.
4. Feature scaling: `StandardScaler`.
5. Stores corpus pairs with table text, writeup, and feature vectors.

Used at inference:

1. Retrieve nearest examples.
2. Use best match for optional template adaptation.

## 5.5 Template cluster routing

From `src/generation/template_clusters.py`:

1. Multiple clinically-targeted template clusters (C01+).
2. Cluster chosen from table-level features and flags.
3. Each cluster defines required slots and narrative scaffold.

Technique type:

- Deterministic NLG with controlled variation by cluster.

## 5.6 Verification guard

From `HallucinationGuardian` in `src/generation/slot_fill_generator.py`:

1. Numeric grounding check against source table numbers.
2. Tolerance handling for parse/rounding differences.
3. Arm attribution check for percentage-to-arm consistency.
4. Returns `(verified, numeric_accuracy, warnings)`.

This makes the ML path suitable for high-integrity regulatory contexts.

---

## 6. API behavior and request handling

### `/summarize` (text)

Request:

```json
{
  "table_text": "start_table ... end_table",
  "return_features": false,
  "strict_mode": false
}
```

Response includes:

1. Summary text.
2. Verification and numeric accuracy.
3. Cluster metadata.
4. Warnings and slot diagnostics.
5. Optional 32-feature vector when requested.

### `/upload` (file)

Accepted types:

1. PDF
2. DOCX/DOC
3. JPEG/JPG
4. PNG
5. BMP
6. TIFF

Behavior:

1. Extract tables using PDF/doc/image extractors.
2. Run full deterministic pipeline on each extracted table.
3. Return per-table structured results and total timing.

### Utility endpoints

1. `/health`
2. `/clusters`
3. `/features/names`
4. `/evaluate/single`
5. `/summarize/batch`

---

## 7. Safety and compliance-oriented design choices

1. Deterministic slot filling instead of free-form generation.
2. Mandatory arm-context parsing before extraction.
3. Regulatory overrides in content selection for SAE/severe/fatal patterns.
4. Numeric and arm-attribution verification before accepting output.
5. Warning-rich outputs for reviewer transparency.

---

## 8. Known assumptions and operational notes

1. Arm inference relies on header quality and control-keyword heuristics.
2. If no LightGBM checkpoint is present, selector remains rule-based (still safe, less adaptive).
3. KNN adaptation is optional; system can operate purely on static cluster templates.
4. Uploaded scanned content quality can affect extraction confidence; verification still runs after generation.

---

## 9. Minimal end-to-end pseudocode

```text
receive(table_or_file)
  -> if file: extract tables -> linearized text(s)
  -> for each table_text:
       arm_ctx = parse_arms(headers)
       slots = extract_overview_slots(table_text, arm_ctx)
       row_vectors = extract_row_features(table_text, arm_ctx.experimental_arm_index)
       selected_rows = content_selector.select(row_vectors)
       table_features = statistical_feature_extractor.extract(table_text)
       cluster = select_cluster(table_features)
       template = get_template(cluster)
       if knn_available: template = adapt_from_best_neighbor(...)
       overview = fill_slots(template, slots)
       ae_detail = render_individual_ae(selected_rows, arm_ctx)
       narrative = merge_and_polish(overview, ae_detail)
       verified, accuracy, warnings = guardian.verify(narrative, table_text, arm_ctx)
       return narrative + metadata
```

---

## 10. File map (ML implementation)

1. `src/api/main_ml.py` - ML service orchestration and endpoints.
2. `src/generation/slot_fill_generator.py` - primary deterministic generation pipeline.
3. `src/models/content_selector.py` - LightGBM/rule selector.
4. `src/feature_engineering/row_feature_extractor.py` - row-level 12-feature extraction.
5. `src/feature_engineering/statistical_features.py` - table-level 32-feature extraction.
6. `src/retrieval/knn_retrieval_engine.py` - similarity retrieval layer.
7. `src/generation/template_clusters.py` - cluster templates and routing.
8. `scripts/train_content_selector.py` - selector training pipeline.
9. `scripts/datasynt.py` - synthetic-data and alternate LightGBM training experiment script.

---

## 11. Practical summary

When a table comes in, the system does not immediately write text.

It first decides:

1. Which arm is control vs experimental.
2. Which values belong to each clinical category.
3. Which individual AE rows are important enough to mention.
4. Which narrative structure is appropriate for this table profile.
5. Whether every generated number and arm attribution is valid.

Only after these checks does it return the final summary.
