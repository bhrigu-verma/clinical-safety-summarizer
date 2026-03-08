# 📋 Clinical Safety Table Summarization — Complete Project Specification

**Document Purpose:** Autonomous AI Research Agent Handover Document  
**Version:** 1.0 (March 2026)  
**Project Root:** `/Users/bhriguverma/medicalproject`  
**Primary Architecture:** Hybrid Deterministic ML Pipeline + QLoRA Fine-Tuned DL Polish Layer  

---

## 1. Project Objective

This system converts structured **clinical safety tables** (adverse event summaries from pharmaceutical clinical trial reports — specifically Pfizer CSRs) into **natural language summaries** for regulatory submission. The primary target document type is the **"Overview of Treatment-Emergent Adverse Events"** table found in ICH E3-format Clinical Study Reports (CSRs).

**Core Design Philosophy:**  
Standard LLMs (GPT-4, T5 out-of-the-box) are unsuitable because they hallucinate numbers. In regulatory medical writing, **0.0% hallucination rate** is non-negotiable. The system achieves this through a **deterministic ML pipeline** that mathematically guarantees every number in the output exists in the source table, with an optional **LLM polish layer** (Flan-T5-XXL via QLoRA) gated by a hard-coded hallucination guardian before output is released.

**Regulatory Context:** Summaries must comply with **ICH E3 guidelines**, use passive voice, report the drug arm before the control arm, and use standardized clinical terminology (TEAE, SAE, Grade 3–4, etc.).

---

## 2. Data Architecture

### 2.1 Raw Data Sources

| Source | Count | Location |
|--------|-------|----------|
| Pfizer CSR Scrape (authentic) | ~148 CSR documents | `pfizer_csrs/` |
| Gold extracted table-writeup pairs | ~172–191 pairs | `data/processed/raw_pairs.json` |
| KNN-specific gold pairs | variable | `data/processed/knn_pairs.json` |
| Pfizer gold pairs (JSONL) | variable | `data/pfizer_gold_pairs.jsonl` |
| Augmented synthetic pairs | ~437 (expanded to 1411) | `data/augmented/*.jsonl` |

The original ground-truth corpus was extracted from authentic Pfizer CSR documents using `pfizzer_scraper.py` and the PDF extractor (`src/data_processing/pdf_extractor.py`), yielding ~200 initial table-writeup pairs.

### 2.2 Input Format — Linearized Table String

All tables are converted into a single-line **linearized string** format before any ML/DL processing. This is the canonical input format for the entire pipeline.

**Format Specification:**
```
start_table [TABLE_TITLE: Table X — Title of the Table] [HEADERS: | Arm A Label (N=XXX) | Arm B Label (N=XXX) | ] [ROW] Row Label | N_a (pct_a%) | N_b (pct_b%) [ROW] Row Label | N_a (pct_a%) | N_b (pct_b%) end_table
```

**Critical Tokens:**
- `start_table` / `end_table` — bounding delimiters
- `[TABLE_TITLE: ...]` — extracted table caption
- `[HEADERS: | col1 | col2 | ...]` — pipe-separated column headers **including arm names and N= counts**
- `[ROW]` — each data row separator; cells are pipe-separated
- Cell format: `N (pct%)` e.g., `75 (92.6%)`

**Arm Name Convention:**
- `ARM_A` = **control arm** (placebo, chemotherapy, standard of care, etc.)
- `ARM_B` = **experimental arm** (drug under investigation)
- The `ArmParser` class identifies which is which by scanning headers for `CONTROL_KEYWORDS` (placebo, vehicle, docetaxel, carboplatin, cisplatin, etc.)

### 2.3 JSONL Pair Format

All training and evaluation pairs are stored as:
```json
{
  "table_text": "start_table [TABLE_TITLE: ...] [HEADERS: ...] [ROW] ... end_table",
  "writeup": "Treatment-emergent adverse events (TEAEs) were reported for ...",
  "source": "gold | pfizer | synthetic"
}
```

> **Key:** Some older files use `input`/`output` keys instead of `table_text`/`writeup`. All loaders must handle both key naming conventions.

### 2.4 Output Format — Clinical Narrative

The output is a **single paragraph** of 2–5 sentences in ICH E3 regulatory style. Example:

> "Treatment-emergent adverse events (TEAEs) were reported for 75 (92.6%) of 81 subjects in the Pembrolizumab arm and 68 (84.0%) of 81 subjects in the Chemotherapy arm. Grade 3–4 TEAEs were reported for 42.0% of subjects in the Pembrolizumab arm and 38.3% of subjects in the Chemotherapy arm. Treatment-emergent SAEs were reported for 18 (22.2%) of subjects in the Pembrolizumab arm and 15 (18.5%) of subjects in the Chemotherapy arm. TEAEs leading to treatment discontinuation were reported for 12 (14.8%) of subjects in the Pembrolizumab arm and 5 (6.2%) in the Chemotherapy arm."

**Output Constraints:**
- Every number in the output **must** exist in the source table (within ±0.11 floating-point tolerance)
- Arm labels **must** come from actual table headers, not hardcoded strings
- Numbers attributed to an arm **must** come from that arm's column
- Small integers (0, 1, 2, 3, 4, 5) are exempt from hallucination checking
- Table references (e.g., "Table 15") are also exempt

---

## 3. Model Pipeline

The system has **two fully operational parallel pipelines** both served via FastAPI:

```
Input (Linearized Table)
         │
         ├─── ML BACKEND (Port 8000) ─────────────────────────────────────────────┐
         │         │                                                               │
         │    Stage 0: ArmParser (CONTROL_KEYWORDS)                                │
         │         │                                                               │
         │    Stage 1: Content Selection (LightGBM Binary Classifier)              │
         │         │   - 6 features per AE row                                    │
         │         │   - Regulatory override: is_sae=1 or is_severe=1 → forced in │
         │         │                                                               │
         │    Stage 2: Clinical Microplanning (Agglomerative Clustering)           │
         │         │   - 'ward' linkage, distance_threshold=4.0                   │
         │         │   - Groups AEs into sentence clusters                        │
         │         │                                                               │
         │    Stage 3: Surface Realization                                         │
         │         ├── KNN Retrieval Engine (k=5, cosine similarity)              │
         │         │   - 172–191 pairs in index                                   │
         │         │   - Retrieves most similar historical writeup                │
         │         │   - De-lexicalizes retrieved text → adapted template         │
         │         │                                                               │
         │         ├── Template Cluster Router (select_cluster)                   │
         │         │   - 14 clusters (C01–C14)                                    │
         │         │   - Routes based on has_grade34, has_sae, has_disc, etc.     │
         │         │                                                               │
         │         └── SlotFillGenerator                                           │
         │             - OverviewSlotExtractor → fills slots from table           │
         │             - IndividualAERenderer → generates per-AE sentences        │
         │             - HallucinationGuardian v4 → verifies output              │
         │             - Polish pass (remove empty parens, fix whitespace)         │
         │                                                                         │
         └── DL BACKEND (Port 8001) ──────────────────────────────────────────────┘
                   │
              Flan-T5-XXL (11B params, 4-bit NF4 QLoRA)
                   │
              LoRA Adapter: models/best_clinical_adapter_xxl/
                   │
              Prompt: "Rewrite this clinical paragraph in regulatory medical tone
                       following ICH E3 guidelines. Preserve ALL numbers exactly.
                       Use passive voice. Report drug arm before placebo."
                   │
              HallucinationGuardian gate → if fail → fallback to ML output
```

### 3.1 Stage 0 — Arm Parser

**File:** `src/generation/slot_fill_generator.py` → class `ArmParser`

Parses `[HEADERS: ...]` token. Extracts arm labels and N= values. Identifies the control arm using a keyword list:
```python
CONTROL_KEYWORDS = [
    'placebo', 'vehicle', 'soc', 'standard of care', 'standard care',
    'best supportive', 'observation', 'watchful waiting',
    'chemotherapy', 'chemo', 'docetaxel', 'paclitaxel', 'cabazitaxel',
    'carboplatin', 'cisplatin', 'gemcitabine', 'fluorouracil',
    'control', 'comparator', 'reference arm',
]
```
If neither arm matches, defaults to `experimental_arm_index=1` (rightmost column). This fixed a critical historical bug (v3→v4) where Chemotherapy was being mislabeled as experimental.

Returns `ArmContext` dataclass with `arm_a_label`, `arm_b_label`, `n_a`, `n_b`, `experimental_arm_index`.

### 3.2 Stage 1 — Content Selection (LightGBM)

**File:** `scripts/datasynt.py`, `scripts/train_content_selector.py`, `src/models/content_selector.py`  
**Trained Model Artifact:** `models/content_selector.pkl`

**6 Input Features per AE Row:**
| Feature | Description |
|---------|-------------|
| `drug_pct` | Incidence % in experimental arm |
| `placebo_pct` | Incidence % in control arm |
| `risk_diff` | Absolute difference (drug_pct - placebo_pct) |
| `relative_risk` | drug_pct / (placebo_pct + 0.001) |
| `is_sae` | Boolean: row contains "SAE", "serious adverse", or "fatal"/"death" |
| `is_severe` | Boolean: row contains "Grade 3", "Grade 4", or "severe" |

**Binary label assignment (during training):**
- Label `1` = event name appears in human-written reference writeup (fuzzy string match)
- Label `0` = event not mentioned in reference writeup

**LightGBM Training Hyperparameters:**
```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'verbose': -1,
    'early_stopping_rounds': 15,
    'num_boost_round': 150
}
```

**Class Imbalance Handling:**
1. **Rule-Based Synthetic Generator** (`generate_synthetic_insignificant_events`): Creates 900 synthetic Label=0 events (low incidence, risk_diff < 2%, never SAE/severe) to balance the ~950 Label=1 real events
2. **SMOTE**: Applied after rule-based generation to smooth decision boundaries (`sampling_strategy='auto'`, `random_state=42`)

**Regulatory Override (Hard Rule):** Events with `is_sae=1` OR `is_severe=1` **bypass** the LightGBM probability threshold and are **force-included** in the summary regardless of their probability score.

**Decision Threshold:** `probability > 0.5` = include event.  
> **Open Question:** Should this be tuned to 0.3 (recall-focused, more events included) or 0.7 (precision-focused, fewer events)?

### 3.3 Stage 2 — Clinical Microplanning (Agglomerative Clustering)

**File:** `scripts/stage2.py`

Groups the LightGBM-selected AE rows into sentence-level clusters using unsupervised learning.

**Algorithm:** `sklearn.cluster.AgglomerativeClustering`
- `linkage='ward'` (minimizes within-cluster variance)
- `n_clusters=None`
- `distance_threshold=4.0`

**Logic:** Events with similar `drug_pct`, `placebo_pct`, `risk_diff`, `relative_risk`, `is_sae`, `is_severe` are grouped together. Mild high-frequency events (Nausea 12.5% vs 4.1%) cluster separately from rare severe events (Cardiac Arrest 0.5% vs 0.1%, is_sae=1).

> **Open Question:** Is `distance_threshold=4.0` optimal? Should it be adjusted based on medical writer feedback?

### 3.4 Stage 3 — Surface Realization

**Files:** `src/generation/slot_fill_generator.py`, `src/generation/template_clusters.py`, `src/retrieval/knn_retrieval_engine.py`

#### 3.4.1 KNN Retrieval Engine

**File:** `src/retrieval/knn_retrieval_engine.py` → class `KNNRetrievalEngine`  
**Index:** `data/retrieval_index.pkl` (serialized with pickle)

- Encodes all corpus pairs as 32-dimensional clinical feature vectors (from `StatisticalFeatureExtractor`)
- Normalizes with `StandardScaler`
- K=5, cosine distance, brute-force exact search
- At query time: finds the single most similar historical table (`retrieve_best`)
- The retrieved historical writeup is **de-lexicalized**: drug names replaced with `{ARM_A}/{ARM_B}`, numeric values replaced with slot names like `{TEAE_PCT_B}`, to become a **reusable template**
- If de-lexicalization fails, falls back to hard-coded template clusters

#### 3.4.2 Template Cluster System

**File:** `src/generation/template_clusters.py`  
**14 Clusters (C01–C14):**

| ID | Description | Key Trigger |
|----|-------------|-------------|
| C01 | Standard 2-arm overview | has_grade34 + has_sae + has_disc |
| C02 | Drug-related emphasis | sae_drug_related_flag |
| C03 | High incidence (>95%) both arms | has_high_teae |
| C04 | SAE and fatal emphasis | has_death + has_sae |
| C05 | No Grade 3-4, SAE only | not has_grade34 + has_sae |
| C06 | Open-label extension | has_dual_phase |
| C07 | Minimal (TEAE + DISC only) | not has_grade34 + not has_sae |
| C08 | Drug-related full profile | sae_drug_related_flag + has_disc |
| C09 | Safety population note | fallback/default |
| C10 | TEAE-only table | not has_sae + not has_grade34 + not has_disc |
| C11 | Grade 3-4, no discontinuation | has_grade34 + has_sae + not has_disc |
| C12 | High discontinuation rate (>10% delta) | disc_delta > 10 |
| C13 | Single-arm study | arm_count = 1 |
| C14 | Individual AE listing | num_rows ≥ 4 + not has_sae + not has_disc |

Template slots use `{SLOT_NAME}` format. Example slots: `{TEAE_N_B}`, `{TEAE_PCT_B}`, `{ARM_B}`, `{N_B}`, `{G34_PCT_A}`, `{SAE_N_A}`, `{DISC_PCT_B}`, `{FATAL_PCT_A}`, `{DR_TEAE_PCT_B}`, `{DR_SAE_PCT_A}`, etc.

**All slots embed the arm label inline** (e.g., "... of subjects in the `{ARM_B}` arm") — this was a critical v4 fix to eliminate wrong-arm attribution.

#### 3.4.3 Slot Extractor

**Class:** `OverviewSlotExtractor`

Priority-ordered regex row classification (`ROW_PATTERNS`):
1. `disc_drug` → drug-related discontinuation → `DR_DISC`
2. `grade34_drug` → drug-related Grade 3-4 → `DR_G34`
3. `sae_drug` → drug-related SAE → `DR_SAE`
4. `teae_drug` → drug-related TEAE → `DR_TEAE`
5. `sae_grade34` → SAE + Grade 3-4 combined → `SAE_G34`
6. `grade34` → Grade 3-4 → `G34`
7. `disc` → discontinuation → `DISC`
8. `sae` → SAE → `SAE`
9. `fatal` → fatal/death → `FATAL`
10. `teae` → overall TEAE → `TEAE`

The `experimental_arm_index` from `ArmParser` controls which column maps to `_B` (drug) and which maps to `_A` (control) — this is the **core fix** from v3→v4 that eliminated the `95.0% → 92.0%` percentage swap bug.

#### 3.4.4 Hallucination Guardian v4

**Class:** `HallucinationGuardian`

Two-phase verification:
1. **Numeric Check:** Every number in the narrative must match a number in the source table within ±0.11 floating-point tolerance. Exemptions: `{0, 1, 2, 3, 4, 5}` and table reference integers.
2. **Arm Attribution Check (`_check_arm_attribution`):** Parses `"{pct}% of subjects in the {arm_name} arm"` patterns. Verifies the percentage value actually exists in that arm's column in the source table (tolerance ±0.15%). Reports misattribution as warnings.

Returns: `(verified: bool, numeric_accuracy: float, warnings: List[str])`

`verified=False` in non-strict mode generates warnings but still returns the narrative. In `strict_mode=True`, it raises `HallucinationError`.

### 3.5 DL Path — Flan-T5-XXL QLoRA

**Script:** `notebooks/02_train_t5xxl_optimized.py`  
**Base Model:** `google/flan-t5-xxl` (11B parameters)  
**Adapter Output:** `models/best_clinical_adapter_xxl/`  
**Hardware Target:** NVIDIA H100 (35GB VRAM cap, uses ~15.4GB)

**QLoRA Configuration:**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16   # H100 native
)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,                     # Down from 64: research shows 16 better for narrow tasks
    lora_alpha=32,            # 2x rank standard scaling
    lora_dropout=0.05,
    target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],  # All linear layers
    bias="none",
    modules_to_save=["lm_head"],  # Train output head
)
```

**Training Hyperparameters:**
```python
num_epochs = 12
per_device_train_batch_size = 2
gradient_accumulation_steps = 8    # Effective batch = 16
learning_rate = 5e-5               # Lower LR needed with r=16
warmup_ratio = 0.06
weight_decay = 0.01
lr_scheduler_type = "cosine"
bf16 = True                        # H100 must use BF16, not FP16
gradient_checkpointing = True
optim = "adamw_bnb_8bit"          # 8-bit Adam, saves ~50% optimizer VRAM
max_grad_norm = 1.0
max_source_len = 512
max_target_len = 384
```

**Evaluation during training:**
- Strategy: every 200 steps
- `metric_for_best_model = "rougeL"` (ROUGE-L chosen over ROUGE-1 per technical research report)
- `num_beams_eval = 4` (beam search during evaluation)

**Input Prompt Format:**
```
Rewrite this clinical paragraph in regulatory medical tone following ICH E3 guidelines. 
Preserve ALL numbers exactly. Use passive voice. Report drug arm before placebo.

{table_text or ML-generated draft}

Rewritten Narrative:
```

**Template Augmentation (Anti-Leakage):**
- 20% of training pairs have their source text sentence order randomly shuffled
- Intent: prevent model from memorizing template structure instead of learning clinical writing style

**Data Sources for DL Training:**
1. Gold pairs (`data/processed/raw_pairs.json`, `data/processed/knn_pairs.json`)
2. Pfizer extracted pairs (`data/pfizer_gold_pairs.jsonl`)
3. Synthetic augmented pairs (`data/augmented/*.jsonl`)
- 90/10 train/eval split via `dataset.train_test_split(test_size=0.1, seed=42)`
- MD5-based deduplication applied before splitting

**Also Exists (XL Adapter):** A smaller `google/flan-t5-xl` (3B params) adapter at `models/best_clinical_adapter/` was trained in an earlier iteration using batch_size=8, 15 epochs, LR=2e-4, r=32, alpha=64. The XXL 11B version supersedes it.

---

## 4. Evaluation Metrics

> ⚠️ **This is the most critical section.** The evaluation system is defined in `src/evaluation/eval_suite.py`.

### 4.1 Triple Metric Suite

The system evaluates on **three simultaneous dimensions** as required by Roadmap v2.0:

#### Metric 1: ROUGE Scores (Text Quality)

**Variants computed:** ROUGE-1, ROUGE-2, ROUGE-L  
**Library:** `evaluate.load("rouge")` (HuggingFace Evaluate)  
**Stemmer:** `use_stemmer=True` (Porter stemmer applied)

**Pre-processing:** References are "scrubbed" before comparison:
```python
def scrub_reference(text):
    # Remove non-predictable artifacts:
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'Document\s+ID:\s+[\w\d\-]+', '', text)
    text = re.sub(r'Confidential', '', text)
    return re.sub(r'\s+', ' ', text).strip()
```

#### Metric 2: Numeric Accuracy (Clinical Safety Metric)

Measures **what fraction of numbers in the ground-truth reference appear in the model's prediction**.

```python
def compute_numeric_accuracy(prediction, reference):
    ref_nums  = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', reference)]
    pred_nums = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', prediction)]
    correct = 0
    for r in ref_nums:
        if any(abs(r - p) <= 1.01 for p in pred_nums):   # ±1.0 rounding tolerance
            correct += 1
    return correct / len(ref_nums)
```

> **Key:** Tolerance is ±1.01 to account for rounding (e.g., reference says 92.6%, model outputs 93%). This is a **precision vs recall** tradeoff decision — the tolerance is intentionally generous to avoid penalizing valid rounding.

#### Metric 3: Hallucination Rate (Safety/Regulatory Metric)

Measures **what fraction of numbers in the prediction do NOT exist in the source table**.

```python
def compute_hallucination_rate(prediction, source_table):
    source_nums = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', source_table)]
    pred_raw = re.findall(r'\b\d+\.?\d*\b', prediction)
    # Exempt: Table X references and tiny context integers {0, 1, 2}
    bad = [p for p in pred_raw if not any(abs(float(p)-s) <= 0.01 for s in source_nums)
           and p not in tbl_refs and p not in allowed_context]
    return len(bad) / len(pred_raw)
```

> **Current Status (ML Path):** Hallucination Rate = **0.0%** (by deterministic design).  
> Target for DL path: < 4%.

### 4.2 Target Thresholds (Roadmap v2.0 Hard Requirements)

| Metric | Threshold | Status |
|--------|-----------|--------|
| P10 ROUGE-1 | > 0.75 | Aspirational |
| Mean ROUGE-1 | > 0.85 | Target for DL fine-tuned |
| Mean Numeric Accuracy | > 0.96 | ✅ Achieved by ML path |
| Hallucination Rate | < 0.04 | ✅ 0.0% achieved by ML path |
| **Baseline ML ROUGE-1** | **~0.27 (template-only)** | Confirmed baseline |
| **Phase 1 ML ROUGE-1** | **~0.35** | After KNN routing |
| **DL Fine-tuned ROUGE-1** | **0.85+ target** | Not yet achieved (training needed) |

### 4.3 Evaluation Methods

#### Method 1: Leave-One-Out (LOO) Cross-Validation
**Function:** `run_loo_evaluation()` in `src/evaluation/eval_suite.py`

For each of N pairs:
1. Build KNN index **excluding** pair i
2. Retrieve best neighbor from the held-out index
3. Generate narrative using full pipeline
4. Score against ground truth

This is the **primary evaluation method** because it prevents data leakage from the KNN retrieval step.

#### Method 2: Baseline Template-Only ROUGE
**Function:** `run_baseline_rouge()` — uses `ClinicalNLGEngine` without KNN, purely deterministic templates.

#### Method 3: Multi-Mode Comparison (API-based)
**Script:** `scripts/measure_all_metrics.py`

Hits both APIs simultaneously and produces a comparison table:
```
APPROACH             | ROUGE-1    | ROUGE-L    | NUM ACC    | HALL RATE
---------------------------------------------------------------------
ML                   | x.xxxx     | x.xxxx     | x.xxxx     | 0.0000
BASE_DL              | x.xxxx     | x.xxxx     | x.xxxx     | x.xxxx
FINETUNED            | x.xxxx     | x.xxxx     | x.xxxx     | x.xxxx
```

#### Method 4: Aggregate Statistics
`evaluate_dataset()` computes **mean, std, P10, P50, P90, min, max** for each metric across all examples.

**BERTScore** is listed in `src/requirements.txt` (`bert-score==0.3.13`) and mentioned in API endpoint comments (v3.0 `main_ml.py`) but is not yet integrated into the main evaluation loop. It is a planned addition.

---

## 5. Current Codebase

### 5.1 Directory Structure

```
medicalproject/
├── src/                          # Core library (all importable modules)
│   ├── api/
│   │   ├── main_ml.py            # FastAPI ML backend → Port 8000 (v3.0)
│   │   └── main_dl.py            # FastAPI DL backend → Port 8001
│   ├── data_processing/
│   │   ├── pdf_extractor.py      # PDF/DOCX → linearized table pairs
│   │   └── image_extractor.py    # JPEG/PNG → linearized table (EasyOCR)
│   ├── evaluation/
│   │   └── eval_suite.py         # Triple-metric evaluator + LOO CV
│   ├── feature_engineering/
│   │   ├── statistical_features.py   # 32-feat global table extractor
│   │   └── row_feature_extractor.py  # 6-feat per-row AE extractor
│   ├── generation/
│   │   ├── slot_fill_generator.py    # SlotFillGenerator v4.0 (main generator)
│   │   ├── template_clusters.py       # 14 cluster templates + select_cluster()
│   │   └── clinical_nlg_engine.py    # DEPRECATED v3 generator (kept for baseline)
│   ├── models/
│   │   └── content_selector.py    # ContentSelector wrapper (LightGBM)
│   ├── retrieval/
│   │   └── knn_retrieval_engine.py  # KNN engine (scikit-learn NearestNeighbors)
│   ├── training/
│   │   └── finetune_dl.py         # DL fine-tuning scaffold (A100/H100)
│   └── requirements.txt           # Full pinned dependency list
├── notebooks/
│   ├── 01_pfizer_data_extraction.py  # Pfizer CSR scrape + extraction pipeline
│   └── 02_train_t5xxl_optimized.py   # T5-XXL QLoRA training script (H100)
├── scripts/
│   ├── datasynt.py               # LightGBM training with synthetic data + SMOTE
│   ├── generate_synthetic_data.py # Extended synthetic pair generation
│   ├── train_content_selector.py  # Train ContentSelector from gold pairs
│   ├── measure_all_metrics.py     # Multi-mode evaluation runner
│   ├── baseline_rouge.py          # Compute baseline template ROUGE
│   ├── run_loo_eval.py            # Run LOO cross-validation
│   ├── parse_dataset.py           # Parse raw extracted data
│   ├── scrub_dataset.py           # Clean and validate dataset
│   ├── setup_and_run.py           # Full pipeline setup script
│   ├── stage2.py                  # Stage 2 clustering demo script
│   ├── stage3.py                  # Stage 3 generation demo script
│   └── testinference.py           # Quick inference smoke test
├── data/
│   ├── processed/raw_pairs.json   # Gold standard ground truth pairs
│   ├── processed/knn_pairs.json   # KNN index pairs
│   ├── pfizer_gold_pairs.jsonl    # Pfizer-extracted pairs (JSONL)
│   ├── retrieval_index.pkl        # Pre-built KNN index (serialized)
│   └── augmented/*.jsonl          # Synthetic augmented pairs
├── models/
│   ├── content_selector.pkl       # Trained LightGBM content selector
│   ├── best_clinical_adapter/     # Flan-T5-XL (3B) LoRA adapter
│   └── best_clinical_adapter_xxl/ # Flan-T5-XXL (11B) LoRA adapter
├── pfizer_csrs/                   # Raw Pfizer CSR PDFs (148 files)
├── pfizzer_scraper.py             # Pfizer CSR web scraper
├── requirements.txt               # Root requirements (abbreviated)
├── BACKEND_TECHNICAL_SPEC.md      # Backend architecture doc
├── SUMMARY_GENERATION_PROCESS.md  # ML pipeline explanation
├── ROADMAP_TO_PERFECTION.md       # 3-phase improvement plan
├── H100_FINETUNING_PLAN.md        # H100 fine-tuning instructions
└── context.md                     # Legacy comprehensive guide (ML+DL)
```

### 5.2 Key Dependencies

#### ML Stack
| Package | Version | Purpose |
|---------|---------|---------|
| `lightgbm` | 4.3.0 | Stage 1 Content Selector (primary ML model) |
| `xgboost` | 2.0.3 | Stage 1 ensemble (alternative/combined) |
| `catboost` | 1.2.5 | Stage 1 ensemble member |
| `scikit-learn` | 1.4.2 | Agglomerative Clustering, KNN, StandardScaler, SMOTE |
| `numpy` | 1.26.4 | Numerical operations |
| `pandas` | 2.2.2 | Data manipulation |
| `imbalanced-learn` | — | SMOTE implementation |

#### DL Stack
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.3.0 | PyTorch (base DL framework) |
| `transformers` | 4.43.0 | Flan-T5-XXL model loading |
| `peft` | 0.11.0 | LoRA / QLoRA adapter management |
| `bitsandbytes` | 0.43.1 | 4-bit NF4 quantization |
| `accelerate` | 0.31.0 | Distributed training + device_map |
| `trl` | 0.9.4 | Seq2Seq training utilities |
| `datasets` | 2.20.0 | HuggingFace Dataset management |

#### Evaluation
| Package | Version | Purpose |
|---------|---------|---------|
| `evaluate` | 0.4.2 | ROUGE metric loading |
| `rouge-score` | 0.1.2 | Underlying ROUGE implementation |
| `bert-score` | 0.3.13 | BERTScore (loaded, not yet in main eval) |

#### API & Extraction
| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.111.0 | REST API framework |
| `uvicorn` | 0.29.0 | ASGI server |
| `pdfplumber` | 0.11.2 | PDF table extraction (text-based) |
| `PyMuPDF` | 1.24.5 | PDF rendering + OCR fallback |
| `camelot-py` | 0.11.0 | Advanced PDF table extraction |
| `easyocr` | 1.7.1 | Image OCR |
| `img2table` | 1.2.2 | Image-to-table detection |
| `python-docx` | 1.1.2 | DOCX table extraction |
| `jinja2` | 3.1.4 | Legacy template rendering |
| `anthropic` | 0.30.0 | Claude API for LLM augmentation |

### 5.3 API Endpoints (ML Backend, Port 8000)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Status, corpus size, index loaded |
| `/summarize` | POST | Summarize a linearized table string |
| `/upload` | POST | ★ Primary: upload PDF/DOCX/JPEG/PNG |
| `/summarize/batch` | POST | Batch summarize up to 50 tables |
| `/clusters` | GET | List all 14 template clusters |
| `/features/names` | GET | List all 32 feature names |
| `/evaluate/single` | POST | Evaluate a prediction vs reference |

---

## 6. Current Bottlenecks and Active Bug History

### 6.1 Bugs Fixed in v4.0 (Critical)

#### Bug C1 — Arm Mislabeling ("Chemotherapy → Placebo" Bug)
**Symptom:** When a table had Chemotherapy as the control arm (not Placebo), the ArmParser was incorrectly identifying it as the experimental arm, causing all percentages to be attributed to the wrong arm.  
**Root Cause:** Original `ArmParser` only recognized "Placebo/Vehicle" as control arms.  
**Fix:** Extended `CONTROL_KEYWORDS` list to include all standard-of-care chemotherapy agents (`docetaxel`, `paclitaxel`, `carboplatin`, `cisplatin`, etc.)

#### Bug C2 — Column Order "95.0% → 92.0%" Bug
**Symptom:** Drug arm percentage was reading from the control arm's column.  
**Root Cause:** `experimental_arm_index` was computed in `ArmParser` but NOT propagated to `OverviewSlotExtractor`. The slot extractor was always reading column index `1` for the drug arm regardless.  
**Fix:** Passed `arm_ctx.experimental_arm_index` explicitly to `OverviewSlotExtractor.extract_slots()`. The extractor now uses `exp_idx` and `ctrl_idx = 1 - exp_idx` dynamically.

#### Bug C3 — Dynamic Slot Labels (Hardcoded "Arm A"/"Arm B" Strings)
**Symptom:** Output would say "Pembrolizumab arm" for one sentence and "Arm B" for another, or fall back to "Arm A" for named arms.  
**Root Cause:** Template slots were inconsistently filled with actual arm names vs. hardcoded fallback strings.  
**Fix:** `ARM_A` and `ARM_B` slots now always filled from `ArmParser.arm_a_label`/`arm_b_label` which come from actual HEADERS tokens.

#### Bug C4 — Individual AE Omission (Rash/Infusion Reaction Bug)
**Symptom:** Named individual AEs from detailed AE tables (e.g., per-PT tables listing Nausea, Rash, Headache) were being omitted entirely from summaries.  
**Root Cause:** The overview slot extractor only looked for TEAE/SAE/Grade 3-4/Discontinuation categories. Tables listing individual AEs by MedDRA Preferred Term had no matching slot patterns.  
**Fix:** Added `IndividualAERenderer` class and `C14_individual_ae_listing` cluster. `RowFeatureExtractor` extracts individual AE rows, `ContentSelector` (LightGBM) scores them, and `IndividualAERenderer` generates per-AE sentences appended to the overview.

#### Bug C5 — Arm Attribution Hallucination
**Symptom:** Percentage values were being attributed to the wrong arm in the output text (e.g., "42.0% in the Pembrolizumab arm" when that was actually the Chemotherapy arm's percentage).  
**Fix:** `HallucinationGuardian._check_arm_attribution()` parses "X% of subjects in the {arm_name} arm" patterns and cross-checks the percentage against the correct column in the source table.

#### Bug C6 — Ambiguous Template Attribution
**Symptom:** Templates used bare percentage slots like `{G34_PCT_B}%` without embedding the arm label, making the sentence grammatically ambiguous.  
**Fix:** All templates (v4.0) now embed arm labels inline: `"{G34_PCT_B}% of subjects in the {ARM_B} arm"`.

### 6.2 Ongoing Challenges and Open Questions

#### Challenge 1: ROUGE Score Ceiling (~0.27–0.35 for ML Path)
The deterministic template system produces grammatically correct and numerically exact text, but the n-gram overlap with human-written summaries is limited (~0.27 ROUGE-1 template-only, ~0.35 with KNN routing). Human medical writers vary sentence structure dramatically. **This is the primary open problem.**

**Current Plan (3-Phase Roadmap):**
- Phase 1 (Active): Expand to 14 template clusters + KNN stylistic routing (complete)
- Phase 2 (Next): "Bucketizer" — stratify events into Tier 1 (>10%), Tier 2 (>5%), Tier 3 (SAE/Severe) before surface realization
- Phase 3 (R&D): Flan-T5-XXL fine-tuned on 1,411 pairs as a "polish layer" over deterministic ML output, gated by HallucinationGuardian

#### Challenge 2: KNN De-Lexicalization Reliability
The `_adapt_retrieved` method in `SlotFillGenerator` attempts to replace historical drug names and numbers with slot placeholders using regex. This can fail if:
- Arm labels contain regex-special characters
- Number formatting is inconsistent between retrieved writeup and source table
- Arm labels are too short (< 2 chars) to safely regex-replace

**Fix path:** When adaptation fails, system falls back to hard-coded template clusters. Warning logged: `"KNN adaptation failed."`

#### Challenge 3: Data Volume for DL Fine-Tuning
Only ~172–191 verified gold pairs exist. DL models require significantly more. The augmentation pipeline (synthetic data, LLM paraphrase) expanded this to ~1,411 pairs, but the quality of augmented data is lower than gold data. Fine-tuning on low-quality synthetic data risks:
- ROUGE improvements that don't generalize to real unseen tables
- Template structure leakage (model memorizes sentence order rather than writing style)

**Mitigation:** 20% template augmentation (sentence shuffling) during DL training to prevent template memorization.

#### Challenge 4: LightGBM Label Assignment Quality
The current label assignment strategy uses **fuzzy string matching** of event names in the reference writeup. This is imperfect:
- Short event names (e.g., "rash") match anywhere in the writeup
- Abbreviations (e.g., "AE" vs. "adverse event") are not normalized
- Negations (e.g., "no rash was reported") are not handled

**Open Question:** Should we invest in MedDRA API/dictionary integration for proper semantic term mapping?

#### Challenge 5: Agglomerative Clustering Distance Threshold
`distance_threshold=4.0` was set heuristically. No systematic cross-validation has been done to determine optimal clustering granularity. The current threshold may lump together events that regulatory writers would separate (e.g., "mild GI events" with "moderate cardiac events" that have similar percentage profiles but different clinical significance).

#### Challenge 6: BERTScore Not Yet Integrated
`bert-score==0.3.13` is installed and mentioned but not connected to the main `evaluate_dataset()` function or the multi-mode comparison script. This would be a valuable add, especially for detecting semantically-correct but low-ROUGE outputs (e.g., a summary that correctly describes the data but uses different clinical phrasing than the reference).

### 6.3 Last Active Work (Conversation 9b85e0c4)
- Debugged the DL backend API endpoint
- Verified that `SlotFillGenerator` (v4) was correctly loaded by `main_dl.py` instead of the deprecated `ClinicalNLGEngine` (v3)
- Confirmed frontend (Next.js 15) and both backend services restart cleanly
- Confirmed that the write-up bug causing incorrect outputs was traced to the v3→v4 generator mismatch (system was calling old engine)

---

## 7. System Architecture Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Next.js 15 Frontend                          │
│              (PDF/Image upload UI + narrative display)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌─────────────────┐         ┌──────────────────┐
   │  ML Backend     │         │   DL Backend     │
   │  Port 8000      │         │   Port 8001      │
   │  main_ml.py     │         │   main_dl.py     │
   └────────┬────────┘         └────────┬─────────┘
            │                           │
   ┌────────▼────────┐         ┌────────▼──────────┐
   │  ArmParser      │         │  Flan-T5-XXL 4bit │
   │  LightGBM       │         │  + LoRA Adapter   │
   │  AggloClustering│         │  ICH E3 Prompt    │
   │  KNN Retrieval  │         └────────┬──────────┘
   │  Template Router│                  │
   │  SlotFiller     │         ┌────────▼──────────┐
   │  HalluGuardian  │         │  HalluGuardian    │
   └─────────────────┘         │  (Gate: pass/fail)│
                                └───────────────────┘
                                         │ if fail → fallback to ML output

   ┌─────────────────────────────────────────────────────────────┐
   │                 Evaluation Suite                            │
   │  ROUGE-1/2/L   |   Numeric Accuracy (±1.0)   |   Halluc %  │
   │  LOO Cross-Validation   |   Multi-mode comparison           │
   └─────────────────────────────────────────────────────────────┘
```

---

## 8. Quickstart for AI Research Agent

### Running the ML Backend
```bash
cd /Users/bhriguverma/medicalproject
source venv/bin/activate
uvicorn src.api.main_ml:app --host 0.0.0.0 --port 8000 --reload
```

### Running Evaluation
```bash
# LOO Cross-validation (primary)
python src/evaluation/eval_suite.py --pairs data/processed/raw_pairs.json --mode loo

# Multi-mode comparison (requires both backends running)
python scripts/measure_all_metrics.py

# Baseline template ROUGE only
python src/evaluation/eval_suite.py --pairs data/processed/raw_pairs.json --mode baseline
```

### Training the Content Selector
```bash
python scripts/train_content_selector.py --pairs data/processed/raw_pairs.json --output models/content_selector.pkl
```

### Training DL Model (H100 required)
```bash
python notebooks/02_train_t5xxl_optimized.py --epochs 12 --batch-size 2 --lr 5e-5 --lora-r 16
```

### Rebuilding the KNN Index
```bash
python src/retrieval/knn_retrieval_engine.py --pairs data/processed/raw_pairs.json --output data/retrieval_index.pkl --k 5
```

---

*Document generated: March 8, 2026. Based on full codebase review of `/Users/bhriguverma/medicalproject`.*
