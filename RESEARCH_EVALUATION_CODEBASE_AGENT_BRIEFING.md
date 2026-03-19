# RESEARCH EVALUATION CODEBASE — AGENT BRIEFING DOCUMENT
## Clinical Safety Summarization: Evaluation, Metrics, and Paper Figure Generation
### Author: Bhrigu Verma | Last updated: March 2026

---

## WHO SHOULD READ THIS

This document is the **primary briefing for any coding agent, developer, or collaborator** working
on the evaluation and figure-generation components of the clinical safety summarization research
project. Read this **before touching any code**. It tells you what every file does, what every
metric means, how to run the system, and how to debug common failures.

---

## PROJECT CONTEXT (SHORT VERSION)

We have built a hybrid AI system that converts clinical safety tables (adverse event tables
from pharmaceutical Clinical Study Reports) into regulatory-style narrative summaries.

The system has **three modes**:
- **ML** (deterministic): slot-fill template pipeline, always grounded
- **DL-base**: Flan-T5-XL without fine-tuning
- **DL-finetuned**: Flan-T5-XL + QLoRA adapter, with a verification gate

**The DL fine-tuned model runs on GPU only** (too large for CPU inference).
All metric computation runs on CPU.

We are preparing a research paper for submission to JAMIA or npj Digital Medicine.
The evaluation code generates all metrics and paper figures.

---

## REPOSITORY STRUCTURE (EVALUATION-RELEVANT FILES)

```
medicalproject/
├── src/
│   ├── api/
│   │   ├── main_ml.py        ← ML backend (FastAPI, port 8000)
│   │   └── main_dl.py        ← DL backend (FastAPI, port 8001)
│   └── evaluation/
│       ├── metrics_suite.py          ← ALL METRIC IMPLEMENTATIONS (start here)
│       ├── run_full_evaluation.py    ← Loads benchmarks, calls APIs, computes metrics
│       ├── generate_figures.py       ← Generates all 8 paper figures
│       └── eval_suite.py             ← Original eval file (legacy, kept for ROUGE/numeric)
├── data/
│   ├── benchmark/
│   │   ├── tier1_gold.jsonl     ← 41 gold-standard examples (hand-annotated)
│   │   └── tier2_stress.jsonl   ← 78 stress-test examples (edge cases)
│   ├── eval_results/            ← OUTPUT: CSVs + aggregate JSONs go here
│   └── figures/                 ← OUTPUT: all 8 paper figures go here
└── scripts/
    ├── benchmark_extract.py     ← Creates tier1/tier2 from the full corpus
    └── run_loo_eval.py          ← Generates LOO learning curve data for figure 7
```

---

## THE THREE EVALUATION FILES — WHAT EACH ONE DOES

### 1. `src/evaluation/metrics_suite.py`

**This is the core file. All metric logic lives here.**

Contains these classes:

| Class | What it computes |
|---|---|
| `RougeMetrics` | ROUGE-1, ROUGE-2, ROUGE-L (uses rouge-score library) |
| `BertScoreMetric` | BERTScore F1 (uses bert-score library, can batch) |
| `MeteorMetric` | METEOR score (uses nltk) |
| `ReadabilityMetrics` | Flesch-Kincaid grade level (uses textstat) |
| `NumericAccuracyRate` | NAR — what fraction of output numbers exist in source |
| `HallucinationRate` | HR — sentence-level hallucination detection |
| `OmissionRate` | OR — fraction of source facts missing from output |
| `SeverityWeightedOmissionScore` | **★ NOVEL** SWOS — OR weighted by clinical severity |
| `NumericDriftIndex` | **★ NOVEL** NDI — continuous measure of numeric drift |
| `ArmConfusionRate` | **★ NOVEL** ACR — detects drug/placebo arm swap errors |
| `ClinicalEvaluationSuite` | Unified interface — call `evaluate_single()` or `evaluate_batch()` |

Also contains utility functions:
- `extract_numbers(text)` — regex-based number extraction from clinical text
- `extract_arm_value_pairs(text, arm_names)` — extracts (arm, value) pairs from text
- `sentence_tokenize(text)` — splits text into sentences
- `detect_severity_tier(text)` — returns 1-4 severity tier from clinical keywords
- `extract_clinical_facts(text)` — returns structured facts with severity from text

### 2. `src/evaluation/run_full_evaluation.py`

**This is the runner. It orchestrates the full evaluation pipeline.**

What it does:
1. Loads benchmark JSONL files
2. Calls ML backend (port 8000) or DL backend (port 8001) for each example
3. Runs `ClinicalEvaluationSuite.evaluate_batch()` on all examples
4. Saves per-example CSV and aggregate JSON to `data/eval_results/`

Important behaviour:
- If a backend is offline, it logs a warning and skips that mode
  (it does NOT crash — you can test metrics on pre-computed outputs)
- If your benchmark JSONL has a `summary_ml`, `summary_dl_base`, or
  `summary_finetuned` key, those pre-computed outputs are used WITHOUT
  calling the API. This is the recommended workflow for GPU-only models.

### 3. `src/evaluation/generate_figures.py`

**This generates all 8 paper figures from the CSV results.**

Run this AFTER `run_full_evaluation.py` has produced CSV files.
Each figure function is independent — if one fails, others still run.
Figures are saved as both PNG (300 DPI, for paper submission) and SVG
(for editing in Inkscape or Illustrator).

---

## THE THREE NOVEL METRICS — FULL EXPLANATION FOR PEER REVIEW

These are the metrics we are PROPOSING in this paper. They do not exist in any
prior clinical NLP paper in this exact form.

### ★ Metric 1: Severity-Weighted Omission Score (SWOS)

**The problem it solves:** Plain Omission Rate (OR) treats all omitted facts equally.
Omitting a fatal event is catastrophically more dangerous than omitting a mild TEAE.

**How it works:**
- Each source fact is assigned a severity tier: TEAE=1, Grade3-4=2, SAE=3, Fatal=4
- Each omitted fact contributes its tier weight to the numerator
- SWOS = sum(tier_i × omit_i) / sum(tier_i for all facts)

**Formula:**
```
SWOS = Σ (severity_tier_i × omission_indicator_i) / Σ severity_tier_i
```

**Why it's novel:**
- Asgari et al. (2025) reports a flat 3.45% omission rate with no severity weighting
- No prior table-to-text paper distinguishes fatal omissions from mild AE omissions
- SWOS directly operationalises the ICH E3 requirement to report serious events

**How to interpret the result:**
- SWOS = OR → omissions uniformly distributed across severity (neutral)
- SWOS > OR → model disproportionately omits HIGH-severity facts (DANGEROUS)
- SWOS < OR → model correctly prioritises serious events (GOOD)

**Code location:** `SeverityWeightedOmissionScore.compute()` in `metrics_suite.py`

---

### ★ Metric 2: Numeric Drift Index (NDI)

**The problem it solves:** NAR is binary — a number is either grounded or not.
This misses "soft hallucinations" where a model outputs a value that's close but
wrong — e.g. writing "12%" when the source says "12.3%", or "1,230" when the
source says "1,234". These would fail regulatory review but pass NAR if the
tolerance is set too loose.

**How it works:**
- For each output number g_i, find the closest source number s_i
- Compute relative drift: |g_i - s_i| / max(|s_i|, 1e-9)
- NDI = mean of all per-value drifts

**Formula:**
```
NDI = mean_i( |g_i - closest_source(g_i)| / max(|closest_source(g_i)|, ε) )
```

**Why it's novel:**
- All prior work uses binary correct/incorrect for numeric evaluation
- NDI is the first continuous measure of "how far" values have drifted
- Inspired by data-to-text faithfulness work (Wiseman 2017, Dusek 2020) but
  not previously applied to clinical safety table summarization

**Interpretation thresholds:**
- NDI < 0.01 → excellent (sub-1% average drift)
- NDI 0.01–0.05 → acceptable (small rounding errors only)
- NDI > 0.10 → concerning (systematic imprecision or wrong-table reference)
- NDI > 0.20 → severe (likely hallucination or wrong source used)

**Code location:** `NumericDriftIndex.compute()` in `metrics_suite.py`

---

### ★ Metric 3: Arm Confusion Rate (ACR)

**The problem it solves:** NAR and HR only detect NUMBER hallucinations — when a
value doesn't exist in the source. They CANNOT detect the ARM-SWAP error: when a
correct value is attributed to the wrong treatment arm.

Example where NAR fails to detect a critical error:
```
Source:    "Drug A: 12.3% TEAE; Placebo: 5.1% TEAE"
Output:    "Placebo reported 12.3% TEAE; Drug A reported 5.1% TEAE"

NAR = 1.0  ← ALL numbers are grounded! NAR sees no problem!
ACR = 1.0  ← arm swap detected correctly
```

**How it works:**
- Extract (arm_label, value) pairs from both source and generated text
- For each generated (arm_g, val_g) pair, find the source arm that owns val_g
- If arm_g ≠ source_arm → this is an arm confusion
- ACR = n_confused / n_evaluated_pairs

**Why it's novel:**
- No prior metric distinguishes "fabricated number" from "correct number, wrong arm"
- Arm attribution is architecturally monitored in our ArmParser but not previously
  quantified as a standalone evaluation metric
- Directly relevant to CSR adverse event reporting where treatment/control confusion
  is a critical safety error

**Code location:** `ArmConfusionRate.compute()` in `metrics_suite.py`

---

## HOW TO RUN THE FULL EVALUATION (STEP BY STEP)

### Prerequisites
```bash
pip install rouge-score bert-score nltk textstat numpy pandas matplotlib seaborn scipy tqdm requests
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 1: Start backends (in separate terminals)
```bash
# Terminal 1: ML backend
cd /Users/bhriguverma/medicalproject
uvicorn src.api.main_ml:app --host 0.0.0.0 --port 8000

# Terminal 2: DL backend (requires GPU for finetuned mode)
uvicorn src.api.main_dl:app --host 0.0.0.0 --port 8001
```

### Step 2: Pre-compute GPU summaries (IMPORTANT — DL fine-tuned is GPU only)

Since the fine-tuned T5-XL runs only on GPU, pre-compute all summaries in one
batch on the GPU machine, then add them to your benchmark JSONL files:

```python
# scripts/precompute_dl_summaries.py  (run on GPU machine)
import json
import requests

with open("data/benchmark/tier1_gold.jsonl") as fin, \
     open("data/benchmark/tier1_gold_with_summaries.jsonl", "w") as fout:
    for line in fin:
        ex = json.loads(line)
        # Call finetuned mode
        resp = requests.post("http://localhost:8001/summarize",
                             json={"table_text": ex["table_text"], "mode": "finetuned"})
        ex["summary_finetuned"] = resp.json().get("summary", "")
        # Call base DL mode
        resp = requests.post("http://localhost:8001/summarize",
                             json={"table_text": ex["table_text"], "mode": "base_dl"})
        ex["summary_dl_base"] = resp.json().get("summary", "")
        fout.write(json.dumps(ex) + "\n")
```

After this step, copy the `_with_summaries.jsonl` files to your CPU evaluation machine.
The runner will use the pre-computed summaries without calling the GPU backend.

### Step 3: Run evaluation
```bash
# Tier 1 evaluation (41 gold examples)
python src/evaluation/run_full_evaluation.py \
    --tier tier1 \
    --modes ml dl_base finetuned \
    --bertscore-model distilbert-base-uncased \
    --bertscore-device cpu \
    --output-dir data/eval_results

# Tier 2 evaluation (78 stress examples)
python src/evaluation/run_full_evaluation.py \
    --tier tier2 \
    --modes ml dl_base finetuned \
    --bertscore-model distilbert-base-uncased \
    --output-dir data/eval_results

# Quick test (5 examples only — good for verifying setup)
python src/evaluation/run_full_evaluation.py --tier tier1 --n-max 5
```

### Step 4: Generate all paper figures
```bash
python src/evaluation/generate_figures.py \
    --results-dir data/eval_results \
    --figures-dir data/figures \
    --tier tier1
```

---

## DEBUGGING GUIDE

### Issue: "Benchmark file not found"
**Symptom:** `FileNotFoundError: Benchmark file not found: data/benchmark/tier1_gold.jsonl`
**Fix:** Run `python scripts/benchmark_extract.py` to generate the benchmark tier files
from the full `clinical_10k_final.jsonl` corpus.

### Issue: "Backend not reachable"
**Symptom:** `ML backend not reachable at http://localhost:8000/health`
**Effect:** The runner will skip API calls and use empty strings as generated summaries.
**Fix:** Start the appropriate backend before running evaluation. The runner still
continues and computes metrics on empty strings (which will give 0.0 scores, which is
useful to see as a "floor" baseline).
**Workaround:** Add `summary_ml`, `summary_dl_base`, or `summary_finetuned` keys to
your benchmark JSONL. The runner will use these instead of calling the API.

### Issue: "bert-score model download slow or fails"
**Symptom:** BERTScore hangs during first run while downloading model weights.
**Fix:** Pre-download the model:
```python
from bert_score import BERTScorer
scorer = BERTScorer(model_type="distilbert-base-uncased", device="cpu")
```
For publication-quality BERTScore, use `roberta-large` (slower) or
`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (clinical domain).

### Issue: "METEOR returns 0.0 for everything"
**Symptom:** All METEOR scores are 0.0
**Fix:** NLTK wordnet data not downloaded.
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Issue: "No arm-attributed values found" in ACR
**Symptom:** ACR = 0.0 for all examples with note "No arm-attributed values found"
**Cause:** Arm names in your benchmark JSONL don't match the text in the generated summary.
**Fix:** Check `arm_names` field in your benchmark examples. The arm names must appear
verbatim in the generated text for ACR to work. Common issue: "Drug A" in arm_names but
"treatment arm" in the generated text.
**Debug:**
```python
from src.evaluation.metrics_suite import extract_arm_value_pairs
pairs = extract_arm_value_pairs(generated_text, ["Drug A", "Placebo"])
print(pairs)  # should show (arm, value) tuples
```

### Issue: "SWOS always equals OR"
**Cause:** No severity keywords detected in source table text.
**Fix:** Check that your source table linearization includes clinical keywords like
"TEAE", "SAE", "Grade 3", "fatal", "death", etc. The table's row labels must be in
the linearized string for `detect_severity_tier()` to work.
**Debug:**
```python
from src.evaluation.metrics_suite import extract_clinical_facts
facts = extract_clinical_facts(source_table_text)
for f in facts:
    print(f['severity_tier'], f['text'][:80])
```

### Issue: "Figure generation fails with no data"
**Symptom:** `FileNotFoundError: No result CSV files matching pattern`
**Fix:** Run `run_full_evaluation.py` first. Figure generation depends on the CSVs
produced by the evaluation runner.

### Issue: NDI values are all very high (>0.5)
**Cause:** Usually means source table numbers and output numbers are in different formats.
For example, source has "12.3" but output has "12.3%" — the % is stripped but "0.123"
might be in the output instead.
**Fix:** Check `extract_numbers()` output for both source and generated text:
```python
from src.evaluation.metrics_suite import extract_numbers
print("Source:", extract_numbers(source_table[:500]))
print("Output:", extract_numbers(generated_summary))
```
Adjust the `numeric_tolerance` parameter if needed.

### Issue: "OMP: Error #15: Initializing libiomp5" (macOS + PyTorch)
**Symptom:** Crash when running BERTScore on macOS
**Fix:** Add to your shell profile: `export KMP_DUPLICATE_LIB_OK=TRUE`
Or run: `PYTORCH_ENABLE_MPS_FALLBACK=1 python src/evaluation/...`

---

## BENCHMARK FILE FORMAT

Your `tier1_gold.jsonl` and `tier2_stress.jsonl` files should have this format,
one JSON object per line:

```json
{
  "table_id": "pfizer_csr_042_table_3",
  "table_text": "start_table\n[TABLE_TITLE: Summary of Treatment-Emergent Adverse Events]\n[HEADERS: | Event Type | Drug A (N=124) | Placebo (N=121) |]\n[ROW] TEAE | 89 | 71.8% | 65 | 53.7% |\n[ROW] SAE | 12 | 9.7% | 5 | 4.1% |\n[ROW] Grade 3-4 | 8 | 6.5% | 3 | 2.5% |\n[ROW] Fatal | 1 | 0.8% | 0 | 0.0% |\nend_table",
  "reference": "In the Drug A arm (N=124), 89 patients (71.8%) experienced treatment-emergent adverse events (TEAEs) compared with 65 patients (53.7%) in the Placebo arm (N=121). Serious adverse events were reported in 12 (9.7%) Drug A patients versus 5 (4.1%) Placebo patients. Grade 3 or 4 events occurred in 8 (6.5%) and 3 (2.5%) patients, respectively. One fatal event (0.8%) was observed in the Drug A arm; no fatal events occurred in the Placebo arm.",
  "arm_names": ["Drug A", "Placebo"],
  "n_drug_arm": 124,
  "n_control_arm": 121,
  "complexity": "medium",
  "summary_ml": "",
  "summary_dl_base": "",
  "summary_finetuned": ""
}
```

The `summary_*` keys can be pre-populated on a GPU machine and the runner will
use those instead of calling the live API. Leave them as `""` if using live backends.

---

## METRIC BENCHMARKS FROM LITERATURE

Use these values to contextualise your results in the paper:

| Metric | Published value | Source | Notes |
|---|---|---|---|
| Hallucination Rate | 1.47% | Asgari et al., npj Digit Med 2025 | GPT-4 clinical notes |
| Omission Rate | 3.45% | Asgari et al., npj Digit Med 2025 | GPT-4 clinical notes |
| LLaMA-2 hallucination | 38% | Granular Fact-Checking paper, 2024 | Zero-shot PICO eval |
| ROUGE-1 (clinical) | 0.58 | Various clinical NLP papers | Range 0.45–0.65 |
| BERTScore F1 | 0.91 | Various clinical NLP papers | roberta-large base |

Your target values for publication:
- NAR >= 0.95
- HR <= 0.02 (beat the Asgari 1.47% baseline)
- SWOS <= OR (model prioritises serious events)
- NDI <= 0.05 (sub-5% average numeric drift)
- ACR <= 0.01 (near-zero arm confusion)

---

## PUBLICATION CHECKLIST

Before submission, verify:

- [ ] All three modes evaluated on both Tier 1 AND Tier 2 benchmarks
- [ ] BERTScore computed with `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
      (not distilbert) for the final paper numbers
- [ ] Human expert scores collected for at least 20 examples (5-point rubric)
- [ ] LOO evaluation run with `scripts/run_loo_eval.py` to generate Figure 7 data
- [ ] Comparison against GPT-4 zero-shot baseline (call OpenAI API on tier1 examples)
- [ ] Comparison against Flan-T5-base (no LoRA) as ablation baseline
- [ ] All 8 figures generated at 300 DPI
- [ ] Aggregate table formatted for paper (mean ± std for all metrics, all modes)
- [ ] Novel metric formulas written in LaTeX for Methods section

---

## QUICK REFERENCE: KEY IMPORTS

```python
# ── Main evaluation entry point ──────────────────────────────────────
from src.evaluation.metrics_suite import ClinicalEvaluationSuite

suite = ClinicalEvaluationSuite(
    bertscore_model="distilbert-base-uncased",  # or "roberta-large" for publication
    bertscore_device="cpu",                     # or "cuda" if GPU available
    numeric_tolerance=0.01,
)

# Single example
result = suite.evaluate_single(
    generated_summary="Drug A had 71.8% TEAE vs Placebo 53.7%.",
    reference_summary="In Drug A (N=124), 71.8% had TEAE vs 53.7% in Placebo.",
    source_table="start_table ... [ROW] TEAE | Drug A | 89 | 71.8% | Placebo | 65 | 53.7% end_table",
    arm_names=["Drug A", "Placebo"],
)
print(f"NAR: {result.numeric_accuracy_rate:.3f}")
print(f"SWOS: {result.severity_weighted_omission_score:.3f}")
print(f"NDI: {result.numeric_drift_index:.3f}")
print(f"ACR: {result.arm_confusion_rate:.3f}")
print(f"Safety: {result.safety_score():.3f}")

# Batch evaluation
results = suite.evaluate_batch(examples_list)
agg = suite.aggregate_results(results)
print(agg["numeric_accuracy_rate"])  # {"mean": ..., "std": ..., "p10": ..., "p50": ..., "p90": ...}

# ── Utility functions (usable standalone) ─────────────────────────────
from src.evaluation.metrics_suite import (
    extract_numbers,
    extract_arm_value_pairs,
    detect_severity_tier,
    extract_clinical_facts,
)
```

---

## FILE DEPENDENCY GRAPH

```
metrics_suite.py          ← contains all metric logic
       ↑
run_full_evaluation.py    ← imports ClinicalEvaluationSuite
       ↓ writes
data/eval_results/*.csv
data/eval_results/*.json
       ↑
generate_figures.py       ← reads CSVs, writes figures
       ↓ writes
data/figures/figure*.png
data/figures/figure*.svg
```

The three files are intentionally decoupled. You can:
- Add a new metric ONLY in metrics_suite.py
- Run evaluation ONLY with run_full_evaluation.py
- Regenerate figures ONLY with generate_figures.py (no backend needed)

---

*End of Agent Briefing Document*
*For questions: refer to the paper draft or contact Bhrigu Verma*
