# Evaluation Debug Run Report (Mar 16, 2026)

## Scope
This report documents:
- Commands that were run during debugging
- Outputs produced
- Errors observed
- Confirmed root causes
- Recommended fix path

## Environment
- Workspace: /Users/bhriguverma/medicalproject
- Python env: venv (Python 3.9.6)
- Date: 2026-03-16

---

## Commands Executed and Outcomes

### 1) Figure generation (initial failing attempt)
Command:

```bash
python src/evaluation/generate_figures.py \
  --results-dir data/eval_results \
  --figures-dir data/figures \
  --tier tier1
```

Initial outcome:
- Exit code 2
- Error: script path mismatch in repository layout (file existed at repo root, not under src/evaluation at that time).

Action taken:
- Added compatibility wrappers so documented commands under src/evaluation now work.

---

### 2) Re-run figure generation after compatibility fix
Command:

```bash
/Users/bhriguverma/medicalproject/venv/bin/python src/evaluation/generate_figures.py \
  --results-dir data/eval_results \
  --figures-dir data/figures \
  --tier tier1
```

Outcome:
- Figure 1 generated successfully.
- Then warning/error: no tier1 result CSV files found yet.

Message observed:
- "No result CSV files matching pattern: data/eval_results/results_tier1_*.csv"

---

### 3) Run evaluation to create CSVs
Command:

```bash
/Users/bhriguverma/medicalproject/venv/bin/python src/evaluation/run_full_evaluation.py \
  --tier tier1 \
  --modes ml dl_base finetuned \
  --n-max 5 \
  --bertscore-model distilbert-base-uncased \
  --bertscore-device cpu \
  --output-dir data/eval_results
```

First outcome:
- Failed due to import path assumptions in root-level run_full_evaluation.py (expected src/evaluation location).

Action taken:
- Patched project-root detection in run_full_evaluation.py.

Second outcome:
- Failed due to module loading edge case in wrappers (dataclass initialization via dynamic import).

Action taken:
- Registered dynamically loaded modules in sys.modules in all wrappers:
  - src/evaluation/metrics_suite.py
  - src/evaluation/run_full_evaluation.py
  - src/evaluation/generate_figures.py

Final outcome:
- Evaluation completed for tier1 (n=5) and wrote CSV/JSON outputs.

---

### 4) Generate figures after evaluation outputs exist
Command:

```bash
/Users/bhriguverma/medicalproject/venv/bin/python src/evaluation/generate_figures.py \
  --results-dir data/eval_results \
  --figures-dir data/figures \
  --tier tier1
```

Outcome:
- Completed successfully.
- All figure files (1-8) generated in PNG and SVG.
- Warnings present for missing tier2 data and missing LOO files (expected with current dataset scope).

---

### 5) Backend diagnostics (why summaries are empty)
Commands:

```bash
curl -s -o /tmp/ml_health.txt -w "%{http_code}" http://localhost:8000/health
curl -s -o /tmp/dl_health.txt -w "%{http_code}" http://localhost:8001/health
```

Observed:
- 8000 => 404 with body {"detail":"Not Found"}
- 8001 => 404 with body {"detail":"Not Found"}

Further diagnostics:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

Observed:
- Both ports were owned by Docker process, not the expected uvicorn apps for src.api.main_ml/main_dl.

OpenAPI path checks showed unrelated APIs on 8000/8001, confirming endpoint mismatch.

---

## Outputs Produced

### Evaluation outputs
- data/eval_results/results_tier1_ml_20260316_114045.csv
- data/eval_results/results_tier1_dl_base_20260316_114045.csv
- data/eval_results/results_tier1_finetuned_20260316_114045.csv
- data/eval_results/aggregate_tier1_ml_20260316_114045.json
- data/eval_results/aggregate_tier1_dl_base_20260316_114045.json
- data/eval_results/aggregate_tier1_finetuned_20260316_114045.json

### Figure outputs
- data/figures/figure1_architecture.png
- data/figures/figure1_architecture.svg
- data/figures/figure2_nar_comparison.png
- data/figures/figure2_nar_comparison.svg
- data/figures/figure3_hallucination_comparison.png
- data/figures/figure3_hallucination_comparison.svg
- data/figures/figure4_safety_fluency_scatter.png
- data/figures/figure4_safety_fluency_scatter.svg
- data/figures/figure5_gate_breakdown.png
- data/figures/figure5_gate_breakdown.svg
- data/figures/figure6_rouge_vs_complexity.png
- data/figures/figure6_rouge_vs_complexity.svg
- data/figures/figure7_loo_learning_curve.png
- data/figures/figure7_loo_learning_curve.svg
- data/figures/figure8_metric_correlation.png
- data/figures/figure8_metric_correlation.svg

---

## Errors and Root Causes

### Error A: Exit code 2 when running src/evaluation/generate_figures.py
Root cause:
- File path/layout mismatch before wrappers existed.

Fix applied:
- Added compatibility wrapper file at src/evaluation/generate_figures.py.

### Error B: No CSV files found for figure generation
Root cause:
- Evaluation had not yet been run successfully.

Fix applied:
- Ran run_full_evaluation and generated tier1 CSV/JSON outputs.

### Error C: ModuleNotFoundError / dataclass dynamic import failures
Root cause:
- Wrapper dynamic import module lifecycle not fully registered in sys.modules.

Fix applied:
- Added explicit sys.modules registration before exec_module.

### Error D: Empty generated summaries in evaluation outputs
Root cause (primary):
- Services on localhost:8000 and localhost:8001 were not the expected ML/DL summarization backends.
- Health and summarize endpoints expected by runner were unavailable (404 / endpoint mismatch).

Impact:
- Runner used fallback empty strings for generated summaries.
- Metrics are placeholder-like and not final research-quality numbers.

---

## Remaining Gaps to Reach Final Research-Quality Runs

1. Start correct ML/DL apps on non-conflicting ports (recommended 8010/8011) or free 8000/8001.
2. Re-run full tier1 and tier2 evaluations against correct endpoints.
3. Ensure bert-score package/model is available for non-zero BERTScore values.
4. Generate LOO files for Figure 7 if non-illustrative curve is required.

---

## Recommended next command set

```bash
# ML backend
uvicorn src.api.main_ml:app --host 0.0.0.0 --port 8010

# DL backend
uvicorn src.api.main_dl:app --host 0.0.0.0 --port 8011

# Evaluation
/Users/bhriguverma/medicalproject/venv/bin/python src/evaluation/run_full_evaluation.py \
  --tier tier1 --modes ml dl_base finetuned \
  --ml-url http://localhost:8010 --dl-url http://localhost:8011 \
  --output-dir data/eval_results

# Figures
/Users/bhriguverma/medicalproject/venv/bin/python src/evaluation/generate_figures.py \
  --results-dir data/eval_results --figures-dir data/figures --tier tier1
```
