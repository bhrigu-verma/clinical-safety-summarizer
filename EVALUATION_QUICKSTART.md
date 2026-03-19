# Evaluation Quickstart

This is the one-page runbook for evaluation and figure generation.
For full theory, metrics rationale, and peer-review narrative, see:
[RESEARCH_EVALUATION_CODEBASE_AGENT_BRIEFING.md](RESEARCH_EVALUATION_CODEBASE_AGENT_BRIEFING.md)

## 1) Install Dependencies

```bash
pip install -r requirements_eval.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 2) Start Backends (live mode)

```bash
# Terminal A
uvicorn src.api.main_ml:app --host 0.0.0.0 --port 8000

# Terminal B
uvicorn src.api.main_dl:app --host 0.0.0.0 --port 8001
```

## 3) (Recommended) Precompute DL Summaries on GPU

Use this when the fine-tuned model can only run on GPU:

```bash
python scripts/precompute_dl_summaries.py \
  --input data/benchmark/tier1_gold.jsonl \
  --output data/benchmark/tier1_gold_with_summaries.jsonl \
  --dl-url http://localhost:8001 \
  --modes base_dl finetuned
```

Then evaluate using the precomputed benchmark file (replace the original file or point to it in your workflow).

## 4) Run Evaluation

```bash
# Tier 1 (gold)
python src/evaluation/run_full_evaluation.py \
  --tier tier1 \
  --modes ml dl_base finetuned \
  --bertscore-model distilbert-base-uncased \
  --bertscore-device cpu \
  --output-dir data/eval_results

# Tier 2 (stress)
python src/evaluation/run_full_evaluation.py \
  --tier tier2 \
  --modes ml dl_base finetuned \
  --bertscore-model distilbert-base-uncased \
  --bertscore-device cpu \
  --output-dir data/eval_results
```

Smoke test:

```bash
python src/evaluation/run_full_evaluation.py --tier tier1 --n-max 5
```

## 5) Generate All Paper Figures

```bash
python src/evaluation/generate_figures.py \
  --results-dir data/eval_results \
  --figures-dir data/figures \
  --tier tier1
```

## 6) Output Locations

- Per-example metrics CSV: `data/eval_results/results_<tier>_<mode>_<timestamp>.csv`
- Aggregate metrics JSON: `data/eval_results/aggregate_<tier>_<mode>_<timestamp>.json`
- Figures (PNG + SVG): `data/figures/figure*.png`, `data/figures/figure*.svg`

## 7) Most Common Fixes

- Missing benchmark files: run `python scripts/benchmark_extract.py`
- METEOR always zero: download NLTK wordnet data
- DL backend unavailable: use precomputed `summary_dl_base` and `summary_finetuned`
- macOS PyTorch OMP issue:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```
