"""
Evaluation Suite — Roadmap v2.0, Section 5

Computes the triple-metric evaluation suite required by the Roadmap:
  1. ROUGE-1, ROUGE-2, ROUGE-L
  2. Numeric Accuracy (custom clinical metric)
  3. Hallucination Rate

Also implements Leave-One-Out (LOO) cross-validation as specified in §5.2.

Targets (hard requirements from Roadmap):
  - P10 ROUGE-1 > 0.75  → production ready
  - Mean ROUGE-1 > 0.85 → publication quality
  - Mean Numeric Accuracy > 0.96
  - Hallucination Rate < 0.04
"""
import json
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import evaluate as hf_evaluate

rouge_metric = hf_evaluate.load('rouge')


# ── Metric definitions ────────────────────────────────────────────────────────

def scrub_reference(text: str) -> str:
    """Remove clinical artifacts that shouldn't be predicted (Page X of Y, etc)."""
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Document\s+ID:\s+[\w\d\-]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Confidential', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def compute_rouge(predictions: List[str], references: List[str]) -> Dict:
    # Scrub references before ROUGE calculation to ensure fair comparison
    scrubbed_refs = [scrub_reference(r) for r in references]
    result = rouge_metric.compute(
        predictions=predictions,
        references=scrubbed_refs,
        use_stemmer=True
    )
    return {k: round(v, 6) for k, v in result.items()}


def compute_numeric_accuracy(prediction: str, reference: str) -> Tuple[float, int, int]:
    """
    Check that all numbers in the reference appear (loosely) in the prediction.
    Allows +/- 1.0 for rounding (Roadmap v2 requirement).
    """
    ref_nums  = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', reference)]
    pred_nums = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', prediction)]
    
    if not ref_nums:
        return 1.0, 0, 0
    
    correct = 0
    for r in ref_nums:
        # Check if any pred_num is within 1.01 of r
        if any(abs(r - p) <= 1.01 for p in pred_nums):
            correct += 1
            
    return round(correct / len(ref_nums), 4), correct, len(ref_nums)


def compute_hallucination_rate(prediction: str, source_table: str) -> float:
    """
    Fraction of numbers in the prediction NOT present in the source table.
    Ignores indices (Table 1) and small context integers (0, 1, 2).
    """
    source_nums = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', source_table)]
    pred_raw    = re.findall(r'\b\d+\.?\d*\b', prediction)
    
    # Filter out Table refs
    tbl_refs = set(re.findall(r'Table\s+(\d+)', prediction, re.IGNORECASE))
    allowed_context = {'0', '1', '2', '0.0', '1.0', '2.0'}
    
    if not pred_raw:
        return 0.0
        
    bad = []
    for p_str in pred_raw:
        if p_str in tbl_refs or p_str in allowed_context:
            continue
        
        p_val = float(p_str)
        # Check if p_val exists in source as either exact or rounded
        if not any(abs(p_val - s) <= 0.01 for s in source_nums):
            bad.append(p_str)
            
    return round(len(bad) / len(pred_raw), 4)


# ── Single-example evaluation ─────────────────────────────────────────────────

def evaluate_single(
    prediction: str,
    reference:  str,
    source_table: str = ''
) -> Dict:
    rouge = compute_rouge([prediction], [reference])
    num_acc, correct, total = compute_numeric_accuracy(prediction, reference)
    hall_rate = compute_hallucination_rate(prediction, source_table) if source_table else None
    return {
        'rouge1': rouge['rouge1'],
        'rouge2': rouge['rouge2'],
        'rougeL': rouge['rougeL'],
        'numeric_accuracy': num_acc,
        'numeric_correct':  correct,
        'numeric_total':    total,
        'hallucination_rate': hall_rate,
    }


# ── Full dataset evaluation ───────────────────────────────────────────────────

def evaluate_dataset(results: List[Dict]) -> Dict:
    """
    Aggregate metrics across all examples.
    Each element of results must have keys: rouge1, rouge2, rougeL,
    numeric_accuracy, hallucination_rate.
    """
    rouge1_scores = [r['rouge1'] for r in results]
    rouge2_scores = [r['rouge2'] for r in results]
    rougeL_scores = [r['rougeL'] for r in results]
    num_acc_scores = [r['numeric_accuracy'] for r in results]
    hall_rates = [r['hallucination_rate'] for r in results if r.get('hallucination_rate') is not None]

    def stats(values):
        return {
            'mean': round(float(np.mean(values)), 4),
            'std':  round(float(np.std(values)),  4),
            'p10':  round(float(np.percentile(values, 10)), 4),
            'p50':  round(float(np.percentile(values, 50)), 4),
            'p90':  round(float(np.percentile(values, 90)), 4),
            'min':  round(float(np.min(values)),  4),
            'max':  round(float(np.max(values)),  4),
        }

    agg = {
        'rouge1': stats(rouge1_scores),
        'rouge2': stats(rouge2_scores),
        'rougeL': stats(rougeL_scores),
        'numeric_accuracy': stats(num_acc_scores),
    }
    if hall_rates:
        agg['hallucination_rate'] = stats(hall_rates)

    # Roadmap thresholds
    agg['thresholds'] = {
        'p10_rouge1_gt_0.75':    agg['rouge1']['p10'] > 0.75,
        'mean_rouge1_gt_0.85':   agg['rouge1']['mean'] > 0.85,
        'mean_num_acc_gt_0.96':  agg['numeric_accuracy']['mean'] > 0.96,
        'mean_hall_lt_0.04':     (
            agg['hallucination_rate']['mean'] < 0.04
            if hall_rates else None
        ),
    }

    agg['n_examples'] = len(results)
    return agg


def print_report(agg: Dict):
    print("\n" + "="*60)
    print("EVALUATION REPORT (Roadmap v2 — Triple Metric Suite)")
    print("="*60)
    for metric in ('rouge1', 'rouge2', 'rougeL', 'numeric_accuracy'):
        s = agg[metric]
        print(f"\n{metric.upper()}")
        print(f"  Mean: {s['mean']:.4f}  |  P10: {s['p10']:.4f}  |  P90: {s['p90']:.4f}")
    if 'hallucination_rate' in agg:
        s = agg['hallucination_rate']
        print(f"\nHALLUCINATION RATE")
        print(f"  Mean: {s['mean']:.4f}  |  P90: {s['p90']:.4f}")
    print(f"\nEXAMPLES: {agg['n_examples']}")
    print("\nROADMAP THRESHOLD CHECKS:")
    for k, v in agg['thresholds'].items():
        marker = "✅" if v else "❌"
        print(f"  {marker} {k}: {v}")
    print("="*60 + "\n")


# ── Leave-One-Out Evaluation ──────────────────────────────────────────────────

def run_loo_evaluation(
    pairs: List[Dict],
    index_path: str = 'data/retrieval_index.pkl',
    verbose: bool = True
) -> Dict:
    """
    LOO cross-validation as specified in Roadmap v2, §5.2.

    For each of N pairs:
      1. Remove it from the retrieval index (LOO split).
      2. Generate its writeup using the pipeline.
      3. Score against ground truth.

    Returns aggregated metrics across all examples.
    """
    from src.retrieval.knn_retrieval_engine import KNNRetrievalEngine
    from src.generation.slot_fill_generator import SlotFillGenerator

    engine_full = KNNRetrievalEngine(k=5)
    engine_full.build(pairs)

    generator = SlotFillGenerator(strict_mode=False)

    results = []
    for i, pair in enumerate(pairs):
        if verbose and (i + 1) % 10 == 0:
            print(f"  LOO [{i+1}/{len(pairs)}]...")

        # Build LOO index (exclude example i)
        loo_engine = engine_full.build_loo(pairs, exclude_idx=i)

        # Retrieve best neighbour
        retrieved = loo_engine.retrieve_best(pair['table_text'])
        retrieved_writeup = retrieved.get('writeup', '')
        retrieved_table_text = retrieved.get('table_text', '')

        # Generate narrative
        try:
            result = generator.generate(
                pair['table_text'],
                retrieved_writeup=retrieved_writeup,
                retrieved_table_text=retrieved_table_text
            )
            prediction = result.narrative
        except Exception as e:
            prediction = ''
            if verbose:
                print(f"  Warning: generation failed for example {i}: {e}")

        # Score
        metrics = evaluate_single(
            prediction=prediction,
            reference=pair['writeup'],
            source_table=pair['table_text']
        )
        results.append(metrics)

    agg = evaluate_dataset(results)
    print_report(agg)
    return agg


# ── Baseline (template-only) ROUGE ────────────────────────────────────────────

def run_baseline_rouge(pairs: List[Dict]) -> Dict:
    """
    Run baseline ROUGE using our NEW Phase 1 & 2 ClinicalNLGEngine.
    """
    from src.generation.clinical_nlg_engine import ClinicalNLGEngine
    generator = ClinicalNLGEngine(model_path='models/content_selector.pkl')

    results = []
    for pair in pairs:
        try:
            prediction = generator.generate(pair['table_text'])
        except Exception:
            prediction = ''
        metrics = evaluate_single(
            prediction=prediction,
            reference=pair['writeup'],
            source_table=pair['table_text']
        )
        results.append(metrics)

    agg = evaluate_dataset(results)
    print("\n[BASELINE — template-only, no KNN]")
    print_report(agg)
    return agg


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation suite for Clinical ML Narrative Engine')
    parser.add_argument('--pairs',    required=True, help='Path to 187_pairs.json')
    parser.add_argument('--mode',     default='loo',
                        choices=['loo', 'baseline', 'both'],
                        help='Evaluation mode')
    parser.add_argument('--output',   default=None, help='Optional output JSON path')
    args = parser.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)

    all_results = {}

    if args.mode in ('baseline', 'both'):
        all_results['baseline'] = run_baseline_rouge(pairs)

    if args.mode in ('loo', 'both'):
        all_results['loo'] = run_loo_evaluation(pairs)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ Results saved to {args.output}")
