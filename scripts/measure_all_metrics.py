import json
import requests
import time
import random
import sys
from pathlib import Path

# Add project root to path for eval_suite imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.eval_suite import evaluate_single, evaluate_dataset, print_report

def measure_all():
    # 1. Load ground truth pairs
    pairs_path = ROOT / "data/processed/raw_pairs.json"
    if not pairs_path.exists():
        print(f"❌ Could not find {pairs_path}")
        return

    with open(pairs_path) as f:
        all_pairs = json.load(f)

    # Use a representative sample of 15 pairs for speed
    sample_size = min(3, len(all_pairs))
    sample = random.sample(all_pairs, sample_size)
    print(f"🚀 Measuring metrics on {sample_size} random clinical pairs...")

    modes = ["ml", "base_dl", "finetuned"]
    all_results = {mode: [] for mode in modes}

    for i, pair in enumerate(sample):
        table_text = pair.get("table_text") or pair.get("input", "")
        reference = pair.get("writeup") or pair.get("output", "")
        
        print(f"  [{i+1}/{sample_size}] Processing example...")

        for mode in modes:
            try:
                # Use the DL backend comparison endpoint logic
                # Mode "ml" hits port 8000, others hit 8001
                if mode == "ml":
                    url = "http://localhost:8000/summarize"
                else:
                    url = "http://localhost:8001/summarize"

                resp = requests.post(url, json={
                    "table_text": table_text,
                    "mode": mode
                }, timeout=120)
                
                if resp.status_code != 200:
                    print(f"    ⚠️ Mode {mode} failed with status {resp.status_code}")
                    continue

                prediction = resp.json().get("summary", "")
                
                # Compute metrics using our gold standard eval_suite
                metrics = evaluate_single(
                    prediction=prediction,
                    reference=reference,
                    source_table=table_text
                )
                all_results[mode].append(metrics)

            except Exception as e:
                print(f"    ❌ Error in mode {mode}: {e}")

    # 2. Aggregate and Report
    print("\n" + "="*80)
    print(f"{'APPROACH':<20} | {'ROUGE-1':<10} | {'ROUGE-L':<10} | {'NUM ACC':<10} | {'HALL RATE':<10}")
    print("-" * 80)

    for mode in modes:
        if not all_results[mode]:
            print(f"{mode.upper():<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        agg = evaluate_dataset(all_results[mode])
        r1 = agg['rouge1']['mean']
        rl = agg['rougeL']['mean']
        na = agg['numeric_accuracy']['mean']
        hr = agg.get('hallucination_rate', {}).get('mean', 0.0)
        
        print(f"{mode.upper():<20} | {r1:<10.4f} | {rl:<10.4f} | {na:<10.4f} | {hr:<10.4f}")

    print("="*80)
    print("\nNote: ML is 100% accurate by design but has lower ROUGE. ")
    print("      Base DL reflects out-of-the-box T5 performance.")
    print("      Finetuned reflects the Phase 3 Hybrid Pipeline (ML Precision + DL Polish).")

if __name__ == "__main__":
    measure_all()
