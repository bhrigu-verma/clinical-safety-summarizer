"""
scripts/benchmark_10k.py
━━━━━━━━━━━━━━━━━━━━━━━━
Runs all 3 models on all 9,630 tables from clinical_10k_final.jsonl.

Saves COMPLETE results (input table, reference writeup, and all 3 model
outputs with metrics) into a single consolidated JSONL file:
    data/benchmark/results/full_10k_results.jsonl

Features:
  • Checkpoint-safe: saves every 10 tables, can resume if interrupted
  • Logs progress with ETA
  • Computes per-table: hallucination rate, numeric accuracy, ROUGE scores

Usage:
    # Make sure DL backend is running first:
    #   ./venv/bin/python3 src/api/main_dl.py &
    
    # Run (ML only — fast, ~15 min):
    ./venv/bin/python3 scripts/benchmark_10k.py --models ml
    
    # Run all 3 (slow on CPU, fast on GPU):
    ./venv/bin/python3 scripts/benchmark_10k.py --models all
"""

import re
import json
import time
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INPUT_FILE  = ROOT / "clinical_10k_final.jsonl"
OUTPUT_FILE = ROOT / "data" / "benchmark" / "results" / "full_10k_results.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

DL_BASE_URL = "http://localhost:8001"
CHECKPOINT_EVERY = 10


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hallucination_rate(prediction: str, source_table: str) -> float:
    if not prediction:
        return 0.0
    source_nums = set()
    for n in re.findall(r"\b\d+\.?\d*\b", source_table):
        source_nums.add(float(n))
    pred_raw = re.findall(r"\b\d+\.?\d*\b", prediction)
    tbl_refs = set(re.findall(r"Table\s+(\d+)", prediction, re.IGNORECASE))
    allowed  = {"0", "1", "2", "0.0", "1.0", "2.0"}
    if not pred_raw:
        return 0.0
    bad = 0
    for p_str in pred_raw:
        if p_str in tbl_refs or p_str in allowed:
            continue
        p_val = float(p_str)
        if not any(abs(p_val - s) <= 0.11 for s in source_nums):
            bad += 1
    return round(bad / len(pred_raw), 4)


def compute_numeric_accuracy(prediction: str, reference: str) -> Optional[float]:
    if not reference or not prediction:
        return None
    ref_nums  = [float(n) for n in re.findall(r"\b\d+\.?\d*\b", reference)]
    pred_nums = [float(n) for n in re.findall(r"\b\d+\.?\d*\b", prediction)]
    if not ref_nums:
        return 1.0
    correct = sum(1 for r in ref_nums if any(abs(r - p) <= 1.01 for p in pred_nums))
    return round(correct / len(ref_nums), 4)


def compute_rouge(prediction: str, reference: str) -> Dict:
    if not reference or not prediction:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 6),
            "rouge2": round(scores['rouge2'].fmeasure, 6),
            "rougeL": round(scores['rougeL'].fmeasure, 6)
        }
    except Exception:
        return {"rouge1": None, "rouge2": None, "rougeL": None}


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

_ml_gen = None

def run_ml(table_text: str) -> Dict:
    global _ml_gen
    if _ml_gen is None:
        from src.generation.slot_fill_generator import SlotFillGenerator
        _ml_gen = SlotFillGenerator(strict_mode=False)

    start = time.time()
    try:
        result = _ml_gen.generate(table_text)
        summary  = result.narrative
        verified = result.verified
    except Exception as e:
        summary  = ""
        verified = False
    elapsed = round((time.time() - start) * 1000, 2)
    return {"summary": summary, "verified": verified, "time_ms": elapsed}


def run_dl(table_text: str, mode: str = "base_dl") -> Dict:
    import requests
    start = time.time()
    try:
        resp = requests.post(
            f"{DL_BASE_URL}/summarize",
            json={"table_text": table_text, "mode": mode, "num_beams": 2},
            timeout=120,
        )
        elapsed = round((time.time() - start) * 1000, 2)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "summary":  data.get("summary", ""),
                "verified": data.get("verified", False),
                "time_ms":  data.get("inference_time_ms", elapsed),
            }
        else:
            return {"summary": "", "verified": False, "time_ms": elapsed}
    except Exception:
        return {"summary": "", "verified": False,
                "time_ms": round((time.time() - start) * 1000, 2)}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> set:
    done = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ml",
                        choices=["all", "ml", "dl_base", "dl_ft"],
                        help="Which models to run (default: ml)")
    args = parser.parse_args()

    do_ml   = args.models in ("all", "ml")
    do_base = args.models in ("all", "dl_base")
    do_ft   = args.models in ("all", "dl_ft")

    # Load input
    print("=" * 65)
    print("  10K BENCHMARK — Full Results Generator")
    print("=" * 65)

    with open(INPUT_FILE) as f:
        tables = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"  Input tables: {len(tables)}")

    # Check DL server
    if do_base or do_ft:
        import requests
        try:
            r = requests.get(f"{DL_BASE_URL}/health", timeout=5)
            print(f"  DL server: ✅ ({r.json().get('device')})")
        except Exception:
            print("  DL server: ❌ Not reachable. Skipping DL models.")
            do_base = False
            do_ft   = False

    # Load checkpoint
    done_ids = load_checkpoint()
    print(f"  Already done: {len(done_ids)} (checkpoint)")
    remaining = len(tables) - len(done_ids)
    print(f"  Remaining:    {remaining}")
    print()

    out_f = open(OUTPUT_FILE, "a")
    start_all = time.time()
    processed = 0

    for i, entry in enumerate(tables):
        table_id = f"t10k_{i}"

        if table_id in done_ids:
            continue

        table_text = entry.get("table_text", "")
        reference  = entry.get("writeup", "")
        tier       = entry.get("tier", "")
        difficulty = entry.get("difficulty", "")

        if not table_text:
            continue

        record = {
            "id":         table_id,
            "idx":        i,
            "table_text": table_text,
            "reference":  reference,
            "tier":       tier,
            "difficulty": difficulty,
        }

        # ── ML Engine ─────────────────────────────────────────────────────
        if do_ml:
            ml = run_ml(table_text)
            rouge_ml = compute_rouge(ml["summary"], reference)
            record["ml_summary"]       = ml["summary"]
            record["ml_verified"]      = ml["verified"]
            record["ml_time_ms"]       = ml["time_ms"]
            record["ml_hallucination"] = compute_hallucination_rate(ml["summary"], table_text)
            record["ml_num_accuracy"]  = compute_numeric_accuracy(ml["summary"], reference)
            record["ml_rouge1"]        = rouge_ml["rouge1"]
            record["ml_rouge2"]        = rouge_ml["rouge2"]
            record["ml_rougeL"]        = rouge_ml["rougeL"]

        # ── DL Base ───────────────────────────────────────────────────────
        if do_base:
            dl_b = run_dl(table_text, "base_dl")
            rouge_b = compute_rouge(dl_b["summary"], reference)
            record["dl_base_summary"]       = dl_b["summary"]
            record["dl_base_verified"]      = dl_b["verified"]
            record["dl_base_time_ms"]       = dl_b["time_ms"]
            record["dl_base_hallucination"] = compute_hallucination_rate(dl_b["summary"], table_text)
            record["dl_base_num_accuracy"]  = compute_numeric_accuracy(dl_b["summary"], reference)
            record["dl_base_rouge1"]        = rouge_b["rouge1"]
            record["dl_base_rouge2"]        = rouge_b["rouge2"]
            record["dl_base_rougeL"]        = rouge_b["rougeL"]

        # ── DL Fine-tuned ─────────────────────────────────────────────────
        if do_ft:
            dl_ft = run_dl(table_text, "finetuned")
            rouge_ft = compute_rouge(dl_ft["summary"], reference)
            record["dl_ft_summary"]       = dl_ft["summary"]
            record["dl_ft_verified"]      = dl_ft["verified"]
            record["dl_ft_time_ms"]       = dl_ft["time_ms"]
            record["dl_ft_hallucination"] = compute_hallucination_rate(dl_ft["summary"], table_text)
            record["dl_ft_num_accuracy"]  = compute_numeric_accuracy(dl_ft["summary"], reference)
            record["dl_ft_rouge1"]        = rouge_ft["rouge1"]
            record["dl_ft_rouge2"]        = rouge_ft["rouge2"]
            record["dl_ft_rougeL"]        = rouge_ft["rougeL"]

        out_f.write(json.dumps(record) + "\n")
        processed += 1

        if processed % CHECKPOINT_EVERY == 0:
            out_f.flush()
            elapsed = time.time() - start_all
            rate    = processed / elapsed
            eta     = (remaining - processed) / rate if rate > 0 else 0
            pct     = (processed / remaining) * 100
            print(f"  [{processed}/{remaining}] {pct:.1f}%  "
                  f"| {rate:.1f} tables/s  "
                  f"| ETA: {int(eta//60)}m {int(eta%60)}s")

    out_f.close()

    total_time = time.time() - start_all
    print()
    print("=" * 65)
    print("  10K BENCHMARK COMPLETE")
    print("=" * 65)
    print(f"  Tables processed:  {processed}")
    print(f"  Total time:        {int(total_time//60)}m {int(total_time%60)}s")
    print(f"  Output file:       {OUTPUT_FILE}")
    print(f"  File size:         {OUTPUT_FILE.stat().st_size / (1024*1024):.1f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
