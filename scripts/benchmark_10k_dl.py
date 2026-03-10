"""
scripts/benchmark_10k_dl.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adds DL Base and DL Fine-tuned results to the existing ML results file.
Reads full_10k_results.jsonl, adds dl_base_* and dl_ft_* columns, 
saves to full_10k_all_models.jsonl.

Usage:
    ./venv/bin/python3 scripts/benchmark_10k_dl.py
"""

import re
import json
import time
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INPUT_FILE  = ROOT / "data" / "benchmark" / "results" / "full_10k_results.jsonl"
OUTPUT_FILE = ROOT / "data" / "benchmark" / "results" / "full_10k_all_models.jsonl"
DL_BASE_URL = "http://localhost:8001"
CHECKPOINT_EVERY = 5


def compute_hallucination_rate(prediction: str, source_table: str) -> float:
    if not prediction: return 0.0
    source_nums = set(float(n) for n in re.findall(r"\b\d+\.?\d*\b", source_table))
    pred_raw = re.findall(r"\b\d+\.?\d*\b", prediction)
    tbl_refs = set(re.findall(r"Table\s+(\d+)", prediction, re.IGNORECASE))
    allowed  = {"0", "1", "2", "0.0", "1.0", "2.0"}
    if not pred_raw: return 0.0
    bad = sum(1 for p in pred_raw if p not in tbl_refs and p not in allowed
              and not any(abs(float(p) - s) <= 0.11 for s in source_nums))
    return round(bad / len(pred_raw), 4)


def compute_numeric_accuracy(prediction: str, reference: str) -> Optional[float]:
    if not reference or not prediction: return None
    ref_nums  = [float(n) for n in re.findall(r"\b\d+\.?\d*\b", reference)]
    pred_nums = [float(n) for n in re.findall(r"\b\d+\.?\d*\b", prediction)]
    if not ref_nums: return 1.0
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
        return {"summary": "", "verified": False, "time_ms": elapsed}
    except Exception:
        return {"summary": "", "verified": False,
                "time_ms": round((time.time() - start) * 1000, 2)}


def load_done_ids() -> set:
    done = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    try: done.add(json.loads(line.strip())["id"])
                    except: pass
    return done


def main():
    print("=" * 65)
    print("  10K BENCHMARK — Adding DL Model Results")
    print("=" * 65)

    # Load existing ML results
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    print(f"  Input records (ML done): {len(records)}")

    # Check DL server
    import requests
    try:
        r = requests.get(f"{DL_BASE_URL}/health", timeout=5)
        health = r.json()
        print(f"  DL server: ✅ ({health.get('device')})")
        print(f"    Base model loaded: {health.get('base_model_loaded')}")
        print(f"    Finetuned loaded:  {health.get('finetuned_model_loaded')}")
    except Exception:
        print("  ❌ DL server not reachable. Start it first.")
        return

    done_ids = load_done_ids()
    remaining = len(records) - len(done_ids)
    print(f"  Already done: {len(done_ids)}")
    print(f"  Remaining:    {remaining}")
    print()

    out_f = open(OUTPUT_FILE, "a")
    start_all = time.time()
    processed = 0

    for rec in records:
        if rec["id"] in done_ids:
            continue

        table_text = rec.get("table_text", "")
        reference  = rec.get("reference", "")

        # DL Base
        dl_b = run_dl(table_text, "base_dl")
        rouge_b = compute_rouge(dl_b["summary"], reference)
        rec["dl_base_summary"]       = dl_b["summary"]
        rec["dl_base_verified"]      = dl_b["verified"]
        rec["dl_base_time_ms"]       = dl_b["time_ms"]
        rec["dl_base_hallucination"] = compute_hallucination_rate(dl_b["summary"], table_text)
        rec["dl_base_num_accuracy"]  = compute_numeric_accuracy(dl_b["summary"], reference)
        rec["dl_base_rouge1"]        = rouge_b["rouge1"]
        rec["dl_base_rouge2"]        = rouge_b["rouge2"]
        rec["dl_base_rougeL"]        = rouge_b["rougeL"]

        # DL Fine-tuned
        dl_ft = run_dl(table_text, "finetuned")
        rouge_ft = compute_rouge(dl_ft["summary"], reference)
        rec["dl_ft_summary"]       = dl_ft["summary"]
        rec["dl_ft_verified"]      = dl_ft["verified"]
        rec["dl_ft_time_ms"]       = dl_ft["time_ms"]
        rec["dl_ft_hallucination"] = compute_hallucination_rate(dl_ft["summary"], table_text)
        rec["dl_ft_num_accuracy"]  = compute_numeric_accuracy(dl_ft["summary"], reference)
        rec["dl_ft_rouge1"]        = rouge_ft["rouge1"]
        rec["dl_ft_rouge2"]        = rouge_ft["rouge2"]
        rec["dl_ft_rougeL"]        = rouge_ft["rougeL"]

        out_f.write(json.dumps(rec) + "\n")
        processed += 1

        if processed % CHECKPOINT_EVERY == 0:
            out_f.flush()
            elapsed = time.time() - start_all
            rate    = processed / elapsed
            eta     = (remaining - processed) / rate if rate > 0 else 0
            pct     = (processed / remaining) * 100
            print(f"  [{processed}/{remaining}] {pct:.1f}%  "
                  f"| {rate:.2f} tables/s  "
                  f"| ETA: {int(eta//3600)}h {int((eta%3600)//60)}m {int(eta%60)}s")

    out_f.close()
    total_time = time.time() - start_all
    print()
    print("=" * 65)
    print(f"  DL PASS COMPLETE — {processed} records updated")
    print(f"  Time: {int(total_time//3600)}h {int((total_time%3600)//60)}m")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
