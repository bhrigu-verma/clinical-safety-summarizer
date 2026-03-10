"""
scripts/benchmark_run.py
━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2 of the Benchmark Pipeline.

Runs all 3 models on every table in Tier 1 + Tier 2 and saves
per-table results to JSONL files with checkpointing.

Models tested:
  1. ML Engine     → calls SlotFillGenerator directly (no server needed)
  2. DL Base       → calls /summarize?mode=base_dl on port 8001
  3. DL Fine-tuned → calls /summarize?mode=finetuned on port 8001

Usage:
    # Start DL backend first:
    ./venv/bin/python3 src/api/main_dl.py &

    # Then run benchmark:
    ./venv/bin/python3 scripts/benchmark_run.py [--tier all|1|2] [--models all|ml|dl_base|dl_ft]
"""

import re
import json
import time
import argparse
import sys
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TIER1_FILE  = ROOT / "data" / "benchmark" / "tier1_gold.jsonl"
TIER2_FILE  = ROOT / "data" / "benchmark" / "tier2_stress.jsonl"
RESULTS_DIR = ROOT / "data" / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DL_BASE_URL = "http://localhost:8001"
CHECKPOINT_EVERY = 25   # Save checkpoint every N rows


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_hallucination_rate(prediction: str, source_table: str) -> float:
    source_nums = [float(n) for n in re.findall(r"\b\d+\.?\d*\b", source_table)]
    pred_raw    = re.findall(r"\b\d+\.?\d*\b", prediction)
    tbl_refs    = set(re.findall(r"Table\s+(\d+)", prediction, re.IGNORECASE))
    allowed     = {"0", "1", "2", "0.0", "1.0", "2.0"}
    if not pred_raw:
        return 0.0
    bad = []
    for p_str in pred_raw:
        if p_str in tbl_refs or p_str in allowed:
            continue
        p_val = float(p_str)
        if not any(abs(p_val - s) <= 0.11 for s in source_nums):
            bad.append(p_str)
    return round(len(bad) / len(pred_raw), 4)


def compute_numeric_accuracy(prediction: str, reference: str) -> float:
    if not reference:
        return None
    ref_nums  = [float(n) for n in re.findall(r"\b\d+\.?\d*\b", reference)]
    pred_nums = [float(n) for n in re.findall(r"\b\d+\.?\d*\b", prediction)]
    if not ref_nums:
        return 1.0
    correct = sum(1 for r in ref_nums if any(abs(r - p) <= 1.01 for p in pred_nums))
    return round(correct / len(ref_nums), 4)


def compute_rouge(prediction: str, reference: str) -> Dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 6),
            "rouge2": round(scores['rouge2'].fmeasure, 6),
            "rougeL": round(scores['rougeL'].fmeasure, 6)
        }
    except Exception as e:
        return {"rouge1": None, "rouge2": None, "rougeL": None, "error": str(e)}


# ── Model runners ──────────────────────────────────────────────────────────────

def run_ml(table_text: str) -> Dict:
    """Run the ML engine directly (no server needed)."""
    try:
        from src.generation.slot_fill_generator import SlotFillGenerator
        _gen = run_ml._gen  # reuse cached instance
    except AttributeError:
        from src.generation.slot_fill_generator import SlotFillGenerator
        run_ml._gen = SlotFillGenerator(strict_mode=False)
        _gen = run_ml._gen

    start = time.time()
    try:
        result = _gen.generate(table_text)
        summary = result.narrative
        warnings = result.warnings
        verified = result.verified
        num_acc  = result.numeric_accuracy
    except Exception as e:
        summary  = ""
        warnings = [str(e)]
        verified = False
        num_acc  = 0.0

    elapsed = round((time.time() - start) * 1000, 2)
    return {
        "model":        "ml_engine",
        "summary":      summary,
        "verified":     verified,
        "numeric_accuracy_self": num_acc,
        "inference_time_ms": elapsed,
        "warnings":     warnings,
        "success":      bool(summary),
    }


def run_dl(table_text: str, mode: str = "base_dl") -> Dict:
    """Call the DL backend server."""
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
                "model":        f"dl_{mode}",
                "summary":      data.get("summary", ""),
                "verified":     data.get("verified", False),
                "numeric_accuracy_self": data.get("numeric_accuracy", 0.0),
                "inference_time_ms": data.get("inference_time_ms", elapsed),
                "warnings":     data.get("warnings", []),
                "tokens_generated": data.get("tokens_generated"),
                "success":      bool(data.get("summary", "")),
            }
        else:
            return {
                "model": f"dl_{mode}", "summary": "", "verified": False,
                "numeric_accuracy_self": 0.0, "inference_time_ms": elapsed,
                "warnings": [f"HTTP {resp.status_code}: {resp.text[:200]}"],
                "success": False,
            }
    except requests.exceptions.ConnectionError:
        return {
            "model": f"dl_{mode}", "summary": "", "verified": False,
            "numeric_accuracy_self": 0.0,
            "inference_time_ms": round((time.time() - start) * 1000, 2),
            "warnings": ["DL server not running on port 8001"],
            "success": False,
        }
    except Exception as e:
        return {
            "model": f"dl_{mode}", "summary": "", "verified": False,
            "numeric_accuracy_self": 0.0,
            "inference_time_ms": round((time.time() - start) * 1000, 2),
            "warnings": [str(e)], "success": False,
        }


# ── Score a single result record ───────────────────────────────────────────────

def score_result(model_out: Dict, table_text: str, reference: Optional[str]) -> Dict:
    summary = model_out.get("summary", "")
    record  = {**model_out}

    # Hallucination rate (always — no reference needed)
    record["hallucination_rate"] = compute_hallucination_rate(summary, table_text)

    # ROUGE + numeric accuracy vs reference (Tier 1 only)
    if reference and summary:
        rouge = compute_rouge(summary, reference)
        record["rouge1"] = rouge.get("rouge1")
        record["rouge2"] = rouge.get("rouge2")
        record["rougeL"] = rouge.get("rougeL")
        record["numeric_accuracy_vs_ref"] = compute_numeric_accuracy(summary, reference)
    else:
        record["rouge1"] = None
        record["rouge2"] = None
        record["rougeL"] = None
        record["numeric_accuracy_vs_ref"] = None

    return record


# ── Load + filter input datasets ───────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    items = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def load_checkpoint(path: Path) -> set:
    """Load already-processed pair IDs from an existing results file."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["pair_id"])
                    except Exception:
                        pass
    return done


# ── Main benchmark runner ──────────────────────────────────────────────────────

def run_benchmark(tier: str = "all", model_flags: str = "all"):
    print("=" * 65)
    print("  BENCHMARK RUNNER — Phase 2")
    print("=" * 65)

    # --- Determine which datasets to use ---
    datasets = []  # list of (name, pairs)
    if tier in ("all", "1"):
        tier1 = load_jsonl(TIER1_FILE)
        if tier1:
            datasets.append(("tier1", tier1))
            print(f"  Tier 1 (Gold):  {len(tier1)} pairs")
        else:
            print("  ⚠️  Tier 1 file empty or missing. Run benchmark_extract.py first.")
    if tier in ("all", "2"):
        tier2 = load_jsonl(TIER2_FILE)
        if tier2:
            # Cap at 2000 for Tier 2
            tier2 = tier2[:2000]
            datasets.append(("tier2", tier2))
            print(f"  Tier 2 (Stress): {len(tier2)} tables")
        else:
            print("  ⚠️  Tier 2 file empty or missing. Run benchmark_extract.py first.")

    if not datasets:
        print("\n❌ No data to benchmark. Exiting.")
        return

    # --- Determine which models ---
    run_ml_flag  = model_flags in ("all", "ml")
    run_base_flag = model_flags in ("all", "dl_base")
    run_ft_flag  = model_flags in ("all", "dl_ft")

    # --- Check DL server (warn if unavailable) ---
    dl_available = False
    if run_base_flag or run_ft_flag:
        try:
            r = requests.get(f"{DL_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                dl_available = True
                print(f"\n  ✅ DL server reachable at {DL_BASE_URL}")
                health = r.json()
                print(f"     base_model_loaded: {health.get('base_model_loaded')}")
                print(f"     finetuned_model_loaded: {health.get('finetuned_model_loaded')}")
        except Exception:
            print(f"\n  ⚠️  DL server not reachable at {DL_BASE_URL}")
            print("     DL models will be skipped. Start with:")
            print("     ./venv/bin/python3 src/api/main_dl.py &")
            run_base_flag = False
            run_ft_flag   = False

    print()

    # --- Run over each dataset ---
    for ds_name, pairs in datasets:
        print(f"\n{'─'*65}")
        print(f"  Running on: {ds_name.upper()} ({len(pairs)} examples)")
        print(f"{'─'*65}")

        # Output files per model
        out_files = {}
        checkpoints = {}
        writers = {}
        if run_ml_flag:
            p = RESULTS_DIR / f"ml_{ds_name}.jsonl"
            out_files["ml"]   = p
            checkpoints["ml"] = load_checkpoint(p)
            writers["ml"]     = open(p, "a")
        if run_base_flag and dl_available:
            p = RESULTS_DIR / f"dl_base_{ds_name}.jsonl"
            out_files["dl_base"]   = p
            checkpoints["dl_base"] = load_checkpoint(p)
            writers["dl_base"]     = open(p, "a")
        if run_ft_flag and dl_available:
            p = RESULTS_DIR / f"dl_ft_{ds_name}.jsonl"
            out_files["dl_ft"]   = p
            checkpoints["dl_ft"] = load_checkpoint(p)
            writers["dl_ft"]     = open(p, "a")

        if not writers:
            print("  No models to run. Skipping.")
            continue

        skipped = sum(len(v) for v in checkpoints.values())
        if skipped:
            print(f"  Resuming — {skipped} already done (checkpoint)")

        for i, pair in enumerate(tqdm(pairs, desc=f"{ds_name}", unit="table")):
            pair_id    = pair.get("pair_id", f"{ds_name}_{i}")
            table_text = pair.get("table_text", "")
            reference  = pair.get("writeup")  # None for Tier 2

            if not table_text:
                continue

            # ── ML ────────────────────────────────────────────────────────────
            if "ml" in writers and pair_id not in checkpoints["ml"]:
                ml_out = run_ml(table_text)
                record = score_result(ml_out, table_text, reference)
                record["pair_id"]    = pair_id
                record["source_pdf"] = pair.get("source_pdf", "")
                record["table_text"] = table_text
                record["reference"]  = reference
                writers["ml"].write(json.dumps(record) + "\n")
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    writers["ml"].flush()

            # ── DL Base ───────────────────────────────────────────────────────
            if "dl_base" in writers and pair_id not in checkpoints["dl_base"]:
                dl_base_out = run_dl(table_text, mode="base_dl")
                record = score_result(dl_base_out, table_text, reference)
                record["pair_id"]    = pair_id
                record["source_pdf"] = pair.get("source_pdf", "")
                record["table_text"] = table_text
                record["reference"]  = reference
                writers["dl_base"].write(json.dumps(record) + "\n")
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    writers["dl_base"].flush()

            # ── DL Fine-Tuned ─────────────────────────────────────────────────
            if "dl_ft" in writers and pair_id not in checkpoints["dl_ft"]:
                dl_ft_out = run_dl(table_text, mode="finetuned")
                record = score_result(dl_ft_out, table_text, reference)
                record["pair_id"]    = pair_id
                record["source_pdf"] = pair.get("source_pdf", "")
                record["table_text"] = table_text
                record["reference"]  = reference
                writers["dl_ft"].write(json.dumps(record) + "\n")
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    writers["dl_ft"].flush()

        for w in writers.values():
            w.close()

        print(f"\n  ✅ {ds_name} benchmark complete.")
        for key, path in out_files.items():
            n = sum(1 for _ in open(path))
            print(f"     {key}: {n} records → {path.name}")

    print("\n" + "=" * 65)
    print("  ALL BENCHMARKS DONE. Run benchmark_report.py next.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClinicalSafe Benchmark Runner")
    parser.add_argument("--tier",   default="all",
                        choices=["all", "1", "2"],
                        help="Which tier to benchmark (default: all)")
    parser.add_argument("--models", default="all",
                        choices=["all", "ml", "dl_base", "dl_ft"],
                        help="Which models to run (default: all)")
    args = parser.parse_args()
    run_benchmark(tier=args.tier, model_flags=args.models)
