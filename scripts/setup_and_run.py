#!/usr/bin/env python3
"""
scripts/setup_and_run.py
━━━━━━━━━━━━━━━━━━━━━━━━
One-script setup: extracts data, builds KNN index, starts both backends.

Usage:
    # Full setup from DOCX:
    python scripts/setup_and_run.py --docx data/raw/Dataset.docx

    # Full setup with synthetic data generation (free tiers only):
    python scripts/setup_and_run.py --docx data/raw/Dataset.docx --augment

    # With LLM augmentation (requires API key):
    python scripts/setup_and_run.py --docx data/raw/Dataset.docx --augment --api-key sk-ant-...

    # Start servers only (skip data prep):
    python scripts/setup_and_run.py --serve-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def step(msg: str):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def extract_data(docx_path: str) -> int:
    step(f"Step 1: Extracting table-writeup pairs from {docx_path}")

    from src.data_processing.pdf_extractor import ClinicalPDFExtractor

    extractor = ClinicalPDFExtractor(docx_path)
    pairs = extractor.extract_all()

    # Filter: only keep pairs that have both table and writeup
    valid_pairs = [p for p in pairs if p.table_text and p.writeup]
    invalid = len(pairs) - len(valid_pairs)

    print(f"  Extracted: {len(pairs)} total, {len(valid_pairs)} valid, {invalid} skipped (no writeup)")

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Save as raw_pairs.json (for KNN index + augmentation)
    with open("data/processed/raw_pairs.json", "w") as f:
        json.dump([p.to_dict() for p in valid_pairs], f, indent=2)

    # Also save in KNN-compatible format (table_text / writeup keys)
    knn_pairs = [
        {"table_text": p.table_text, "writeup": p.writeup}
        for p in valid_pairs
    ]
    with open("data/processed/knn_pairs.json", "w") as f:
        json.dump(knn_pairs, f, indent=2)

    print(f"  Saved → data/processed/raw_pairs.json ({len(valid_pairs)} pairs)")
    return len(valid_pairs)


def augment_data(api_key: str = None, tiers: list = None):
    step("Step 2: Generating synthetic training data")

    with open("data/processed/raw_pairs.json") as f:
        raw = json.load(f)

    pairs = [
        {"table_text": p.get("table_text") or p.get("input", ""),
         "writeup": p.get("writeup") or p.get("output", "")}
        for p in raw
        if (p.get("table_text") or p.get("input")) and (p.get("writeup") or p.get("output"))
    ]

    from scripts.generate_synthetic_data import run_pipeline
    run_pipeline(
        input_pairs=pairs,
        output_dir="data/augmented",
        api_key=api_key,
        tiers=tiers or ([1, 2, 3, 4] if api_key else [1, 2]),
        target_count=1000,
    )


def build_knn_index():
    step("Step 3: Building KNN retrieval index")

    knn_path = "data/processed/knn_pairs.json"
    if not Path(knn_path).exists():
        # Fall back to augmented data
        knn_path = "data/augmented/synthetic_1000.jsonl"

    if not Path(knn_path).exists():
        print("  ⚠️  No training data found. Skipping KNN build.")
        return

    from src.retrieval.knn_retrieval_engine import KNNRetrievalEngine

    # Load pairs
    if knn_path.endswith(".jsonl"):
        pairs = []
        with open(knn_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                tw = ex.get("table_text") or ex.get("input", "")
                wr = ex.get("writeup") or ex.get("output", "")
                if tw and wr:
                    pairs.append({"table_text": tw, "writeup": wr})
    else:
        with open(knn_path) as f:
            raw = json.load(f)
        pairs = [
            {"table_text": p.get("table_text") or p.get("input", ""),
             "writeup": p.get("writeup") or p.get("output", "")}
            for p in raw
            if (p.get("table_text") or p.get("input")) and (p.get("writeup") or p.get("output"))
        ]

    print(f"  Building index from {len(pairs)} pairs...")
    engine = KNNRetrievalEngine(k=5)
    engine.build(pairs)

    Path("data").mkdir(parents=True, exist_ok=True)
    engine.save("data/retrieval_index.pkl")
    print(f"  ✅ KNN index saved → data/retrieval_index.pkl")


def start_servers():
    step("Step 4: Starting ML and DL backend servers")

    print("  Starting ML backend on http://localhost:8000 ...")
    ml_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main_ml:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT,
    )

    time.sleep(2)

    print("  Starting DL backend on http://localhost:8001 ...")
    dl_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main_dl:app",
         "--host", "0.0.0.0", "--port", "8001"],
        cwd=ROOT,
    )

    print("\n  ✅ Servers running!")
    print("  ML API:  http://localhost:8000/docs")
    print("  DL API:  http://localhost:8001/docs")
    print("\n  Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping servers...")
        ml_proc.terminate()
        dl_proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx",       help="Path to the DOCX dataset")
    parser.add_argument("--augment",    action="store_true", help="Generate synthetic data")
    parser.add_argument("--api-key",    default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--tiers",      nargs="+", type=int, default=None)
    parser.add_argument("--serve-only", action="store_true", help="Skip data prep, just start servers")
    parser.add_argument("--no-serve",   action="store_true", help="Prepare data but don't start servers")
    args = parser.parse_args()

    os.chdir(ROOT)

    if not args.serve_only:
        if args.docx:
            n = extract_data(args.docx)
            if n == 0:
                print("⚠️  No pairs extracted. Check your DOCX format.")
        elif not Path("data/processed/raw_pairs.json").exists():
            print("❌ No --docx provided and no existing data found.")
            sys.exit(1)

        if args.augment:
            augment_data(api_key=args.api_key, tiers=args.tiers)

        build_knn_index()

    if not args.no_serve:
        start_servers()
