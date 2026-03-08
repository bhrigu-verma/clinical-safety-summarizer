#!/usr/bin/env python3
"""
Roadmap v2.0 §9.1 - Baseline ROUGE (Generator only, no KNN)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.eval_suite import run_baseline_rouge
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', default='data/187_pairs.json')
    args = parser.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)

    print(f"🚀 Running Baseline Evaluation (Template-only) on {len(pairs)} pairs...")
    run_baseline_rouge(pairs)

if __name__ == '__main__':
    main()
