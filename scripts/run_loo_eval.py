#!/usr/bin/env python3
"""
Roadmap v2.0 §5.2 - Leave-One-Out Evaluation Wrapper
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.eval_suite import run_loo_evaluation
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', default='data/187_pairs.json')
    args = parser.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)

    print(f"🚀 Running LOO Evaluation on {len(pairs)} pairs...")
    run_loo_evaluation(pairs)

if __name__ == '__main__':
    main()
