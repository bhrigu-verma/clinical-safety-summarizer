"""
scripts/train_content_selector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training pipeline for the LightGBM Content Selector (Stage 1).
Parses extracted table-writeup pairs to learn which clinical events
are typically selected for mention in summaries.
"""

import json
import argparse
import numpy as np
import sys
from pathlib import Path

# Add project root to path so we can import from src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.row_feature_extractor import RowFeatureExtractor
from src.models.content_selector import ContentSelector

def load_data(pairs_path: str):
    """Load and parse pairs, extracting feature vectors and labels."""
    if not Path(pairs_path).exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    with open(pairs_path, 'r') as f:
        if pairs_path.endswith('.jsonl'):
            pairs = [json.loads(line) for line in f]
        else:
            pairs = json.load(f)

    # Use default rightmost drug arm (common in our dataset)
    extractor = RowFeatureExtractor(experimental_arm_index=1)
    
    all_features = []
    all_labels = []

    print(f"Processing {len(pairs)} table-writeup pairs...")
    
    for pair in pairs:
        table_text = pair.get('table_text', '')
        writeup = pair.get('writeup', '').lower()
        
        if not table_text or not writeup:
            continue
            
        # 1. Extract features for all AE rows in this table
        try:
            row_vectors = extractor.extract_all(table_text)
        except Exception as e:
            print(f"Error parsing table: {e}")
            continue
        
        for rv in row_vectors:
            # 2. Assign Label
            # 1 = Mentioned in writeup, 0 = Ignored
            event_name = rv.row.label.lower()
            
            # Simple fuzzy match: is the event name mentioned in the human writeup?
            # We also check for common abbreviations or parts of the name
            if len(event_name) > 3 and event_name in writeup:
                all_labels.append(1)
            # Handle short terms with word boundaries
            elif event_name in writeup.split():
                all_labels.append(1)
            else:
                all_labels.append(0)
                
            all_features.append(rv.features)

    if not all_features:
        raise ValueError("No valid features extracted from pairs.")

    return np.array(all_features), np.array(all_labels)

def train(pairs_path: str, output_path: str):
    """Execution pipeline."""
    X, y = load_data(pairs_path)
    
    print(f"\nTraining set size: {len(X)} rows")
    print(f"Positive samples:  {sum(y)} (clinical significance)")
    print(f"Negative samples:  {len(y) - sum(y)} (ignorable/noise)")

    # Initialise and train
    selector = ContentSelector()
    selector.fit(X, y)
    
    # Save artefacts
    selector.save(output_path)
    print(f"\n✅ Training complete. Model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM AE Content Selector")
    parser.add_argument("--pairs", type=str, required=True, help="Path to raw_pairs.json")
    parser.add_argument("--output", type=str, default="models/content_selector.pkl", help="Output .pkl path")
    
    args = parser.parse_args()
    
    try:
        train(args.pairs, args.output)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
