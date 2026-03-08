"""
Dataset Scrubber — Roadmap v2.0
Filters the 199 pairs to keep only high-quality narratives.
"""
import json

def is_quality(pair):
    w = pair['writeup'].lower()
    # 1. Must be long enough to be a narrative
    if len(w) < 40:
        return False
    # 2. Must not just be a title or note
    if w.startswith('table') and len(w) < 60:
        return False
    if w.startswith('note:') and len(w) < 60:
        return False
    if 'write up of table' in w:
        return False
    # 3. Should contain some descriptive clinical words
    indicators = ['subjects', 'reported', 'adverse', 'safety', 'percent', 'arm', 'treatment']
    if not any(ind in w for ind in indicators):
        return False
    return True

with open('data/187_pairs.json', 'r') as f:
    pairs = json.load(f)

quality_pairs = [p for p in pairs if is_quality(p)]

print(f"Total pairs: {len(pairs)}")
print(f"Quality pairs: {len(quality_pairs)}")

with open('data/187_pairs_clean.json', 'w') as f:
    json.dump(quality_pairs, f, indent=2)
