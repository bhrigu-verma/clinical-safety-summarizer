"""
Stratified splitting for clinical summarization data.
Ensures all table types (AE, demographics, efficacy) appear in each split.
"""
from sklearn.model_selection import train_test_split
import json

def create_splits(examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create stratified splits by table type"""
    
    # Tag examples by table type (detected from content)
    def get_table_type(example):
        inp = example['input'].lower()
        if 'adverse event' in inp or 'teae' in inp or 'sae' in inp:
            return 'adverse_event'
        elif 'demographic' in inp or 'age' in inp or 'gender' in inp:
            return 'demographics'
        elif 'efficacy' in inp or 'response' in inp or 'survival' in inp:
            return 'efficacy'
        else:
            return 'other'
    
    for ex in examples:
        ex['table_type'] = get_table_type(ex)
    
    # Stratified split
    labels = [ex['table_type'] for ex in examples]
    
    train_val, test = train_test_split(
        examples, test_size=test_ratio, stratify=labels, random_state=seed
    )
    
    train_val_labels = [ex['table_type'] for ex in train_val]
    train, val = train_test_split(
        train_val, 
        test_size=val_ratio/(train_ratio + val_ratio),
        stratify=train_val_labels,
        random_state=seed
    )
    
    print(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test
