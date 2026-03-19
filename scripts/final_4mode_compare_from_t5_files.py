import json
from pathlib import Path
import sys
import pandas as pd

root = Path('/Users/bhriguverma/medicalproject')
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from metrics_suite import ClinicalEvaluationSuite

pred_path = root / 'data' / 't5xxl_predictions_tier1.csv'
gold_path = root / 'data' / 'tier1_gold (1).jsonl'
out_dir = root / 'data' / 'eval_results'
out_dir.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(pred_path)
rows = []
with gold_path.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

gold_df = pd.DataFrame(rows)
required_gold = ['table_id', 'table_text', 'reference']
missing_gold = [c for c in required_gold if c not in gold_df.columns]
if missing_gold:
    raise RuntimeError(f'Missing columns in gold file: {missing_gold}')

required_pred = ['table_id', 'summary_t5xxl']
missing_pred_cols = [c for c in required_pred if c not in pred.columns]
if missing_pred_cols:
    raise RuntimeError(f'Missing columns in prediction CSV: {missing_pred_cols}')

pred = pred[['table_id', 'summary_t5xxl'] + ([c for c in ['latency_ms'] if c in pred.columns])].copy()
pred['table_id'] = pred['table_id'].astype(str)
gold_df['table_id'] = gold_df['table_id'].astype(str)
pred = pred.drop_duplicates(subset=['table_id'], keep='last')

merged = gold_df.merge(pred, on='table_id', how='left', suffixes=('_gold', '_pred'))
pred_summary_col = 'summary_t5xxl_pred' if 'summary_t5xxl_pred' in merged.columns else 'summary_t5xxl'
if pred_summary_col not in merged.columns:
    raise RuntimeError(f'Cannot find predicted summary column after merge. Columns: {list(merged.columns)}')

missing_pred = int(merged[pred_summary_col].isna().sum())
empty_pred = int((merged[pred_summary_col].fillna('').str.strip() == '').sum())

subset = merged[merged[pred_summary_col].fillna('').str.strip() != ''].copy()
if subset.empty:
    raise RuntimeError('No non-empty summary_t5xxl rows available for evaluation')

suite = ClinicalEvaluationSuite(bertscore_model='distilbert-base-uncased', numeric_tolerance=0.01)
inputs = []
for _, r in subset.iterrows():
    arm_names = r.get('arm_names', [])
    if not isinstance(arm_names, list):
        arm_names = []
    inputs.append({
        'generated': r[pred_summary_col],
        'reference': r['reference'],
        'source_table': r['table_text'],
        'arm_names': arm_names,
    })

results = suite.evaluate_batch(inputs, show_progress=False)
agg = suite.aggregate_results(results)
ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

summary_t5 = pd.DataFrame([{
    'run_id': f'eval_tier1_t5xxl_{ts}',
    'tier': 'tier1',
    'profile_name': 'full_system',
    'mode': 't5xxl',
    'n_examples': len(results),
    'latency_ms_mean': float(pred['latency_ms'].mean()) if 'latency_ms' in pred.columns else 0.0,
    'latency_ms_p95': float(pred['latency_ms'].quantile(0.95)) if 'latency_ms' in pred.columns else 0.0,
    'rouge_l_mean': float(agg['rouge_l']['mean']),
    'bertscore_f1_mean': float(agg['bertscore_f1']['mean']),
    'meteor_mean': float(agg['meteor']['mean']),
    'fk_grade_mean': float(agg['flesch_kincaid_grade']['mean']),
    'nar_mean': float(agg['numeric_accuracy_rate']['mean']),
    'hr_mean': float(agg['hallucination_rate']['mean']),
    'or_mean': float(agg['omission_rate']['mean']),
    'swos_mean': float(agg['severity_weighted_omission_score']['mean']),
    'ndi_mean': float(agg['numeric_drift_index']['mean']),
    'acr_mean': float(agg['arm_confusion_rate']['mean']),
    'rdi_mean': float(agg['risk_deflation_index']['mean']),
    'rii_mean': float(agg['risk_inflation_index']['mean']),
    'dpe_mean': float(agg['delta_preservation_error']['mean']),
    'cor_mean': float(agg['contraindication_omission_rate']['mean']),
    'safety_score_mean': float(agg['safety_score']['mean']),
    'fluency_score_mean': float(agg['fluency_score']['mean']),
}])

summary_t5_path = out_dir / f'summary_tier1_t5xxl_{ts}.csv'
summary_t5.to_csv(summary_t5_path, index=False)

best = None
for f in sorted(out_dir.glob('summary_tier1_*.csv')):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if {'profile_name', 'mode'}.issubset(d.columns):
        sub = d[d['profile_name'].fillna('full_system') == 'full_system']
        modes = set(sub['mode'].astype(str))
        if {'ml', 'dl_base', 'finetuned'}.issubset(modes):
            best = (f, sub[sub['mode'].isin(['ml', 'dl_base', 'finetuned'])].copy())

if best is None:
    raise RuntimeError('No baseline summary_tier1 with ml/dl_base/finetuned found')

base_path, base_df = best
final4 = pd.concat([base_df, summary_t5], ignore_index=True)
final4 = final4[final4['mode'].isin(['ml', 'dl_base', 'finetuned', 't5xxl'])].copy()
final4_path = out_dir / f'final_4mode_comparison_tier1_{ts}.csv'
final4.to_csv(final4_path, index=False)

print('PRED_ROWS', len(pred))
print('GOLD_ROWS', len(gold_df))
print('MISSING_PRED', missing_pred)
print('EMPTY_PRED', empty_pred)
print('BASELINE_USED', base_path.name)
print('SUMMARY_T5', summary_t5_path)
print('FINAL4', final4_path)
print('\nFINAL4_KEY_METRICS')
print(final4[['mode', 'n_examples', 'nar_mean', 'hr_mean', 'acr_mean', 'safety_score_mean', 'fluency_score_mean', 'fk_grade_mean']].sort_values('mode').to_string(index=False))
