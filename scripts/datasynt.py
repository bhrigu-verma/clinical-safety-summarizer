import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
print("Script is starting...") # Add this line

# ==========================================
# 1. SYNTHETIC GENERATOR: RULE-BASED
# ==========================================
def generate_synthetic_insignificant_events(num_samples: int) -> pd.DataFrame:
    """
    Generates realistic 'noise' data (Label 0). 
    These are adverse events that are too small or statistically 
    insignificant to be mentioned in a regulatory clinical summary.
    """
    np.random.seed(42)
    
    # 1. Generate low incidence rates (0.1% to 4.9%)
    placebo_pct = np.random.uniform(0.1, 4.9, num_samples)
    
    # 2. Generate drug incidence rates very close to placebo (Risk Diff < 2%)
    # Sometimes drug is slightly higher, sometimes slightly lower
    risk_diff = np.random.uniform(-1.5, 1.5, num_samples)
    drug_pct = np.clip(placebo_pct + risk_diff, 0.1, None) 
    
    # 3. Calculate Relative Risk
    relative_risk = drug_pct / (placebo_pct + 0.001)
    
    # 4. Insignificant events are NEVER SAEs and NEVER Grade 3-4
    is_sae = np.zeros(num_samples, dtype=int)
    is_severe = np.zeros(num_samples, dtype=int)
    
    # 5. Label is 0 (Do not mention in summary)
    labels = np.zeros(num_samples, dtype=int)
    
    synthetic_df = pd.DataFrame({
        'drug_pct': np.round(drug_pct, 1),
        'placebo_pct': np.round(placebo_pct, 1),
        'risk_diff': np.round(risk_diff, 1),
        'relative_risk': np.round(relative_risk, 2),
        'is_sae': is_sae,
        'is_severe': is_severe,
        'label': labels
    })
    
    return synthetic_df

# ==========================================
# 2. MOCKING YOUR REAL EXTRACTED DATA
# ==========================================
def get_real_imbalanced_data() -> pd.DataFrame:
    """
    Simulates the data you will extract from your 200 Word Tables.
    As discussed, 95% of the extracted rows will be Label 1 (Salient)
    because human writers only put important stuff in Summary Tables.
    """
    # 950 Important Events (Label 1)
    salient_df = pd.DataFrame({
        'drug_pct': np.random.uniform(10.0, 95.0, 950),
        'placebo_pct': np.random.uniform(5.0, 80.0, 950),
        'risk_diff': np.random.uniform(5.0, 25.0, 950),
        'relative_risk': np.random.uniform(1.5, 5.0, 950),
        'is_sae': np.random.choice([0, 1], 950, p=[0.7, 0.3]),
        'is_severe': np.random.choice([0, 1], 950, p=[0.6, 0.4]),
        'label': 1
    })
    
    # 50 Unimportant Events that accidentally made it into tables (Label 0)
    minor_df = pd.DataFrame({
        'drug_pct': np.random.uniform(1.0, 3.0, 50),
        'placebo_pct': np.random.uniform(1.0, 3.0, 50),
        'risk_diff': np.random.uniform(0.0, 1.0, 50),
        'relative_risk': np.random.uniform(0.9, 1.1, 50),
        'is_sae': 0,
        'is_severe': 0,
        'label': 0
    })
    
    return pd.concat([salient_df, minor_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# 3. PIPELINE INTEGRATION & TRAINING
# ==========================================
def train_lightgbm_pipeline():
    print("--- Starting Clinical ML Training Pipeline ---")
    
    # 1. Load the heavily imbalanced "Real" data
    real_df = get_real_imbalanced_data()
    print(f"\n[Original Data Distribution]\n{real_df['label'].value_counts()}")
    
    # 2. Inject Rule-Based Synthetic Data
    # We add 900 synthetic '0' labels to balance the 950 '1' labels
    synthetic_df = generate_synthetic_insignificant_events(num_samples=900)
    combined_df = pd.concat([real_df, synthetic_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n[Data Distribution After Rule-Based Synthesis]\n{combined_df['label'].value_counts()}")
    
    # Split features and labels
    features = ['drug_pct', 'placebo_pct', 'risk_diff', 'relative_risk', 'is_sae', 'is_severe']
    X = combined_df[features]
    y = combined_df['label']
    
    # 3. Apply SMOTE (Mathematical Synthesis)
    # This smooths out any harsh boundaries created by the rule-based generator
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"\n[Final Data Distribution After SMOTE]\n{y_balanced.value_counts()}")
    
    # 4. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    # 5. Train LightGBM Model
    print("\n--- Training LightGBM Content Selector ---")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=150,
        callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.log_evaluation(period=0)]
    )
    
    # 6. Evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("\n--- Final Model Performance ---")
    print(classification_report(y_test, y_pred, target_names=["Ignore (0)", "Mention in Text (1)"]))
    
    # 7. Save the model for the FastAPI backend
    model.save_model('lgbm_clinical_content_selector.txt')
    print("Model saved to 'lgbm_clinical_content_selector.txt'")

if __name__ == "__main__":
    train_lightgbm_pipeline()