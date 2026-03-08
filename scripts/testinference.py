import lightgbm as lgb
import os

print("--- Clinical Inference Engine V2 ---")

model_path = 'lgbm_clinical_content_selector.txt'
if not os.path.exists(model_path):
    print(f"ERROR: Could not find '{model_path}'")
    exit()

model = lgb.Booster(model_file=model_path)

def clinical_decision_engine(features):
    """
    Features order: ['drug_pct', 'placebo_pct', 'risk_diff', 'relative_risk', 'is_sae', 'is_severe']
    """
    is_sae = features[4]
    is_severe = features[5]
    
    # 1. THE HEURISTIC OVERRIDE (Regulatory Compliance)
    if is_sae == 1 or is_severe == 1:
        return 1.0, "MENTION IN TEXT (Forced by Regulatory Rule: SAE/Severe)"
        
    # 2. THE MACHINE LEARNING FALLBACK (For standard events)
    prob = model.predict([features])[0]
    if prob > 0.5:
        return prob, "MENTION IN TEXT (Selected by ML Model)"
    else:
        return prob, "IGNORE (Filtered by ML Model)"


# --- Test Scenarios ---
row_A = [2.1, 0.5, 1.6, 4.2, 1, 1]  # Rare SAE
row_B = [1.2, 1.1, 0.1, 1.09, 0, 0] # Minor Noise
row_C = [45.0, 15.0, 30.0, 3.0, 0, 0] # Common Mild Event

prob_A, decision_A = clinical_decision_engine(row_A)
prob_B, decision_B = clinical_decision_engine(row_B)
prob_C, decision_C = clinical_decision_engine(row_C)

print("\n--- Final Clinical Decisions ---")
print(f"Scenario A (Rare SAE):       {decision_A}")
print(f"Scenario B (Minor Noise):    {decision_B}")
print(f"Scenario C (Common Event):   {decision_C}")