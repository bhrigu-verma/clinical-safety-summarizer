import os
import re
import numpy as np
import lightgbm as lgb
from jinja2 import Template

# --- Phase 1: Smart Clinical Templates (Deterministic NLG) ---

# Standard Two-Arm Study with common events and optional SAEs
STANDARD_TEMPLATE = """
An overall summary of treatment-emergent adverse events (TEAEs) is presented below.
Overall, TEAEs were reported for {{ data.total_drug_pct }}% of subjects in the {{ data.drug_name }} arm and {{ data.total_placebo_pct }}% in the control arm.

{% if data.very_common_events %}
The most frequently reported adverse events (≥10% incidence) included {{ data.very_common_events[:-1] | join(', ') }}{% if data.very_common_events | length > 1 %}, and {% endif %}{{ data.very_common_events[-1] }}.
{% endif %}

{% if data.common_events %}
Other notable adverse events included {{ data.common_events[:-1] | join(', ') }}{% if data.common_events | length > 1 %} and {% endif %}{{ data.common_events[-1] }}.
{% endif %}

{% if data.serious_events %}
Serious adverse events (SAEs) or severe toxicity (Grade 3-4) were reported, specifically {{ data.serious_events[:-1] | join(', ') }}{% if data.serious_events | length > 1 %} and {% endif %}{{ data.serious_events[-1] }}.
{% else %}
No serious adverse events (SAEs) or severe Grade 3-4 toxicities were reported during the study period.
{% endif %}
"""

# Highly Imbalanced Arms (Drug significantly higher toxicity than placebo)
IMBALANCED_TEMPLATE = """
Treatment with {{ data.drug_name }} was associated with a higher incidence of treatment-emergent adverse events (TEAEs) compared to the control arm ({{ data.total_drug_pct }}% vs {{ data.total_placebo_pct }}%).

{% if data.serious_events %}
A clinically meaningful difference was observed in severe or serious adverse events, which included {{ data.serious_events[:-1] | join(', ') }}{% if data.serious_events | length > 1 %} and {% endif %}{{ data.serious_events[-1] }}.
{% endif %}

{% if data.very_common_events %}
The most frequent TEAEs driving this difference (≥10%) included {{ data.very_common_events[:-1] | join(', ') }}{% if data.very_common_events | length > 1 %}, and {% endif %}{{ data.very_common_events[-1] }}.
{% endif %}

{% if data.common_events %}
Other common events noted were {{ data.common_events[:-1] | join(', ') }}{% if data.common_events | length > 1 %}, and {% endif %}{{ data.common_events[-1] }}.
{% endif %}
"""

# Single Arm safety profile
SINGLE_ARM_TEMPLATE = """
An overall summary of treatment-emergent adverse events (TEAEs) for the {{ data.drug_name }} safety population is presented below.
Overall, TEAEs were reported for {{ data.total_drug_pct }}% of subjects.

{% if data.very_common_events %}
The most frequent adverse events (≥10% incidence) included {{ data.very_common_events[:-1] | join(', ') }}{% if data.very_common_events | length > 1 %}, and {% endif %}{{ data.very_common_events[-1] }}.
{% endif %}

{% if data.common_events %}
Other reported adverse events included {{ data.common_events[:-1] | join(', ') }}{% if data.common_events | length > 1 %} and {% endif %}{{ data.common_events[-1] }}.
{% endif %}

{% if data.serious_events %}
Serious adverse events (SAEs) or severe Grade 3-4 toxicities included {{ data.serious_events[:-1] | join(', ') }}{% if data.serious_events | length > 1 %} and {% endif %}{{ data.serious_events[-1] }}.
{% else %}
There were no serious adverse events (SAEs) or fatal events reported.
{% endif %}
"""


class ClinicalNLGEngine:
    def __init__(self, model_path: str = 'models/content_selector.pkl'):
        # Fallback to the lightgbm text file if pkl doesn't exist directly (or wait, our script saved pkl)
        # Checking to see which model exists:
        if os.path.exists(model_path):
            import pickle
            with open(model_path, "rb") as f:
                artefacts = pickle.load(f)
                self.model = artefacts["model"]
            self.model_type = "pkl"
        elif os.path.exists('data/lgbm_clinical_content_selector.txt'):
            self.model = lgb.Booster(model_file='data/lgbm_clinical_content_selector.txt')
            self.model_type = "txt"
        else:
            raise FileNotFoundError("ML Weights not found. Run training script first.")
            
        self.templates = {
            "standard": Template(STANDARD_TEMPLATE),
            "imbalanced": Template(IMBALANCED_TEMPLATE),
            "single_arm": Template(SINGLE_ARM_TEMPLATE)
        }

    def _parse_linearized_table(self, table_text: str) -> dict:
        parsed_data = {
            "drug_name": "Experimental Drug",
            "total_drug_pct": 0.0,
            "total_placebo_pct": 0.0,
            "is_single_arm": False,
            "events": []
        }
        
        # 1. Parse Headers (identify if single arm or two arms)
        header_match = re.search(r'\[HEADERS:\s*\|\s*(.*?)\s*\]', table_text)
        if header_match:
            headers = [h.strip() for h in header_match.group(1).split('|') if h.strip()]
            if len(headers) == 1:
                parsed_data["is_single_arm"] = True
                parsed_data["drug_name"] = headers[0].split('(')[0].strip()
            elif len(headers) >= 2:
                # Typically Placebo is index 0, Drug is index 1 or vice versa
                parsed_data["drug_name"] = headers[-1].split('(')[0].strip()
                
        # 2. Parse Rows
        rows_text = re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', table_text)
        
        for raw_row in rows_text:
            parts = [p.strip() for p in raw_row.split('|')]
            if len(parts) >= 2:
                event_name = parts[0]
                
                if parsed_data["is_single_arm"]:
                    drug_match = re.search(r'\(?(\d+\.?\d*)\s*%\)?', parts[-1]) or re.search(r'^(\d+\.?\d*)$', parts[-1])
                    drug_pct = float(drug_match.group(1)) if drug_match else 0.0
                    placebo_pct = 0.0
                else:
                    placebo_match = re.search(r'\(?(\d+\.?\d*)\s*%\)?', parts[1])
                    placebo_pct = float(placebo_match.group(1)) if placebo_match else 0.0
                    
                    drug_match = re.search(r'\(?(\d+\.?\d*)\s*%\)?', parts[-1])
                    drug_pct = float(drug_match.group(1)) if drug_match else 0.0
                
                is_sae = 1 if 'serious' in event_name.lower() or 'sae' in event_name.lower() or 'fatal' in event_name.lower() or 'death' in event_name.lower() else 0
                is_severe = 1 if 'grade 3' in event_name.lower() or 'grade 4' in event_name.lower() or 'severe' in event_name.lower() else 0
                
                # Capture Overall row 
                if 'any teae' in event_name.lower() or 'overall' in event_name.lower() or event_name.lower() == 'teae' or ('subjects with' in event_name.lower() and not is_sae and not is_severe):
                    if parsed_data['total_drug_pct'] == 0:
                        parsed_data['total_drug_pct'] = drug_pct
                        parsed_data['total_placebo_pct'] = placebo_pct
                    continue
                    
                risk_diff = abs(drug_pct - placebo_pct)
                relative_risk = drug_pct / (placebo_pct + 0.001)
                
                parsed_data['events'].append({
                    "name": event_name,
                    "drug_pct": drug_pct,
                    "placebo_pct": placebo_pct,
                    "risk_diff": risk_diff,
                    "relative_risk": relative_risk,
                    "is_sae": is_sae,
                    "is_severe": is_severe
                })
        
        # Fallback if no specific overall row
        if parsed_data["total_drug_pct"] == 0.0 and parsed_data["events"]:
            parsed_data["total_drug_pct"] = max([e["drug_pct"] for e in parsed_data["events"]] + [0.0])
            parsed_data["total_placebo_pct"] = max([e["placebo_pct"] for e in parsed_data["events"]] + [0.0])
            
        return parsed_data

    # --- Phase 2: Advanced Clinical Microplanning (Bucketizing) ---
    def generate(self, table_text: str) -> str:
        # 0. Parse raw text into structured features
        table_data = self._parse_linearized_table(table_text)
        
        # 1. LightGBM Content Selection
        selected_events = []
        for event in table_data.get('events', []):
            # Same 6 features used in Phase 1 setup
            features = [
                event['drug_pct'], event['placebo_pct'], 
                event['risk_diff'], event['relative_risk'], 
                event['is_sae'], event['is_severe']
            ]
            
            # Regulatory Guardrail Override (Always Select SAEs and Severe)
            if event['is_sae'] == 1 or event['is_severe'] == 1:
                selected_events.append(event)
            else:
                if self.model_type == "txt":
                    prob = self.model.predict([features])[0]
                else: 
                    # If using sklearn wrapper for LGBMClassifier
                    try:
                        prob = self.model.predict_proba(np.array(features).reshape(1, -1))[0][1]
                    except:
                        # Fallback heuristic if it's not a standard LGBMClassifier with predict_proba
                        prob = 1.0 if event['drug_pct'] >= 5.0 else 0.0
                
                if prob > 0.4:  # Slightly aggressive threshold for better ROUGE recall
                    selected_events.append(event)

        if not selected_events:
            return "No clinically significant or severe adverse events were identified during the reporting period."

        # 2. Stratification Bucketing (Advanced Microplanning)
        very_common_events_text = []
        common_events_text = []
        serious_events_text = []
        
        total_risk_diff = 0.0
        
        for event in selected_events:
            total_risk_diff += event['risk_diff']
            
            if table_data["is_single_arm"]:
                formatted_text = f"{event['name']} ({event['drug_pct']}%)"
            else:
                formatted_text = f"{event['name']} ({event['drug_pct']}% vs {event['placebo_pct']}%)"
                
            if event['is_sae'] == 1 or event['is_severe'] == 1:
                serious_events_text.append(formatted_text)
            elif event['drug_pct'] >= 10.0:
                very_common_events_text.append(formatted_text)
            else:
                common_events_text.append(formatted_text)

        # 3. Dynamic Template Routing (Selects structure based on data)
        if table_data["is_single_arm"]:
            selected_template = self.templates["single_arm"]
        elif total_risk_diff > 15.0 or (table_data["total_drug_pct"] - table_data["total_placebo_pct"]) > 10.0:
            selected_template = self.templates["imbalanced"]
        else:
            selected_template = self.templates["standard"]
            
        # 4. Final Surface Realization
        payload = {
            "drug_name": table_data['drug_name'],
            "total_drug_pct": table_data['total_drug_pct'],
            "total_placebo_pct": table_data['total_placebo_pct'],
            "very_common_events": very_common_events_text,
            "common_events": common_events_text,
            "serious_events": serious_events_text
        }
        
        generated_text = selected_template.render(data=payload)
        return ' '.join(generated_text.split()) # Clean excess whitespace

