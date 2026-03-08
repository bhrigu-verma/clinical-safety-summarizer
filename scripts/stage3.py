from jinja2 import Template

print("--- Stage 3: Clinical Surface Realization Engine ---\n")

# 1. The Output from Stage 2 (Your grouped clusters)
# We add a bit of metadata (Drug Name, Total N-counts) that the template needs.
clinical_payload = {
    "drug_name": "Apalutamide",
    "total_drug_pct": 96.5,
    "total_placebo_pct": 93.2,
    "common_events": ["Nausea (12.5% vs 4.1%)", "Headache (11.0% vs 3.2%)", "Dizziness (10.5% vs 2.1%)"],
    "serious_events": ["Cardiac Arrest (0.5% vs 0.1%)", "Ischemic Stroke (0.4% vs 0.0%)"]
}

# 2. The Deterministic Grammar Template (Zero Hallucination Guarantee)
# This template uses conditional logic ({% if %}) to dynamically build the paragraph 
# based on what the ML passed to it.
CSR_TEMPLATE = """
An overall summary of treatment-emergent adverse events (TEAEs) is presented below. 
Overall, TEAEs were reported for {{ data.total_drug_pct }}% of subjects in the {{ data.drug_name }} arm and {{ data.total_placebo_pct }}% in the placebo arm.

{% if data.common_events %}
The most notable adverse events included {{ data.common_events[:-1] | join(', ') }}{% if data.common_events | length > 1 %}, and {% endif %}{{ data.common_events[-1] }}.
{% endif %}

{% if data.serious_events %}
Serious adverse events (SAEs) were reported, specifically {{ data.serious_events[:-1] | join(', ') }}{% if data.serious_events | length > 1 %} and {% endif %}{{ data.serious_events[-1] }}.
{% else %}
No serious adverse events (SAEs) were reported during the study period.
{% endif %}
"""

# 3. Generate the Text
template = Template(CSR_TEMPLATE)
generated_text = template.render(data=clinical_payload)

# 4. Post-processing (Clean up any weird spacing caused by the template engine)
final_text = ' '.join(generated_text.split())

print("=== FINAL GENERATED CLINICAL NARRATIVE ===")
print(final_text)