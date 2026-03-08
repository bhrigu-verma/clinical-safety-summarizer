"""
src/generation/template_clusters.py  — v4.0 (10/10 rewrite)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key changes vs v3:
  - ALL pairwise stats embed explicit arm labels inline:
    "{TEAE_N_B} ({TEAE_PCT_B}%) of subjects in the {ARM_B} arm"
    NOT just "{TEAE_PCT_B}% vs {TEAE_PCT_A}%"

  - ARM_A = control arm, ARM_B = experimental arm (drug).
    These are filled by ArmParser from actual HEADERS, not hardcoded.

  - Added C14_individual_ae_listing cluster for tables with named
    individual AEs (Nausea, Rash, Dizziness etc). These tables go
    through IndividualAERenderer, not slot-fill templates.

  - Updated select_cluster() logic to route individual-AE tables to C14.
"""
from typing import Dict, List


TEMPLATE_CLUSTERS: Dict[str, Dict] = {

    # ── C01: Standard 2-arm overview (most common) ────────────────────────────
    "C01_standard_two_arm_overall": {
        "description": "Overall TEAE summary with Grade 3-4, SAE, and discontinuation",
        "trigger_conditions": {
            "arm_count": 2, "has_grade34": True, "has_sae": True, "has_discontinuation": True,
        },
        "template": (
            "Treatment-emergent adverse events (TEAEs) were reported for "
            "{TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects in the {ARM_B} arm and "
            "{TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects in the {ARM_A} arm. "
            "Grade 3–4 TEAEs were reported for {G34_PCT_B}% of subjects in the {ARM_B} arm and "
            "{G34_PCT_A}% of subjects in the {ARM_A} arm. "
            "Treatment-emergent SAEs were reported for {SAE_N_B} ({SAE_PCT_B}%) of subjects "
            "in the {ARM_B} arm and {SAE_N_A} ({SAE_PCT_A}%) of subjects in the {ARM_A} arm. "
            "TEAEs leading to treatment discontinuation were reported for {DISC_N_B} ({DISC_PCT_B}%) "
            "of subjects in the {ARM_B} arm and {DISC_N_A} ({DISC_PCT_A}%) in the {ARM_A} arm."
        ),
        "required_slots": [
            "ARM_A", "ARM_B", "N_A", "N_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
            "SAE_N_A", "SAE_PCT_A", "SAE_N_B", "SAE_PCT_B",
            "DISC_N_A", "DISC_PCT_A", "DISC_N_B", "DISC_PCT_B",
        ],
    },

    # ── C02: Drug-related emphasis ─────────────────────────────────────────────
    "C02_drug_related_emphasis": {
        "description": "Emphasises drug-related AEs in all categories",
        "trigger_conditions": {
            "arm_count": 2, "has_grade34": True, "sae_drug_related_flag": True,
        },
        "template": (
            "TEAEs were reported for {TEAE_N_B} ({TEAE_PCT_B}%) of subjects in the {ARM_B} arm and "
            "{TEAE_N_A} ({TEAE_PCT_A}%) of subjects in the {ARM_A} arm; "
            "drug-related TEAEs were reported for {DR_TEAE_PCT_B}% ({ARM_B}) and "
            "{DR_TEAE_PCT_A}% ({ARM_A}), respectively. "
            "Grade 3–4 TEAEs occurred in {G34_PCT_B}% of {ARM_B} subjects and "
            "{G34_PCT_A}% of {ARM_A} subjects. "
            "Drug-related SAEs were reported for {DR_SAE_PCT_B}% of subjects in the {ARM_B} arm "
            "and {DR_SAE_PCT_A}% in the {ARM_A} arm. "
            "Drug-related TEAEs leading to discontinuation were reported for "
            "{DR_DISC_PCT_B}% ({ARM_B}) and {DR_DISC_PCT_A}% ({ARM_A}), respectively."
        ),
        "required_slots": [
            "ARM_A", "ARM_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
            "DR_TEAE_PCT_A", "DR_TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
            "DR_SAE_PCT_A", "DR_SAE_PCT_B",
            "DR_DISC_PCT_A", "DR_DISC_PCT_B",
        ],
    },

    # ── C03: High incidence (>95%) both arms ───────────────────────────────────
    "C03_high_incidence_both_arms": {
        "description": "When TEAE incidence >95% in both arms",
        "trigger_conditions": {"has_high_teae": True, "arm_count": 2},
        "template": (
            "The incidence of TEAEs was high in both arms: "
            "{TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects in the {ARM_B} arm and "
            "{TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects in the {ARM_A} arm. "
            "Grade 3–4 TEAEs were experienced by {G34_PCT_B}% of {ARM_B} subjects "
            "compared with {G34_PCT_A}% of {ARM_A} subjects. "
            "The rate of serious adverse events was {SAE_PCT_B}% in the {ARM_B} arm "
            "and {SAE_PCT_A}% in the {ARM_A} arm. "
            "Discontinuation due to TEAEs was reported in {DISC_PCT_B}% of {ARM_B} subjects "
            "and {DISC_PCT_A}% of {ARM_A} subjects."
        ),
        "required_slots": [
            "ARM_A", "ARM_B", "N_A", "N_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
            "SAE_PCT_A", "SAE_PCT_B",
            "DISC_PCT_A", "DISC_PCT_B",
        ],
    },

    # ── C04: SAE and fatal emphasis ────────────────────────────────────────────
    "C04_sae_fatal_emphasis": {
        "description": "Leads with SAE/fatal events, overall TEAE secondary",
        "trigger_conditions": {"has_death": True, "has_sae": True},
        "template": (
            "Overall, TEAEs were reported for {TEAE_PCT_B}% of subjects in the {ARM_B} arm "
            "and {TEAE_PCT_A}% in the {ARM_A} arm. "
            "Serious adverse events were reported for {SAE_N_B} ({SAE_PCT_B}%) of subjects "
            "in the {ARM_B} arm versus {SAE_N_A} ({SAE_PCT_A}%) in the {ARM_A} arm. "
            "Fatal TEAEs were reported for {FATAL_PCT_B}% of {ARM_B} subjects "
            "and {FATAL_PCT_A}% of {ARM_A} subjects. "
            "Grade 3–4 TEAEs were reported for {G34_PCT_B}% ({ARM_B}) "
            "and {G34_PCT_A}% ({ARM_A})."
        ),
        "required_slots": [
            "ARM_A", "ARM_B",
            "TEAE_PCT_A", "TEAE_PCT_B",
            "SAE_N_A", "SAE_PCT_A", "SAE_N_B", "SAE_PCT_B",
            "FATAL_PCT_A", "FATAL_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
        ],
    },

    # ── C05: No grade3-4, SAE only ────────────────────────────────────────────
    "C05_no_grade34_sae_only": {
        "description": "Tables without Grade 3-4 breakdown, only TEAE + SAE + DISC",
        "trigger_conditions": {"has_grade34": False, "has_sae": True},
        "template": (
            "TEAEs were reported for {TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects "
            "in the {ARM_B} arm and {TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects "
            "in the {ARM_A} arm. "
            "Treatment-emergent serious adverse events were reported for "
            "{SAE_N_B} ({SAE_PCT_B}%) of subjects in the {ARM_B} arm "
            "versus {SAE_N_A} ({SAE_PCT_A}%) in the {ARM_A} arm. "
            "TEAEs leading to discontinuation were reported for {DISC_PCT_B}% ({ARM_B}) "
            "and {DISC_PCT_A}% ({ARM_A})."
        ),
        "required_slots": [
            "ARM_A", "ARM_B", "N_A", "N_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
            "SAE_N_A", "SAE_PCT_A", "SAE_N_B", "SAE_PCT_B",
            "DISC_PCT_A", "DISC_PCT_B",
        ],
    },

    # ── C06: Open-label phase summary ─────────────────────────────────────────
    "C06_open_label_summary": {
        "description": "Open-label extension phase safety summary",
        "trigger_conditions": {"has_dual_phase": True},
        "template": (
            "In the open-label phase, TEAEs were reported for {OL_TEAE_PCT_B}% of "
            "subjects in the {ARM_B} arm and {OL_TEAE_PCT_A}% in the {ARM_A} arm. "
            "Overall (double-blind + open-label), TEAEs were reported for "
            "{TEAE_PCT_B}% ({ARM_B}) and {TEAE_PCT_A}% ({ARM_A}). "
            "Grade 3–4 TEAEs were reported for {G34_PCT_B}% of {ARM_B} subjects "
            "and {G34_PCT_A}% of {ARM_A} subjects. "
            "SAEs were reported for {SAE_PCT_B}% ({ARM_B}) and {SAE_PCT_A}% ({ARM_A})."
        ),
        "required_slots": [
            "ARM_A", "ARM_B",
            "OL_TEAE_PCT_A", "OL_TEAE_PCT_B",
            "TEAE_PCT_A", "TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
            "SAE_PCT_A", "SAE_PCT_B",
        ],
    },

    # ── C07: Minimal table (TEAE + DISC only) ─────────────────────────────────
    "C07_minimal_teae_disc": {
        "description": "Minimal table — TEAE incidence and discontinuation only",
        "trigger_conditions": {"has_grade34": False, "has_sae": False, "has_discontinuation": True},
        "template": (
            "TEAEs were reported for {TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects "
            "in the {ARM_B} arm and {TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects "
            "in the {ARM_A} arm. "
            "TEAEs leading to treatment discontinuation were reported for "
            "{DISC_PCT_B}% of {ARM_B} subjects and {DISC_PCT_A}% of {ARM_A} subjects."
        ),
        "required_slots": [
            "ARM_A", "ARM_B", "N_A", "N_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
            "DISC_PCT_A", "DISC_PCT_B",
        ],
    },

    # ── C08: Drug-related full profile ────────────────────────────────────────
    "C08_drug_related_full": {
        "description": "All categories have drug-related sub-rows",
        "trigger_conditions": {"sae_drug_related_flag": True, "has_discontinuation": True},
        "template": (
            "Overall TEAEs were reported for {TEAE_PCT_B}% of {ARM_B} subjects "
            "and {TEAE_PCT_A}% of {ARM_A} subjects; "
            "drug-related TEAEs were reported in {DR_TEAE_PCT_B}% and {DR_TEAE_PCT_A}%, respectively. "
            "Grade 3–4 TEAEs were reported for {G34_PCT_B}% of {ARM_B} subjects "
            "and {G34_PCT_A}% of {ARM_A} subjects; "
            "drug-related grade 3–4 TEAEs in {DR_G34_PCT_B}% and {DR_G34_PCT_A}%, respectively. "
            "SAEs were reported in {SAE_PCT_B}% ({ARM_B}) and {SAE_PCT_A}% ({ARM_A}). "
            "Drug-related SAEs were reported in {DR_SAE_PCT_B}% and {DR_SAE_PCT_A}%. "
            "TEAEs leading to discontinuation: {DISC_PCT_B}% ({ARM_B}), "
            "{DISC_PCT_A}% ({ARM_A}); drug-related: {DR_DISC_PCT_B}% and {DR_DISC_PCT_A}%."
        ),
        "required_slots": [
            "ARM_A", "ARM_B",
            "TEAE_PCT_A", "TEAE_PCT_B", "DR_TEAE_PCT_A", "DR_TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B", "DR_G34_PCT_A", "DR_G34_PCT_B",
            "SAE_PCT_A", "SAE_PCT_B", "DR_SAE_PCT_A", "DR_SAE_PCT_B",
            "DISC_PCT_A", "DISC_PCT_B", "DR_DISC_PCT_A", "DR_DISC_PCT_B",
        ],
    },

    # ── C09: Safety population note ───────────────────────────────────────────
    "C09_safety_population_note": {
        "description": "Table starts with safety population summary row",
        "trigger_conditions": {"has_headers": True},
        "template": (
            "The safety population comprised {N_B} subjects in the {ARM_B} arm and "
            "{N_A} subjects in the {ARM_A} arm. "
            "TEAEs were reported for {TEAE_N_B} ({TEAE_PCT_B}%) of {ARM_B} subjects "
            "and {TEAE_N_A} ({TEAE_PCT_A}%) of {ARM_A} subjects. "
            "Grade 3–4 TEAEs were reported for {G34_PCT_B}% ({ARM_B}) and "
            "{G34_PCT_A}% ({ARM_A}). "
            "SAEs were reported for {SAE_PCT_B}% ({ARM_B}) and {SAE_PCT_A}% ({ARM_A}). "
            "TEAEs leading to discontinuation: {DISC_PCT_B}% ({ARM_B}), "
            "{DISC_PCT_A}% ({ARM_A})."
        ),
        "required_slots": [
            "ARM_A", "ARM_B", "N_A", "N_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
            "SAE_PCT_A", "SAE_PCT_B",
            "DISC_PCT_A", "DISC_PCT_B",
        ],
    },

    # ── C10: TEAE only (no SAE, grade, or disc data) ──────────────────────────
    "C10_teae_only": {
        "description": "Table contains only TEAE incidence — no subcategories",
        "trigger_conditions": {"has_sae": False, "has_grade34": False, "has_discontinuation": False},
        "template": (
            "Treatment-emergent adverse events were reported for "
            "{TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects in the {ARM_B} arm "
            "and {TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects in the {ARM_A} arm."
        ),
        "required_slots": [
            "ARM_A", "ARM_B", "N_A", "N_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
        ],
    },

    # ── C11: Grade 3-4 only (no disc data) ────────────────────────────────────
    "C11_grade34_no_disc": {
        "description": "Table with TEAE, Grade 3-4, SAE but no discontinuation",
        "trigger_conditions": {"has_grade34": True, "has_sae": True, "has_discontinuation": False},
        "template": (
            "TEAEs were reported for {TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects "
            "in the {ARM_B} arm and {TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects "
            "in the {ARM_A} arm. "
            "Grade 3–4 TEAEs were reported for {G34_PCT_B}% of {ARM_B} subjects "
            "and {G34_PCT_A}% of {ARM_A} subjects. "
            "Treatment-emergent SAEs were reported for {SAE_PCT_B}% ({ARM_B}) "
            "and {SAE_PCT_A}% ({ARM_A})."
        ),
        "required_slots": [
            "ARM_A", "ARM_B", "N_A", "N_B",
            "TEAE_N_A", "TEAE_PCT_A", "TEAE_N_B", "TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
            "SAE_PCT_A", "SAE_PCT_B",
        ],
    },

    # ── C12: High discontinuation flag ────────────────────────────────────────
    "C12_high_discontinuation": {
        "description": "Discontinuation rate notably elevated (>15%) in one arm",
        "trigger_conditions": {"has_discontinuation": True, "discontinuation_delta": True},
        "template": (
            "TEAEs were reported for {TEAE_PCT_B}% of subjects in the {ARM_B} arm "
            "and {TEAE_PCT_A}% in the {ARM_A} arm. "
            "Grade 3–4 TEAEs were reported in {G34_PCT_B}% ({ARM_B}) and {G34_PCT_A}% ({ARM_A}). "
            "SAEs were reported for {SAE_PCT_B}% ({ARM_B}) and {SAE_PCT_A}% ({ARM_A}). "
            "Notably, TEAEs leading to discontinuation were reported for "
            "{DISC_N_B} ({DISC_PCT_B}%) of {ARM_B} subjects versus "
            "{DISC_N_A} ({DISC_PCT_A}%) of {ARM_A} subjects."
        ),
        "required_slots": [
            "ARM_A", "ARM_B",
            "TEAE_PCT_A", "TEAE_PCT_B",
            "G34_PCT_A", "G34_PCT_B",
            "SAE_PCT_A", "SAE_PCT_B",
            "DISC_N_A", "DISC_PCT_A", "DISC_N_B", "DISC_PCT_B",
        ],
    },

    # ── C13: Single-arm study ─────────────────────────────────────────────────
    "C13_single_arm": {
        "description": "Single arm — no comparator column",
        "trigger_conditions": {"arm_count": 1},
        "template": (
            "TEAEs were reported for {TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects "
            "in the {ARM_B} arm. "
            "Grade 3–4 TEAEs were reported for {G34_PCT_B}% of subjects. "
            "SAEs were reported for {SAE_PCT_B}% of subjects. "
            "TEAEs leading to discontinuation were reported for {DISC_PCT_B}% of subjects."
        ),
        "required_slots": [
            "ARM_B", "N_B",
            "TEAE_N_B", "TEAE_PCT_B",
            "G34_PCT_B", "SAE_PCT_B", "DISC_PCT_B",
        ],
    },

    # ── C14: Individual AE listing (NEW v4) ───────────────────────────────────
    # For tables that list named individual AEs (Nausea, Rash, Dizziness, etc.)
    # rather than overview categories (TEAE total, SAE, Grade 3-4, Disc).
    # The template provides the HEADER only; individual AE sentences are
    # appended by IndividualAERenderer. SlotFillGenerator routes to this
    # cluster when the table has no TEAE_PCT_B slot extractable.
    "C14_individual_ae_listing": {
        "description": "Individual named AE listing table (>5% incidence cutoff)",
        "trigger_conditions": {"individual_ae_table": True},
        "template": (
            "The following treatment-emergent adverse events were reported at an "
            "incidence of >5% in either arm or with a notable between-arm difference, "
            "as observed in the {ARM_B} and {ARM_A} arms."
        ),
        "required_slots": ["ARM_A", "ARM_B"],
    },

}


# ── Cluster Selection Logic ────────────────────────────────────────────────────

def select_cluster(features: dict) -> str:
    """
    Rule-based cluster selection from table-level features.

    Priority order matters — more specific conditions evaluated first.
    """
    has_grade34   = bool(features.get('has_grade34'))
    has_sae       = bool(features.get('has_sae'))
    has_disc      = bool(features.get('has_discontinuation'))
    has_death     = bool(features.get('has_death'))
    has_high_teae = bool(features.get('has_high_teae'))
    has_dual      = bool(features.get('has_dual_phase'))
    sae_dr_flag   = bool(features.get('sae_drug_related_flag'))
    disc_delta    = float(features.get('discontinuation_delta', 0))
    arm_count     = int(features.get('num_arms', 2))
    num_rows      = int(features.get('num_rows', 0))

    # Individual AE listing: many rows (>4), no standard overview keywords detected
    # (TEAE_PCT_B would be None after slot extraction — checked separately)
    # Use as a fallback routing; explicit signal = high num_rows + no sae + no disc
    individual_ae_signal = (num_rows >= 4 and not has_sae and not has_disc)

    if arm_count == 1:
        return "C13_single_arm"
    if has_dual:
        return "C06_open_label_summary"
    if has_death and has_sae:
        return "C04_sae_fatal_emphasis"
    if sae_dr_flag and has_disc:
        return "C08_drug_related_full"
    if sae_dr_flag:
        return "C02_drug_related_emphasis"
    if has_high_teae and has_grade34 and has_sae and has_disc:
        return "C03_high_incidence_both_arms"
    if has_grade34 and has_sae and has_disc:
        return "C01_standard_two_arm_overall"
    if has_grade34 and has_sae and not has_disc:
        return "C11_grade34_no_disc"
    if not has_grade34 and has_sae and has_disc:
        return "C05_no_grade34_sae_only"
    if has_disc and disc_delta > 10:
        return "C12_high_discontinuation"
    if has_disc and not has_sae and not has_grade34:
        return "C07_minimal_teae_disc"
    if not has_sae and not has_grade34 and not has_disc:
        return "C14_individual_ae_listing" if individual_ae_signal else "C10_teae_only"
    # Default
    return "C09_safety_population_note"


def get_template(cluster_id: str) -> str:
    return TEMPLATE_CLUSTERS[cluster_id]["template"]


def get_required_slots(cluster_id: str) -> List[str]:
    return TEMPLATE_CLUSTERS[cluster_id]["required_slots"]
