"""
src/generation/paraphrase_families.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-variant sentence templates for each semantic move in clinical narratives.

Each variant is a dict with:
  - 'text': template string with {SLOT} placeholders
  - 'slots': list of required slot names
  - 'conditions': optional dict of data conditions for applicability

Selection is data-driven: the SentenceRealizer picks the best variant
based on what slots are available and the data profile (high incidence,
similar arms, imbalanced, etc.)

These variants are derived from analysis of 511 gold narrative sentences
from 191 clinical writeups, capturing the actual phrasing patterns used
by human medical writers.
"""

# ──────────────────────────────────────────────────────────────────────────────
# TEAE Overview paraphrase family
# Used for the primary TEAE incidence sentence
# ──────────────────────────────────────────────────────────────────────────────

TEAE_OVERVIEW = [
    {
        "id": "teae_v1",
        "text": (
            "Treatment-emergent adverse events (TEAEs) were reported for "
            "{TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects in the {ARM_B} arm and "
            "{TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects in the {ARM_A} arm."
        ),
        "slots": ["TEAE_N_B", "TEAE_PCT_B", "N_B", "ARM_B", "TEAE_N_A", "TEAE_PCT_A", "N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v2",
        "text": (
            "{TEAE_N_B} ({TEAE_PCT_B}%) of {N_B} subjects in the {ARM_B} arm and "
            "{TEAE_N_A} ({TEAE_PCT_A}%) of {N_A} subjects in the {ARM_A} arm "
            "experienced at least 1 TEAE."
        ),
        "slots": ["TEAE_N_B", "TEAE_PCT_B", "N_B", "ARM_B", "TEAE_N_A", "TEAE_PCT_A", "N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v3",
        "text": (
            "The overall incidence of TEAEs was {TEAE_PCT_B}% ({TEAE_N_B} subjects) "
            "in the {ARM_B} arm and {TEAE_PCT_A}% ({TEAE_N_A} subjects) in the {ARM_A} arm."
        ),
        "slots": ["TEAE_PCT_B", "TEAE_N_B", "ARM_B", "TEAE_PCT_A", "TEAE_N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v4",
        "text": (
            "Treatment-emergent adverse events occurred in {TEAE_PCT_B}% "
            "({TEAE_N_B} of {N_B}) of subjects receiving {ARM_B} versus "
            "{TEAE_PCT_A}% ({TEAE_N_A} of {N_A}) receiving {ARM_A}."
        ),
        "slots": ["TEAE_PCT_B", "TEAE_N_B", "N_B", "ARM_B", "TEAE_PCT_A", "TEAE_N_A", "N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v5",
        "text": (
            "A total of {TEAE_N_B} ({TEAE_PCT_B}%) participants in the {ARM_B} group "
            "and {TEAE_N_A} ({TEAE_PCT_A}%) in the {ARM_A} group reported at least "
            "one treatment-emergent adverse event."
        ),
        "slots": ["TEAE_N_B", "TEAE_PCT_B", "ARM_B", "TEAE_N_A", "TEAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v6",
        "text": (
            "The incidence of treatment-emergent AEs was {TEAE_PCT_B}% in the "
            "{ARM_B} arm ({TEAE_N_B} of {N_B} subjects) and {TEAE_PCT_A}% in the "
            "{ARM_A} arm ({TEAE_N_A} of {N_A} subjects)."
        ),
        "slots": ["TEAE_PCT_B", "ARM_B", "TEAE_N_B", "N_B", "TEAE_PCT_A", "ARM_A", "TEAE_N_A", "N_A"],
        "conditions": None,
    },
    {
        "id": "teae_v7",
        "text": (
            "Subjects with at least 1 TEAE numbered {TEAE_N_B} ({TEAE_PCT_B}%) in the "
            "{ARM_B} arm and {TEAE_N_A} ({TEAE_PCT_A}%) in the {ARM_A} arm."
        ),
        "slots": ["TEAE_N_B", "TEAE_PCT_B", "ARM_B", "TEAE_N_A", "TEAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v8",
        "text": (
            "TEAEs were observed in {TEAE_PCT_B}% (n={TEAE_N_B}) of subjects treated "
            "with {ARM_B} and {TEAE_PCT_A}% (n={TEAE_N_A}) of those treated with {ARM_A}."
        ),
        "slots": ["TEAE_PCT_B", "TEAE_N_B", "ARM_B", "TEAE_PCT_A", "TEAE_N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v9_pct_only",
        "text": (
            "TEAEs were reported for {TEAE_PCT_B}% of subjects in the {ARM_B} arm "
            "and {TEAE_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["TEAE_PCT_B", "ARM_B", "TEAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "teae_v10_high",
        "text": (
            "The vast majority of subjects in both groups experienced at least 1 TEAE: "
            "{TEAE_PCT_B}% ({TEAE_N_B} of {N_B}) in the {ARM_B} arm and "
            "{TEAE_PCT_A}% ({TEAE_N_A} of {N_A}) in the {ARM_A} arm."
        ),
        "slots": ["TEAE_PCT_B", "TEAE_N_B", "N_B", "ARM_B", "TEAE_PCT_A", "TEAE_N_A", "N_A", "ARM_A"],
        "conditions": {"min_teae_pct": 80},
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Grade 3-4 paraphrase family
# ──────────────────────────────────────────────────────────────────────────────

GRADE_34 = [
    {
        "id": "g34_v1",
        "text": (
            "Grade 3\u20134 TEAEs were reported for {G34_PCT_B}% of subjects in the "
            "{ARM_B} arm and {G34_PCT_A}% of subjects in the {ARM_A} arm."
        ),
        "slots": ["G34_PCT_B", "ARM_B", "G34_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "g34_v2_concise",
        "text": (
            "Grade 3-4 events: {G34_N_B} ({G34_PCT_B}%) vs {G34_N_A} ({G34_PCT_A}%)."
        ),
        "slots": ["G34_N_B", "G34_PCT_B", "G34_N_A", "G34_PCT_A"],
        "conditions": None,
    },
    {
        "id": "g34_v3",
        "text": (
            "Severe (Grade 3\u20134) TEAEs occurred in {G34_PCT_B}% of subjects receiving "
            "{ARM_B} compared with {G34_PCT_A}% receiving {ARM_A}."
        ),
        "slots": ["G34_PCT_B", "ARM_B", "G34_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "g34_v4",
        "text": (
            "The incidence of Grade 3\u20134 TEAEs was {G34_PCT_B}% in the {ARM_B} arm "
            "versus {G34_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["G34_PCT_B", "ARM_B", "G34_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "g34_v5",
        "text": (
            "Grade 3\u20134 TEAEs: {G34_PCT_B}% ({ARM_B}) and {G34_PCT_A}% ({ARM_A})."
        ),
        "slots": ["G34_PCT_B", "ARM_B", "G34_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "g34_v6_with_n",
        "text": (
            "Grade 3\u20134 TEAEs were experienced by {G34_N_B} ({G34_PCT_B}%) subjects "
            "in the {ARM_B} arm and {G34_N_A} ({G34_PCT_A}%) in the {ARM_A} arm."
        ),
        "slots": ["G34_N_B", "G34_PCT_B", "ARM_B", "G34_N_A", "G34_PCT_A", "ARM_A"],
        "conditions": None,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# SAE paraphrase family
# ──────────────────────────────────────────────────────────────────────────────

SAE_OVERALL = [
    {
        "id": "sae_v1",
        "text": (
            "Treatment-emergent SAEs were reported for {SAE_N_B} ({SAE_PCT_B}%) of subjects "
            "in the {ARM_B} arm and {SAE_N_A} ({SAE_PCT_A}%) of subjects in the {ARM_A} arm."
        ),
        "slots": ["SAE_N_B", "SAE_PCT_B", "ARM_B", "SAE_N_A", "SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "sae_v2",
        "text": (
            "Serious adverse events: {SAE_N_B} ({SAE_PCT_B}%) vs {SAE_N_A} ({SAE_PCT_A}%)."
        ),
        "slots": ["SAE_N_B", "SAE_PCT_B", "SAE_N_A", "SAE_PCT_A"],
        "conditions": None,
    },
    {
        "id": "sae_v3",
        "text": (
            "Serious adverse events occurred in {SAE_PCT_B}% ({SAE_N_B} subjects) of the "
            "{ARM_B} arm versus {SAE_PCT_A}% ({SAE_N_A} subjects) of the {ARM_A} arm."
        ),
        "slots": ["SAE_PCT_B", "SAE_N_B", "ARM_B", "SAE_PCT_A", "SAE_N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "sae_v4",
        "text": (
            "The rate of serious adverse events was {SAE_PCT_B}% in the {ARM_B} arm "
            "and {SAE_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["SAE_PCT_B", "ARM_B", "SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "sae_v5",
        "text": (
            "{SAE_N_B} ({SAE_PCT_B}%) subjects in the {ARM_B} group and "
            "{SAE_N_A} ({SAE_PCT_A}%) in the {ARM_A} group experienced at least 1 SAE."
        ),
        "slots": ["SAE_N_B", "SAE_PCT_B", "ARM_B", "SAE_N_A", "SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "sae_v6_pct_only",
        "text": (
            "Treatment-emergent serious adverse events were reported for "
            "{SAE_PCT_B}% ({ARM_B}) and {SAE_PCT_A}% ({ARM_A}) of subjects."
        ),
        "slots": ["SAE_PCT_B", "ARM_B", "SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "sae_v7",
        "text": (
            "Serious TEAEs were observed in {SAE_PCT_B}% of subjects receiving {ARM_B} "
            "and in {SAE_PCT_A}% of those receiving {ARM_A}."
        ),
        "slots": ["SAE_PCT_B", "ARM_B", "SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "sae_v8_concise",
        "text": (
            "SAEs: {SAE_N_B} ({SAE_PCT_B}%) vs {SAE_N_A} ({SAE_PCT_A}%)."
        ),
        "slots": ["SAE_N_B", "SAE_PCT_B", "SAE_N_A", "SAE_PCT_A"],
        "conditions": None,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Discontinuation paraphrase family
# ──────────────────────────────────────────────────────────────────────────────

DISCONTINUATION = [
    {
        "id": "disc_v1",
        "text": (
            "TEAEs leading to treatment discontinuation were reported for "
            "{DISC_N_B} ({DISC_PCT_B}%) of subjects in the {ARM_B} arm and "
            "{DISC_N_A} ({DISC_PCT_A}%) in the {ARM_A} arm."
        ),
        "slots": ["DISC_N_B", "DISC_PCT_B", "ARM_B", "DISC_N_A", "DISC_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "disc_v2_concise",
        "text": (
            "Discontinuations: {DISC_N_B} ({DISC_PCT_B}%) vs {DISC_N_A} ({DISC_PCT_A}%)."
        ),
        "slots": ["DISC_N_B", "DISC_PCT_B", "DISC_N_A", "DISC_PCT_A"],
        "conditions": None,
    },
    {
        "id": "disc_v3",
        "text": (
            "Treatment discontinuation due to TEAEs occurred in {DISC_PCT_B}% "
            "({DISC_N_B} subjects) of the {ARM_B} arm versus {DISC_PCT_A}% "
            "({DISC_N_A} subjects) of the {ARM_A} arm."
        ),
        "slots": ["DISC_PCT_B", "DISC_N_B", "ARM_B", "DISC_PCT_A", "DISC_N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "disc_v4",
        "text": (
            "A total of {DISC_PCT_B}% of subjects in the {ARM_B} arm and "
            "{DISC_PCT_A}% in the {ARM_A} arm discontinued treatment due to TEAEs."
        ),
        "slots": ["DISC_PCT_B", "ARM_B", "DISC_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "disc_v5",
        "text": (
            "Discontinuations due to AEs: {DISC_N_B} ({DISC_PCT_B}%) "
            "and {DISC_N_A} ({DISC_PCT_A}%)."
        ),
        "slots": ["DISC_N_B", "DISC_PCT_B", "DISC_N_A", "DISC_PCT_A"],
        "conditions": None,
    },
    {
        "id": "disc_v6",
        "text": (
            "The rate of discontinuation due to adverse events was "
            "{DISC_PCT_B}% ({DISC_N_B}) for {ARM_B} and "
            "{DISC_PCT_A}% ({DISC_N_A}) for {ARM_A}."
        ),
        "slots": ["DISC_PCT_B", "DISC_N_B", "ARM_B", "DISC_PCT_A", "DISC_N_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "disc_v7_pct_only",
        "text": (
            "Discontinuation due to TEAEs was reported for {DISC_PCT_B}% of {ARM_B} "
            "subjects and {DISC_PCT_A}% of {ARM_A} subjects."
        ),
        "slots": ["DISC_PCT_B", "ARM_B", "DISC_PCT_A", "ARM_A"],
        "conditions": None,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Drug-related TEAE paraphrase family
# ──────────────────────────────────────────────────────────────────────────────

DRUG_TEAE = [
    {
        "id": "dr_teae_v1",
        "text": (
            "Drug-related TEAEs were reported for {DR_TEAE_PCT_B}% of subjects in the "
            "{ARM_B} arm and {DR_TEAE_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["DR_TEAE_PCT_B", "ARM_B", "DR_TEAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_teae_v2",
        "text": (
            "Treatment-related TEAEs occurred in {DR_TEAE_PCT_B}% ({ARM_B}) and "
            "{DR_TEAE_PCT_A}% ({ARM_A}) of subjects."
        ),
        "slots": ["DR_TEAE_PCT_B", "ARM_B", "DR_TEAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_teae_v3",
        "text": (
            "The incidence of drug-related TEAEs was {DR_TEAE_PCT_B}% in the {ARM_B} "
            "arm compared with {DR_TEAE_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["DR_TEAE_PCT_B", "ARM_B", "DR_TEAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_teae_v4",
        "text": (
            "Treatment-related adverse events were reported by {DR_TEAE_PCT_B}% of "
            "subjects receiving {ARM_B} and {DR_TEAE_PCT_A}% receiving {ARM_A}."
        ),
        "slots": ["DR_TEAE_PCT_B", "ARM_B", "DR_TEAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Drug-related SAE paraphrase family
# ──────────────────────────────────────────────────────────────────────────────

DRUG_SAE = [
    {
        "id": "dr_sae_v1",
        "text": (
            "Drug-related SAEs were reported for {DR_SAE_PCT_B}% of subjects in the "
            "{ARM_B} arm and {DR_SAE_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["DR_SAE_PCT_B", "ARM_B", "DR_SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_sae_v2",
        "text": (
            "Treatment-related serious adverse events occurred in "
            "{DR_SAE_PCT_B}% ({ARM_B}) and {DR_SAE_PCT_A}% ({ARM_A}) of subjects."
        ),
        "slots": ["DR_SAE_PCT_B", "ARM_B", "DR_SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_sae_v3",
        "text": (
            "Drug-related serious TEAEs were reported in {DR_SAE_PCT_B}% of the "
            "{ARM_B} arm versus {DR_SAE_PCT_A}% of the {ARM_A} arm."
        ),
        "slots": ["DR_SAE_PCT_B", "ARM_B", "DR_SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_sae_v4",
        "text": (
            "The rate of treatment-related SAEs was {DR_SAE_PCT_B}% for subjects "
            "receiving {ARM_B} and {DR_SAE_PCT_A}% for those receiving {ARM_A}."
        ),
        "slots": ["DR_SAE_PCT_B", "ARM_B", "DR_SAE_PCT_A", "ARM_A"],
        "conditions": None,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Drug-related Discontinuation paraphrase family
# ──────────────────────────────────────────────────────────────────────────────

DRUG_DISC = [
    {
        "id": "dr_disc_v1",
        "text": (
            "Drug-related TEAEs leading to discontinuation were reported for "
            "{DR_DISC_PCT_B}% of subjects in the {ARM_B} arm and "
            "{DR_DISC_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["DR_DISC_PCT_B", "ARM_B", "DR_DISC_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_disc_v2",
        "text": (
            "Treatment-related TEAEs leading to discontinuation occurred in "
            "{DR_DISC_PCT_B}% ({ARM_B}) and {DR_DISC_PCT_A}% ({ARM_A}) of subjects."
        ),
        "slots": ["DR_DISC_PCT_B", "ARM_B", "DR_DISC_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "dr_disc_v3",
        "text": (
            "Discontinuation due to drug-related TEAEs was reported for "
            "{DR_DISC_PCT_B}% of {ARM_B} subjects and {DR_DISC_PCT_A}% of {ARM_A} subjects."
        ),
        "slots": ["DR_DISC_PCT_B", "ARM_B", "DR_DISC_PCT_A", "ARM_A"],
        "conditions": None,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Fatal events paraphrase family
# ──────────────────────────────────────────────────────────────────────────────

FATAL = [
    {
        "id": "fatal_v1_concise",
        "text": (
            "Fatal events: {ARM_B} {FATAL_N_B} ({FATAL_PCT_B}%), {ARM_A} {FATAL_N_A} ({FATAL_PCT_A}%)."
        ),
        "slots": ["ARM_B", "FATAL_N_B", "FATAL_PCT_B", "ARM_A", "FATAL_N_A", "FATAL_PCT_A"],
        "conditions": None,
    },
    {
        "id": "fatal_v2",
        "text": (
            "Fatal TEAEs: {FATAL_PCT_B}% ({ARM_B}) vs {FATAL_PCT_A}% ({ARM_A})."
        ),
        "slots": ["FATAL_PCT_B", "ARM_B", "FATAL_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "fatal_v3",
        "text": (
            "Fatal TEAEs were reported for {FATAL_PCT_B}% of subjects in the {ARM_B} arm "
            "and {FATAL_PCT_A}% in the {ARM_A} arm."
        ),
        "slots": ["FATAL_PCT_B", "ARM_B", "FATAL_PCT_A", "ARM_A"],
        "conditions": None,
    },
    {
        "id": "fatal_v4",
        "text": (
            "Deaths on study: {FATAL_N_B} ({FATAL_PCT_B}%) vs {FATAL_N_A} ({FATAL_PCT_A}%)."
        ),
        "slots": ["FATAL_N_B", "FATAL_PCT_B", "FATAL_N_A", "FATAL_PCT_A"],
        "conditions": None,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Context sentences (meta-text that human writers include)
# These are generated when table features indicate they are applicable.
# They have NO slot dependencies (no numbers from the table).
# ──────────────────────────────────────────────────────────────────────────────

CONTEXT_TABLE_REF = [
    {
        "id": "ctx_table_ref_v1",
        "text": "An overview of treatment-emergent adverse events is presented below.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_table_ref_v2",
        "text": "A summary of treatment-emergent adverse events (TEAEs) is provided below.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_table_ref_v3",
        "text": "A summary of all-causality treatment-emergent adverse events is presented below.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_table_ref_v4",
        "text": "An overall summary of treatment-emergent AEs is presented below.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_table_ref_v5",
        "text": "The adverse events reported in the study are summarized below.",
        "slots": [],
        "conditions": None,
    },
]

CONTEXT_COUNTING_NOTE = [
    {
        "id": "ctx_count_v1",
        "text": "Subjects are counted only once per treatment in each category.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_count_v2",
        "text": "For each category, subjects are counted only once, even if they experienced multiple events in that category.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_count_v3",
        "text": "Except for the number of adverse events, subjects were counted only once per treatment in each row.",
        "slots": [],
        "conditions": None,
    },
]

CONTEXT_DATA_NOTE = [
    {
        "id": "ctx_data_v1",
        "text": "AEs and SAEs are not separated out for this table.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_data_v2",
        "text": "Adverse events and serious adverse events are not separated out.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_data_v3",
        "text": "Non-serious and serious TEAEs are not separated out in this table.",
        "slots": [],
        "conditions": None,
    },
]

CONTEXT_SIMILAR_ARMS = [
    {
        "id": "ctx_similar_v1",
        "text": "Similar percentages of subjects reported at least 1 TEAE in each treatment group.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_similar_v2",
        "text": "Comparable proportions of subjects experienced TEAEs across both treatment arms.",
        "slots": [],
        "conditions": None,
    },
    {
        "id": "ctx_similar_v3",
        "text": "The safety profiles of both treatment arms were generally similar.",
        "slots": [],
        "conditions": None,
    },
]

from typing import Dict, List, Optional

# ── Selection helpers ──────────────────────────────────────────────────────────

FAMILIES = {
    "teae": TEAE_OVERVIEW,
    "grade34": GRADE_34,
    "sae": SAE_OVERALL,
    "disc": DISCONTINUATION,
    "drug_teae": DRUG_TEAE,
    "drug_sae": DRUG_SAE,
    "drug_disc": DRUG_DISC,
    "fatal": FATAL,
    "context_table_ref": CONTEXT_TABLE_REF,
    "context_counting": CONTEXT_COUNTING_NOTE,
    "context_data_note": CONTEXT_DATA_NOTE,
    "context_similar": CONTEXT_SIMILAR_ARMS,
}


def select_variant(
    family: str,
    available_slots: dict,
    data_profile: dict,
    table_text_hash: int,
) -> Optional[dict]:
    """
    Select the best variant from a paraphrase family.

    Selection is prioritized by:
    1. All required slots must be available
    2. Data conditions must be met (if any)
    3. Deterministic tie-breaking via hash of table_text

    Args:
        family: family name (e.g., 'teae', 'grade34', 'sae')
        available_slots: dict of currently filled slot values
        data_profile: dict of data characteristics (pct_b, pct_a, pct_diff, etc.)
        table_text_hash: hash of the source table string for deterministic selection

    Returns:
        Selected variant dict or None if no variant is applicable
    """
    variants = FAMILIES.get(family, [])
    if not variants:
        return None

    applicable = []
    for v in variants:
        # Check all required slots are available
        if not all(s in available_slots and available_slots[s] for s in v["slots"]):
            continue

        # Check data conditions
        conditions = v.get("conditions")
        if conditions:
            if "min_teae_pct" in conditions:
                teae_pct_b = data_profile.get("teae_pct_b", 0)
                teae_pct_a = data_profile.get("teae_pct_a", 0)
                if min(teae_pct_b, teae_pct_a) < conditions["min_teae_pct"]:
                    continue
            if "max_pct" in conditions:
                g34_b = data_profile.get("g34_pct_b", 0)
                g34_a = data_profile.get("g34_pct_a", 0)
                pct_max = max(
                    data_profile.get("sae_pct_b", 0),
                    data_profile.get("sae_pct_a", 0),
                    g34_b, g34_a,
                )
                if pct_max > conditions["max_pct"]:
                    continue
            if "min_pct" in conditions:
                disc_b = data_profile.get("disc_pct_b", 0)
                disc_a = data_profile.get("disc_pct_a", 0)
                if max(disc_b, disc_a) < conditions["min_pct"]:
                    continue

        applicable.append(v)

    if not applicable:
        # Fallback: return first variant that has fewest missing slots
        best = min(variants, key=lambda v: sum(
            1 for s in v["slots"] if s not in available_slots or not available_slots[s]
        ))
        return best

    # Deterministic selection: pick based on hash
    idx = abs(table_text_hash) % len(applicable)
    return applicable[idx]
