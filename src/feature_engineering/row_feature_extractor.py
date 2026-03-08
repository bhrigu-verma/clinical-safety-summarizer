"""
src/feature_engineering/row_feature_extractor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per-row feature extraction for individual adverse event rows.

These features feed the LightGBM content selector (Stage 1) to decide
WHICH individual AE rows are clinically significant enough to mention
in the generated narrative.

Feature vector (12 features per row):
  1.  drug_pct          — incidence % in experimental arm
  2.  ctrl_pct          — incidence % in control arm
  3.  abs_risk_diff     — |drug_pct - ctrl_pct|  (ALWAYS positive → no column-swap bug)
  4.  risk_dir          — +1 if drug > ctrl, -1 if ctrl > drug (kept separate from magnitude)
  5.  relative_risk     — drug_pct / ctrl_pct  (capped at 20 for inf cases)
  6.  drug_pct_rank     — rank of drug_pct within this table (0.0 – 1.0, higher = more common)
  7.  abs_diff_rank     — rank of abs_risk_diff within this table (0.0 – 1.0)
  8.  is_sae            — 1 if row label contains SAE / serious / fatal
  9.  is_severe         — 1 if row label contains Grade 3 / Grade 4 / severe
  10. drug_above_5pct   — 1 if drug_pct > 5  (regulatory threshold for mention)
  11. drug_above_10pct  — 1 if drug_pct > 10
  12. diff_above_5pp    — 1 if abs_risk_diff > 5 percentage points
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ParsedRow:
    """One fully parsed adverse-event row with per-arm values."""
    label:            str
    drug_n:           Optional[int]
    drug_pct:         float
    ctrl_n:           Optional[int]
    ctrl_pct:         float
    abs_risk_diff:    float
    risk_dir:         int          # +1 drug > ctrl, -1 ctrl > drug, 0 equal
    relative_risk:    float
    is_sae:           bool
    is_severe:        bool
    row_index:        int          # position in the table (0-based)


@dataclass
class RowFeatureVector:
    row:      ParsedRow
    features: np.ndarray           # 12-dim float vector
    feature_names: List[str]


# ── SAE / Severity keyword lists ───────────────────────────────────────────────

SAE_KEYWORDS = [
    'serious adverse', 'treatment-related sae', 'treatment emergent sae',
    r'\bsaes?\b', 'serious teae', 'treatment-emergent serious',
    'fatal', 'death', 'died', 'mortality', 'grade 5',
]

SEVERE_KEYWORDS = [
    r'grade\s*3[-–]?4', r'grade\s*3\s*or\s*4', r'grade\s*3/4',
    r'grade\s*[34]\b', r'\bsevere\b', 'grade ≥ 3', 'grade >= 3',
]

OVERVIEW_PATTERNS = [
    # rows we don't select individually — they belong to overview templates
    r'\bany\s+teae\b', r'total.*teae', r'subjects with teae',
    r'treatment[- ]emergent adverse', r'at least one',
    r'overview of teae', r'summary of teae',
]


def _is_sae(label: str) -> bool:
    return any(re.search(p, label, re.IGNORECASE) for p in SAE_KEYWORDS)

def _is_severe(label: str) -> bool:
    return any(re.search(p, label, re.IGNORECASE) for p in SEVERE_KEYWORDS)

def _is_overview_row(label: str) -> bool:
    """Overview rows belong to the standard template, not individual AE selection."""
    return any(re.search(p, label, re.IGNORECASE) for p in OVERVIEW_PATTERNS)


# ── Main Extractor ─────────────────────────────────────────────────────────────

class RowFeatureExtractor:
    """
    Extracts a 12-feature vector for each individual AE row.

    Requires:
        experimental_arm_index (int): 0 or 1 — which column is the drug arm.
                                      Provided by ArmParser (see slot_fill_generator).
    """

    FEATURE_NAMES = [
        "drug_pct", "ctrl_pct", "abs_risk_diff", "risk_dir",
        "relative_risk", "drug_pct_rank", "abs_diff_rank",
        "is_sae", "is_severe", "drug_above_5pct",
        "drug_above_10pct", "diff_above_5pp",
    ]

    def __init__(self, experimental_arm_index: int = 1):
        """
        Args:
            experimental_arm_index: 0 if drug arm is leftmost column,
                                    1 if drug arm is rightmost column.
        """
        self.exp_idx  = experimental_arm_index
        self.ctrl_idx = 1 - experimental_arm_index

    def extract_all(
        self,
        linearized_text: str,
    ) -> List[RowFeatureVector]:
        """
        Parse every individual AE row in the table and return feature vectors.
        Overview rows (any TEAE, total subjects, etc.) are excluded.
        """
        rows = self._parse_rows(linearized_text)
        if not rows:
            return []

        # Build table-wide stats for ranking features
        drug_pcts   = [r.drug_pct    for r in rows]
        abs_diffs   = [r.abs_risk_diff for r in rows]
        max_drug    = max(drug_pcts)  if drug_pcts else 1.0
        max_diff    = max(abs_diffs)  if abs_diffs else 1.0

        vectors = []
        for r in rows:
            drug_pct_rank = r.drug_pct    / max_drug if max_drug > 0 else 0.0
            abs_diff_rank = r.abs_risk_diff / max_diff if max_diff > 0 else 0.0

            fv = np.array([
                r.drug_pct,
                r.ctrl_pct,
                r.abs_risk_diff,
                float(r.risk_dir),
                min(r.relative_risk, 20.0),   # cap at 20 for inf cases
                drug_pct_rank,
                abs_diff_rank,
                float(r.is_sae),
                float(r.is_severe),
                float(r.drug_pct > 5.0),
                float(r.drug_pct > 10.0),
                float(r.abs_risk_diff > 5.0),
            ], dtype=np.float32)

            vectors.append(RowFeatureVector(row=r, features=fv, feature_names=self.FEATURE_NAMES))

        return vectors

    def _parse_rows(self, text: str) -> List[ParsedRow]:
        """Extract all individual AE rows from a linearized table."""
        raw_rows = re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
        parsed = []

        for idx, raw in enumerate(raw_rows):
            parts = [p.strip() for p in raw.split('|')]
            if len(parts) < 2:
                continue

            label = parts[0].strip()
            if not label:
                continue

            # Skip overview rows — those go through the standard template
            if _is_overview_row(label):
                continue

            # Parse value columns (support N (pct%) or bare pct)
            values = parts[1:]

            # We need at least 2 columns for comparison
            if len(values) < 2:
                continue

            # Align to experimental / control columns
            exp_val_str  = values[self.exp_idx]  if self.exp_idx  < len(values) else ""
            ctrl_val_str = values[self.ctrl_idx] if self.ctrl_idx < len(values) else ""

            drug_n, drug_pct = _parse_cell(exp_val_str)
            ctrl_n, ctrl_pct = _parse_cell(ctrl_val_str)

            if drug_pct == 0 and ctrl_pct == 0:
                continue   # empty row — skip

            abs_diff = abs(drug_pct - ctrl_pct)
            risk_dir = 1 if drug_pct > ctrl_pct else (-1 if ctrl_pct > drug_pct else 0)
            rr = (drug_pct / ctrl_pct) if ctrl_pct > 0 else (20.0 if drug_pct > 0 else 1.0)

            parsed.append(ParsedRow(
                label=label,
                drug_n=drug_n,
                drug_pct=drug_pct,
                ctrl_n=ctrl_n,
                ctrl_pct=ctrl_pct,
                abs_risk_diff=abs_diff,
                risk_dir=risk_dir,
                relative_risk=rr,
                is_sae=_is_sae(label),
                is_severe=_is_severe(label),
                row_index=idx,
            ))

        return parsed


# ── Cell parser ────────────────────────────────────────────────────────────────

def _parse_cell(cell: str) -> Tuple[Optional[int], float]:
    """
    Parse a table cell value into (n, pct).

    Handles:
      "35 (8.5%)"  → (35, 8.5)
      "35 (8.5)"   → (35, 8.5)
      "8.5%"       → (None, 8.5)
      "35"         → (None, 35.0)   ← treated as raw count, pct unknown
      ""           → (None, 0.0)
    """
    cell = cell.strip().replace('\n', ' ').replace(',', '')

    # N (pct%)  or  N (pct)
    m = re.match(r'^(\d+)\s*\((\d+\.?\d*)', cell)
    if m:
        return int(m.group(1)), float(m.group(2))

    # pct% only
    m = re.match(r'^(\d+\.?\d*)\s*%', cell)
    if m:
        return None, float(m.group(1))

    # bare integer
    m = re.match(r'^(\d+)$', cell)
    if m:
        return int(m.group(1)), 0.0

    # bare decimal
    m = re.match(r'^(\d+\.\d+)$', cell)
    if m:
        return None, float(m.group(1))

    return None, 0.0
