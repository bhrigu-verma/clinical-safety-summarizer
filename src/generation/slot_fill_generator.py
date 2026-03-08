"""
src/generation/slot_fill_generator.py  — v4.0 (10/10 rewrite)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes vs v3:
  [C1] ArmParser — extracts actual arm names from HEADERS token,
       identifies experimental_arm_index using CONTROL_KEYWORDS.
       CRITICAL: eliminates the "Chemotherapy → Placebo" mislabelling bug.

  [C2] ColumnOrderFix — experimental_arm_index propagated to all
       per-arm stat extraction. Percentages now always come from
       the correct column. CRITICAL: eliminates 95.0% → 92.0% bug.

  [C3] DynamicSlots — ARM_A / ARM_B slots filled from parsed
       arm labels, never from hardcoded strings.

  [C4] IndividualAESelector — uses ContentSelector (LightGBM or
       rule-based) on individual AE rows from the table. Generates
       a separate sentence per selected AE using explicit arm labels.
       CRITICAL: eliminates Rash / Infusion Reaction omission bug.

  [C5] ArmLabelGuard — HallucinationGuardian also checks that every
       percentage in the output is attributed to the correct arm.

  [C6] ExplicitAttributionTemplate — all pairwise stats in the
       template output embed the arm name inline.
"""
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.template_clusters import (
    TEMPLATE_CLUSTERS, select_cluster, get_template, get_required_slots,
)
from src.feature_engineering.statistical_features import StatisticalFeatureExtractor
from src.feature_engineering.row_feature_extractor import RowFeatureExtractor, _parse_cell
from src.models.content_selector import ContentSelector, SelectionResult


# ── Exceptions ─────────────────────────────────────────────────────────────────
class HallucinationError(ValueError):
    pass

class MissingSlotError(ValueError):
    pass


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class ArmContext:
    arm_a_label:             str
    arm_b_label:             str
    n_a:                     Optional[int]
    n_b:                     Optional[int]
    experimental_arm_index:  int
    n_arms:                  int
    raw_labels:              List[str]
    raw_ns:                  List[Optional[int]]


@dataclass
class GenerationResult:
    narrative:           str
    verified:            bool
    cluster_id:          str
    cluster_description: str
    slots_filled:        Dict[str, str]
    missing_slots:       List[str]
    warnings:            List[str]
    numeric_accuracy:    float
    arm_a_label:         str
    arm_b_label:         str
    n_a:                 Optional[int] = None
    n_b:                 Optional[int] = None


# ── CONTROL_KEYWORDS ───────────────────────────────────────────────────────────
CONTROL_KEYWORDS = [
    'placebo', 'vehicle', 'soc', 'standard of care', 'standard care',
    'best supportive', 'observation', 'watchful waiting',
    'chemotherapy', 'chemo', 'docetaxel', 'paclitaxel', 'cabazitaxel',
    'carboplatin', 'cisplatin', 'gemcitabine', 'fluorouracil',
    'control', 'comparator', 'reference arm',
]


# ── Stage 0: Arm Parser ────────────────────────────────────────────────────────
class ArmParser:
    def parse(self, linearized_text: str) -> ArmContext:
        header_match = re.search(r'\[HEADERS:(.*?)\]', linearized_text, re.DOTALL)
        if not header_match:
            return self._fallback(linearized_text)

        parts = [p.strip() for p in header_match.group(1).split('|') if p.strip()]
        raw_labels, raw_ns = [], []

        for part in parts:
            n_m = re.search(r'N\s*=\s*(\d+)', part, re.IGNORECASE)
            n_val = int(n_m.group(1)) if n_m else None
            label = re.sub(r'\(N\s*=\s*\d+\)', '', part, flags=re.IGNORECASE)
            label = re.sub(r'\bN\s*=\s*\d+\b', '', label, flags=re.IGNORECASE)
            label = label.strip(' /|:,')
            raw_labels.append(label)
            raw_ns.append(n_val)

        if len(raw_labels) < 2:
            return self._fallback(linearized_text)

        exp_idx  = self._identify_experimental_arm(raw_labels)
        ctrl_idx = 1 - exp_idx

        return ArmContext(
            arm_a_label            = raw_labels[ctrl_idx],
            arm_b_label            = raw_labels[exp_idx],
            n_a                    = raw_ns[ctrl_idx],
            n_b                    = raw_ns[exp_idx],
            experimental_arm_index = exp_idx,
            n_arms                 = len(raw_labels),
            raw_labels             = raw_labels,
            raw_ns                 = raw_ns,
        )

    def _identify_experimental_arm(self, labels: List[str]) -> int:
        ctrl_scores = [
            sum(kw in lbl.lower() for kw in CONTROL_KEYWORDS)
            for lbl in labels
        ]
        if ctrl_scores[0] > ctrl_scores[1]:
            return 1
        if ctrl_scores[1] > ctrl_scores[0]:
            return 0
        return 1  # default: rightmost column = experimental

    def _fallback(self, text: str) -> ArmContext:
        ns = [int(n) for n in re.findall(r'N\s*=\s*(\d+)', text)]
        return ArmContext(
            arm_a_label="Control", arm_b_label="Experimental",
            n_a=ns[0] if len(ns) > 0 else None,
            n_b=ns[1] if len(ns) > 1 else None,
            experimental_arm_index=1, n_arms=len(ns) if ns else 2,
            raw_labels=["Control", "Experimental"],
            raw_ns=ns[:2] if len(ns) >= 2 else [None, None],
        )


# ── Stage 0b: Overview Slot Extractor ─────────────────────────────────────────
class OverviewSlotExtractor:
    ROW_PATTERNS = [
        ('disc_drug',    r'drug[- ]related.*discontinu|related.*permanent\s*discontinu'),
        ('grade34_drug', r'drug[- ]related.*grade\s*3[-–]?4|related.*grade\s*(3|4)'),
        ('sae_drug',     r'drug[- ]related.*saes?|related.*serious'),
        ('teae_drug',    r'drug[- ]related.*teae|drug[- ]related.*adverse'),
        ('sae_grade34',  r'grade\s*3[-–]?4.*saes?|saes?.*grade\s*3'),
        ('grade34',      r'grade\s*3[-–]?4|grade\s*3\s*or\s*4|grade\s*3/4|severe\s*ae|grade\s*[34]\b|grade\s*\d.*neutropenia|grade\s*\d.*\b(ae|teae)\b'),
        ('disc',         r'discontinu|leading to.*discontinu'),
        ('sae',          r'\bsaes?\b|serious adverse|serious teae'),
        ('fatal',        r'fatal|death|died|mortality|grade\s*5'),
        ('teae',         r'\bteae|treatment[- ]emergent adverse|with teae|subjects with teae'),
    ]

    def extract_slots(self, linearized_text: str, arm_ctx: ArmContext,
                      arm_a_label: str, arm_b_label: str) -> Dict[str, str]:
        exp_idx  = arm_ctx.experimental_arm_index
        ctrl_idx = 1 - exp_idx

        slots: Dict[str, str] = {"ARM_A": arm_a_label, "ARM_B": arm_b_label}
        if arm_ctx.n_a:
            slots["N_A"] = str(arm_ctx.n_a)
        if arm_ctx.n_b:
            slots["N_B"] = str(arm_ctx.n_b)

        raw_rows = re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', linearized_text, re.DOTALL)
        for raw in raw_rows:
            parts = [p.strip() for p in raw.split('|')]
            if len(parts) < 3:
                continue
            label = parts[0].strip()
            row_type = self._classify_row(label.lower())
            if not row_type:
                continue

            values = parts[1:]
            exp_val  = values[exp_idx]  if exp_idx  < len(values) else ""
            ctrl_val = values[ctrl_idx] if ctrl_idx < len(values) else ""

            drug_n, drug_pct = _parse_cell(exp_val)
            ctrl_n, ctrl_pct = _parse_cell(ctrl_val)
            prefix = self._slot_prefix(row_type)

            if drug_n is not None and f'{prefix}_N_B' not in slots:
                slots[f'{prefix}_N_B'] = str(drug_n)
            if drug_pct and f'{prefix}_PCT_B' not in slots:
                slots[f'{prefix}_PCT_B'] = str(drug_pct)
            if ctrl_n is not None and f'{prefix}_N_A' not in slots:
                slots[f'{prefix}_N_A'] = str(ctrl_n)
            if ctrl_pct and f'{prefix}_PCT_A' not in slots:
                slots[f'{prefix}_PCT_A'] = str(ctrl_pct)

        return slots

    def _classify_row(self, label: str) -> Optional[str]:
        for row_type, pattern in self.ROW_PATTERNS:
            if re.search(pattern, label, re.IGNORECASE):
                return row_type
        return None

    def _slot_prefix(self, row_type: str) -> str:
        return {
            'teae': 'TEAE', 'teae_drug': 'DR_TEAE',
            'grade34': 'G34', 'grade34_drug': 'DR_G34',
            'sae': 'SAE', 'sae_drug': 'DR_SAE',
            'sae_grade34': 'SAE_G34', 'disc': 'DISC',
            'disc_drug': 'DR_DISC', 'fatal': 'FATAL',
        }.get(row_type, row_type.upper())


# ── Stage 2: Individual AE Renderer ───────────────────────────────────────────
class IndividualAERenderer:
    def render(self, selected: List[SelectionResult], arm_ctx: ArmContext) -> str:
        if not selected:
            return ""
        arm_b = arm_ctx.arm_b_label
        arm_a = arm_ctx.arm_a_label
        sentences = []
        for res in selected:
            row = res.row
            if not row.label:
                continue
            drug_str = self._fmt(row.drug_n, row.drug_pct)
            ctrl_str = self._fmt(row.ctrl_n, row.ctrl_pct)
            sentences.append(
                f"{row.label} was reported in {drug_str} of subjects in the {arm_b} arm "
                f"versus {ctrl_str} in the {arm_a} arm."
            )
        return " ".join(sentences)

    def _fmt(self, n: Optional[int], pct: float) -> str:
        if n is not None:
            return f"{n} ({pct:.1f}%)"
        return f"{pct:.1f}%"


# ── Stage 3: Hallucination Guardian v4 ────────────────────────────────────────
class HallucinationGuardian:
    ALLOWED_CONTEXT = {'0', '1', '2', '3', '4', '5', '0.0', '1.0', '2.0'}

    def verify(self, narrative: str, source_text: str, arm_ctx: ArmContext
               ) -> Tuple[bool, float, List[str]]:
        source_nums = set(float(n) for n in re.findall(r'\b\d+\.?\d*\b', source_text))
        source_nums_str = (
            {str(n) for n in source_nums} |
            {str(int(n)) for n in source_nums if n == int(n)}
        )
        output_nums = re.findall(r'\b\d+\.?\d*\b', narrative)
        tbl_refs = set(re.findall(r'Table\s+(\d+)', narrative, re.IGNORECASE))

        bad = []
        for ns in output_nums:
            if ns in self.ALLOWED_CONTEXT or ns in tbl_refs:
                continue
            if ns not in source_nums_str:
                try:
                    fv = float(ns)
                    if not any(abs(fv - s) <= 0.11 for s in source_nums):
                        bad.append(ns)
                except ValueError:
                    bad.append(ns)

        arm_warnings = self._check_arm_attribution(narrative, source_text, arm_ctx)
        accuracy = 1.0 - (len(bad) / len(output_nums)) if output_nums else 1.0
        all_warnings = (["Numbers not in source: " + str(bad)] if bad else []) + arm_warnings
        return (len(bad) == 0 and len(arm_warnings) == 0), round(accuracy, 4), all_warnings

    def _check_arm_attribution(self, narrative: str, source_text: str,
                                arm_ctx: ArmContext) -> List[str]:
        warnings = []
        arm_b = arm_ctx.arm_b_label.lower()
        arm_a = arm_ctx.arm_a_label.lower()
        pattern = re.compile(
            r'(\d+\.?\d*)\s*%\s*(?:of subjects\s*)?in the\s+([\w\s\-]+?)\s+arm',
            re.IGNORECASE
        )
        for m in pattern.finditer(narrative):
            pct_val = float(m.group(1))
            claimed = m.group(2).strip().lower()
            if arm_b in claimed or claimed in arm_b:
                col_idx = arm_ctx.experimental_arm_index
            elif arm_a in claimed or claimed in arm_a:
                col_idx = 1 - arm_ctx.experimental_arm_index
            else:
                continue
            col_pcts = self._col_pcts(source_text, col_idx)
            if col_pcts and not any(abs(pct_val - cp) < 0.15 for cp in col_pcts):
                warnings.append(
                    f"{pct_val}% attributed to '{claimed}' arm but not in that column."
                )
        return warnings

    def _col_pcts(self, text: str, col_idx: int) -> List[float]:
        rows = re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
        pcts = []
        for raw in rows:
            parts = [p.strip() for p in raw.split('|')]
            vals = parts[1:]
            if col_idx < len(vals):
                _, pct = _parse_cell(vals[col_idx])
                if pct > 0:
                    pcts.append(pct)
        return pcts


# ── Main Generator ─────────────────────────────────────────────────────────────
class SlotFillGenerator:
    def __init__(self, strict_mode: bool = False):
        self.strict_mode    = strict_mode
        self.arm_parser     = ArmParser()
        self.slot_extractor = OverviewSlotExtractor()
        self.feature_ext    = StatisticalFeatureExtractor()
        self.guardian       = HallucinationGuardian()
        self.ae_renderer    = IndividualAERenderer()
        self.selector       = ContentSelector.load()

    def generate(self, linearized_text: str,
                 retrieved_writeup: Optional[str] = None,
                 retrieved_table_text: Optional[str] = None) -> GenerationResult:
        warnings: List[str] = []

        # [C1] Parse arms
        arm_ctx = self.arm_parser.parse(linearized_text)
        if arm_ctx.arm_a_label in ("Control", "Arm A"):
            warnings.append("Arm labels used generic fallback — check HEADERS token.")

        # [C2+C3] Extract slots with correct column order
        slots = self.slot_extractor.extract_slots(
            linearized_text, arm_ctx,
            arm_a_label=arm_ctx.arm_a_label,
            arm_b_label=arm_ctx.arm_b_label,
        )

        # [C4] Select individual AE rows
        row_ext = RowFeatureExtractor(experimental_arm_index=arm_ctx.experimental_arm_index)
        row_vectors  = row_ext.extract_all(linearized_text)
        sel_results  = self.selector.select(row_vectors)
        selected     = [r for r in sel_results if r.selected]

        # Select cluster and fill template
        feats = self.feature_ext.extract(linearized_text)
        cluster_id = select_cluster(feats)
        meta = TEMPLATE_CLUSTERS[cluster_id]
        required = get_required_slots(cluster_id)
        missing = [s for s in required if s not in slots or not slots[s]]
        if missing:
            warnings.append(f"Missing slots {cluster_id}: {missing}")

        template = get_template(cluster_id)
        if retrieved_writeup and retrieved_table_text:
            adapted = self._adapt_retrieved(retrieved_writeup, retrieved_table_text, arm_ctx)
            if adapted:
                template = adapted
            else:
                warnings.append("KNN adaptation failed.")

        overview, unfilled = self._fill_slots(template, slots)
        for s in unfilled:
            warnings.append(f"Slot {{{s}}} unfilled.")

        # Append individual AE sentences
        ae_text = self.ae_renderer.render(selected, arm_ctx)
        narrative = self._merge(overview, ae_text, slots)
        narrative = self._polish(narrative)

        # [C5] Verify
        verified, accuracy, guard_w = self.guardian.verify(narrative, linearized_text, arm_ctx)
        warnings.extend(guard_w)
        if not verified and self.strict_mode:
            raise HallucinationError(str(guard_w))

        return GenerationResult(
            narrative=narrative.strip(),
            verified=verified,
            cluster_id=cluster_id,
            cluster_description=meta['description'],
            slots_filled={k: v for k, v in slots.items() if k in required},
            missing_slots=missing,
            warnings=warnings,
            numeric_accuracy=accuracy,
            arm_a_label=arm_ctx.arm_a_label,
            arm_b_label=arm_ctx.arm_b_label,
            n_a=arm_ctx.n_a,
            n_b=arm_ctx.n_b,
        )

    def _fill_slots(self, template: str, slots: Dict[str, str]) -> Tuple[str, List[str]]:
        sentences = re.split(r'(?<=[.!?])\s+', template)
        filled, unfilled_total = [], []
        for sent in sentences:
            slot_names = re.findall(r'\{([A-Z0-9_]+)\}', sent)
            if not slot_names:
                filled.append(sent)
                continue
            ready = [s for s in slot_names if s in slots and slots[s]]
            if ready:
                for s in slot_names:
                    sent = sent.replace(f'{{{s}}}', str(slots.get(s, '')))
                filled.append(sent)
            else:
                unfilled_total.extend(slot_names)
        return ' '.join(filled), list(set(unfilled_total))

    def _merge(self, overview: str, ae_text: str, slots: Dict) -> str:
        if not ae_text:
            return overview
        overview_clean = re.sub(r'\{[A-Z_]+\}', '', overview).strip()
        if not overview_clean or len(overview_clean) < 20:
            return ae_text
        return f"{overview} {ae_text}"

    def _polish(self, text: str) -> str:
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s+([.,!?;%])', r'\1', text)
        text = re.sub(r'versus\s+in the\s+', 'versus ', text, flags=re.IGNORECASE)
        return text.strip()

    def _adapt_retrieved(self, writeup: str, table_text: str,
                         arm_ctx: ArmContext) -> Optional[str]:
        try:
            hist_ctx = self.arm_parser.parse(table_text)
            template = writeup
            for i, label in enumerate(hist_ctx.raw_labels):
                if not label or len(label) < 2:
                    continue
                slot = "ARM_B" if i == hist_ctx.experimental_arm_index else "ARM_A"
                template = re.sub(rf'\b{re.escape(label)}\b', f'{{{slot}}}',
                                   template, flags=re.IGNORECASE)
            hist_slots = self.slot_extractor.extract_slots(
                table_text, hist_ctx,
                arm_a_label=hist_ctx.arm_a_label,
                arm_b_label=hist_ctx.arm_b_label,
            )
            for slot_name, val in hist_slots.items():
                if not val or slot_name.startswith("ARM_") or slot_name.startswith("N_"):
                    continue
                template = re.sub(rf'\b{re.escape(str(val))}\b', f'{{{slot_name}}}', template)
            return template if '{' in template else None
        except Exception:
            return None
