"""
src/generation/sentence_realizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SentenceRealizer, SentenceComposer, and ContextGenerator.

These classes replace the monolithic template-fill approach with a
data-driven, multi-variant sentence generation system that produces
fluent clinical narratives while maintaining 0% hallucination.

The SentenceRealizer selects the optimal paraphrase for each semantic
move based on data context. The SentenceComposer fills slots and
composes the final narrative. The ContextGenerator adds meta-text
sentences that are common in human-written writeups.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.generation.paraphrase_families import (
    FAMILIES, select_variant,
    CONTEXT_TABLE_REF, CONTEXT_COUNTING_NOTE,
    CONTEXT_DATA_NOTE, CONTEXT_SIMILAR_ARMS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FactSheet:
    """Structured facts extracted from the clinical table."""
    arm_a_label: str = "Control"
    arm_b_label: str = "Experimental"
    n_a: Optional[int] = None
    n_b: Optional[int] = None

    # TEAE overview
    teae_n_a: Optional[int] = None
    teae_pct_a: Optional[str] = None
    teae_n_b: Optional[int] = None
    teae_pct_b: Optional[str] = None

    # Grade 3-4
    g34_n_a: Optional[int] = None
    g34_pct_a: Optional[str] = None
    g34_n_b: Optional[int] = None
    g34_pct_b: Optional[str] = None

    # SAE
    sae_n_a: Optional[int] = None
    sae_pct_a: Optional[str] = None
    sae_n_b: Optional[int] = None
    sae_pct_b: Optional[str] = None

    # Discontinuation
    disc_n_a: Optional[int] = None
    disc_pct_a: Optional[str] = None
    disc_n_b: Optional[int] = None
    disc_pct_b: Optional[str] = None

    # Drug-related
    dr_teae_pct_a: Optional[str] = None
    dr_teae_pct_b: Optional[str] = None
    dr_sae_pct_a: Optional[str] = None
    dr_sae_pct_b: Optional[str] = None
    dr_g34_pct_a: Optional[str] = None
    dr_g34_pct_b: Optional[str] = None
    dr_disc_pct_a: Optional[str] = None
    dr_disc_pct_b: Optional[str] = None

    # Fatal
    fatal_pct_a: Optional[str] = None
    fatal_pct_b: Optional[str] = None

    def to_slots(self) -> Dict[str, str]:
        """Convert all non-None facts into slot dict for template filling."""
        slots = {
            "ARM_A": self.arm_a_label,
            "ARM_B": self.arm_b_label,
        }
        field_map = {
            "N_A": self.n_a, "N_B": self.n_b,
            "TEAE_N_A": self.teae_n_a, "TEAE_PCT_A": self.teae_pct_a,
            "TEAE_N_B": self.teae_n_b, "TEAE_PCT_B": self.teae_pct_b,
            "G34_N_A": self.g34_n_a, "G34_PCT_A": self.g34_pct_a,
            "G34_N_B": self.g34_n_b, "G34_PCT_B": self.g34_pct_b,
            "SAE_N_A": self.sae_n_a, "SAE_PCT_A": self.sae_pct_a,
            "SAE_N_B": self.sae_n_b, "SAE_PCT_B": self.sae_pct_b,
            "DISC_N_A": self.disc_n_a, "DISC_PCT_A": self.disc_pct_a,
            "DISC_N_B": self.disc_n_b, "DISC_PCT_B": self.disc_pct_b,
            "DR_TEAE_PCT_A": self.dr_teae_pct_a, "DR_TEAE_PCT_B": self.dr_teae_pct_b,
            "DR_SAE_PCT_A": self.dr_sae_pct_a, "DR_SAE_PCT_B": self.dr_sae_pct_b,
            "DR_G34_PCT_A": self.dr_g34_pct_a, "DR_G34_PCT_B": self.dr_g34_pct_b,
            "DR_DISC_PCT_A": self.dr_disc_pct_a, "DR_DISC_PCT_B": self.dr_disc_pct_b,
            "FATAL_PCT_A": self.fatal_pct_a, "FATAL_PCT_B": self.fatal_pct_b,
        }
        for key, val in field_map.items():
            if val is not None:
                slots[key] = str(val)
        return slots

    def data_profile(self) -> Dict[str, float]:
        """Extract data profile for variant selection logic."""
        def pct(v):
            try:
                return float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                return 0.0
        return {
            "teae_pct_b": pct(self.teae_pct_b),
            "teae_pct_a": pct(self.teae_pct_a),
            "g34_pct_b": pct(self.g34_pct_b),
            "g34_pct_a": pct(self.g34_pct_a),
            "sae_pct_b": pct(self.sae_pct_b),
            "sae_pct_a": pct(self.sae_pct_a),
            "disc_pct_b": pct(self.disc_pct_b),
            "disc_pct_a": pct(self.disc_pct_a),
            "pct_diff": abs(pct(self.teae_pct_b) - pct(self.teae_pct_a)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# SentenceRealizer
# ──────────────────────────────────────────────────────────────────────────────

class SentenceRealizer:
    """
    Selects the optimal sentence paraphrase for each semantic move
    based on the available facts and data profile.

    Usage:
        realizer = SentenceRealizer()
        sentences = realizer.realize(cluster_id, facts, table_text)
    """

    def realize(self, cluster_id: str, facts: FactSheet,
                table_text: str) -> List[str]:
        """
        Return a list of filled sentence strings for the given cluster.

        The cluster determines WHAT semantic moves to emit.
        The data profile determines HOW to phrase each move.
        """
        slots = facts.to_slots()
        profile = facts.data_profile()

        # Deterministic hash for variant selection
        h = int(hashlib.md5(table_text.encode()).hexdigest(), 16)

        sentences = []

        # Add context opener (table reference)
        opener = self._select_context(CONTEXT_TABLE_REF, h)
        sentences.append(opener)

        # Determine which facts exist
        has_teae = (facts.teae_pct_b is not None and facts.teae_pct_a is not None)
        has_g34 = (facts.g34_pct_b is not None and facts.g34_pct_a is not None)
        has_sae = (facts.sae_pct_b is not None and facts.sae_pct_a is not None)
        has_disc = (facts.disc_pct_b is not None or facts.disc_n_b is not None)
        has_dr = (facts.dr_teae_pct_b is not None or facts.dr_sae_pct_b is not None)
        has_fatal = (facts.fatal_pct_b is not None)

        # TEAE overview — always emit if available
        if has_teae:
            v = select_variant("teae", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # Grade 3-4
        if has_g34:
            v = select_variant("grade34", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # SAE
        if has_sae:
            v = select_variant("sae", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # Drug-related SAE (only if has drug-related flag and sae exists)
        if has_dr and facts.dr_sae_pct_b:
            v = select_variant("drug_sae", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # Drug-related TEAE
        if has_dr and facts.dr_teae_pct_b:
            v = select_variant("drug_teae", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # Fatal
        if has_fatal:
            v = select_variant("fatal", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # Drug-related discontinuation
        if has_dr and facts.dr_disc_pct_b:
            v = select_variant("drug_disc", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # Discontinuation
        if has_disc:
            v = select_variant("disc", slots, profile, h)
            if v:
                s = self._fill(v["text"], slots)
                if s:
                    sentences.append(s)

        # Add context closers based on data profile
        closers = self._context_closers(facts, profile, h)
        sentences.extend(closers)

        return sentences

    def _fill(self, template_text: str, slots: Dict[str, str]) -> Optional[str]:
        """Fill a template string with slot values. Returns None if any slot is missing."""
        needed = set(re.findall(r'\{([A-Z0-9_]+)\}', template_text))
        missing = [s for s in needed if s not in slots or not slots[s]]
        if missing:
            # Check if any of the missing slots are critical number slots
            # If even one PCT or N slot is missing, the sentence will have no number -> skip
            number_slots = {s for s in needed if 'PCT' in s or '_N_' in s}
            missing_numbers = number_slots & set(missing)
            if missing_numbers:
                return None
            # Only non-number slots (ARM_A, ARM_B, N_A, N_B) are missing
            # Allow partial fill
            result = template_text
            for s in missing:
                result = re.sub(rf'\s*\{{{s}\}}\s*', ' ', result)
            result = re.sub(r'\s+', ' ', result).strip()
            result = re.sub(r'\s+([.,;:])', r'\1', result)
            return result if len(result) > 20 else None

        result = template_text
        for key, val in slots.items():
            result = result.replace(f'{{{key}}}', str(val))
        return result

    def _select_context(self, family: list, h: int) -> str:
        """Select a context sentence variant deterministically."""
        idx = h % len(family)
        return family[idx]["text"]

    def _context_closers(self, facts: FactSheet,
                         profile: Dict[str, float], h: int) -> List[str]:
        """Generate appropriate context sentences based on table profile."""
        closers = []

        # Counting note for complex tables
        if (facts.teae_pct_b is not None and
            facts.g34_pct_b is not None and
            facts.sae_pct_b is not None):
            closers.append(self._select_context(CONTEXT_COUNTING_NOTE, h + 1))

        # Similar arms note
        if profile["pct_diff"] < 5 and profile["teae_pct_b"] > 50:
            closers.append(self._select_context(CONTEXT_SIMILAR_ARMS, h + 2))

        return closers


# ──────────────────────────────────────────────────────────────────────────────
# ContextGenerator (for backward compatibility with old template-based flow)
# ──────────────────────────────────────────────────────────────────────────────

class ContextGenerator:
    """
    Generates contextual meta-sentences based on table features.
    Used as a supplement to the template-based generation pipeline.

    This is lighter than SentenceRealizer — it only generates
    meta-text, not the core narrative sentences.
    """

    def generate(self, features: dict) -> List[str]:
        """
        Return a list of contextual sentences.

        Args:
            features: dict from StatisticalFeatureExtractor.extract()
        """
        sentences = []
        h = hash(str(features))

        # Table-level contextual sentences
        if features.get('has_sae') and features.get('has_grade34') and features.get('has_discontinuation'):
            # Complex table — add counting note
            sentences.append(CONTEXT_COUNTING_NOTE[abs(h) % len(CONTEXT_COUNTING_NOTE)]["text"])

        if not features.get('has_sae') and features.get('num_rows', 0) < 4:
            sentences.append(CONTEXT_DATA_NOTE[abs(h + 1) % len(CONTEXT_DATA_NOTE)]["text"])

        return sentences


# ──────────────────────────────────────────────────────────────────────────────
# SlotFillGenerator helpers (kept for backward compat)
# ──────────────────────────────────────────────────────────────────────────────

def fill_slots_legacy(template: str, slots: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Legacy slot filling — used for backward compatibility with the old
    template cluster approach. Processes sentence-by-sentence.

    Returns: (filled_text, list_of_unfilled_slot_names)
    """
    sentences = re.split(r'(?<=[.!?])\s+', template)
    filled, unfilled_total = [], []
    for sent in sentences:
        slot_names = re.findall(r'\{([A-Z0-9_]+)\}', sent)
        if not slot_names:
            filled.append(sent)
            continue
        ready = [s for s in slot_names if s in slots and slots[s]]
        if len(ready) > len(slot_names) // 2:
            for s in slot_names:
                sent = sent.replace(f'{{{s}}}', str(slots.get(s, '')))
            filled.append(sent)
        else:
            unfilled_total.extend(slot_names)
    return ' '.join(filled), list(set(unfilled_total))
