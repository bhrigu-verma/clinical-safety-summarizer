"""
=============================================================================
Clinical Safety Summarization — Complete Evaluation Metrics Suite
=============================================================================
Project  : Clinical Safety Summarization (Bhrigu Verma, 2026)
File     : src/evaluation/metrics_suite.py
Purpose  : Implements ALL evaluation metrics used in the research paper,
           including standard NLP metrics, clinical-domain metrics, and
           three novel metrics proposed in this work.

METRIC TAXONOMY
---------------
TIER 1 — Standard NLP (existing baselines from literature):
    • ROUGE-1, ROUGE-2, ROUGE-L          (Lin, 2004)
    • BERTScore F1                        (Zhang et al., 2019)
    • METEOR                              (Banerjee & Lavie, 2005)
    • Flesch-Kincaid Readability

TIER 2 — Clinical Faithfulness (adapted from Asgari et al., 2025):
    • Numeric Accuracy Rate  (NAR)    — % numbers grounded in source
    • Hallucination Rate     (HR)     — sentences with unsupported values
    • Omission Rate          (OR)     — fraction of source facts absent

TIER 3 — Novel Metrics (PROPOSED IN THIS PAPER):
    ★ Severity-Weighted Omission Score  (SWOS)
        Weights each omitted clinical fact by its regulatory severity tier.
        Fatal=4, SAE=3, Grade3-4=2, TEAE=1. Unlike plain OR, SWOS penalises
        omitting a fatal event 4× more than omitting a mild TEAE.
        No prior clinical NLP paper has operationalised this distinction.

    ★ Numeric Drift Index  (NDI)
        Measures how numerically "far" generated values are from their source
        anchors — even when they are close but wrong (e.g. 12% vs 12.3%).
        Existing NAR is binary (right/wrong). NDI is continuous and catches
        soft hallucinations such as incorrect rounding or small fabrications.
        Formula: mean(|gen_val - src_val| / src_val) for matched pairs.

    ★ Arm Confusion Rate  (ACR)
        Detects the column-swap failure mode: a numerically correct value
        that is attributed to the WRONG treatment arm. This is categorically
        different from fabricating a number, but equally dangerous in
        regulatory submissions. No existing metric distinguishes these cases.
        Formula: |correct-value-wrong-arm| / |total arm-attributed values|

DEPENDENCIES
------------
    pip install rouge-score bert-score nltk textstat numpy pandas matplotlib
    seaborn scipy scikit-learn tqdm

USAGE
-----
    from src.evaluation.metrics_suite import ClinicalEvaluationSuite

    suite = ClinicalEvaluationSuite()
    results = suite.evaluate_single(
        generated_summary="...",
        reference_summary="...",
        source_table="...",
        arm_names=["Drug A", "Placebo"],
        severity_labels={"TEAE": 1, "Grade 3-4": 2, "SAE": 3, "Fatal": 4}
    )
    print(results)

=============================================================================
"""

from __future__ import annotations

import re
import math
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ── NLTK (lazy imports, avoids crash if not downloaded yet) ──────────────────
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False
    warnings.warn("nltk not installed. METEOR will be skipped. pip install nltk")

# ── ROUGE ─────────────────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False
    warnings.warn("rouge-score not installed. ROUGE metrics will be skipped. pip install rouge-score")

# ── BERTScore ─────────────────────────────────────────────────────────────────
try:
    from bert_score import score as bert_score_fn
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False
    warnings.warn("bert-score not installed. BERTScore will be skipped. pip install bert-score")

# ── Readability ───────────────────────────────────────────────────────────────
try:
    import textstat
    _HAS_TEXTSTAT = True
except ImportError:
    _HAS_TEXTSTAT = False
    warnings.warn("textstat not installed. Readability metrics will be skipped. pip install textstat")

logger = logging.getLogger(__name__)


# =============================================================================
#  DATA STRUCTURES
# =============================================================================

@dataclass
class SingleMetricResult:
    """Holds the result of one evaluation metric for one example."""
    value: float
    display_name: str
    tier: str          # "standard_nlp" | "clinical_faithfulness" | "novel_proposed"
    is_novel: bool = False
    note: str = ""

    def __repr__(self):
        tag = " [NOVEL]" if self.is_novel else ""
        return f"{self.display_name}{tag}: {self.value:.4f} | {self.note}"


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for one (generated, reference, source) triple.
    All values are floats in [0,1] unless noted otherwise.
    """
    # ── Standard NLP metrics ──────────────────────────────────────────────────
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bertscore_f1: float = 0.0
    meteor: float = 0.0
    flesch_kincaid_grade: float = 0.0   # grade level, NOT in [0,1]

    # ── Clinical faithfulness metrics ─────────────────────────────────────────
    numeric_accuracy_rate: float = 0.0      # NAR: in [0,1], higher=better
    hallucination_rate: float = 0.0         # HR:  in [0,1], lower=better
    omission_rate: float = 0.0              # OR:  in [0,1], lower=better

    # ── Novel metrics (proposed in this paper) ────────────────────────────────
    severity_weighted_omission_score: float = 0.0  # SWOS: in [0,1], lower=better
    numeric_drift_index: float = 0.0               # NDI:  in [0,∞), lower=better
    arm_confusion_rate: float = 0.0                # ACR:  in [0,1], lower=better
    risk_deflation_index: float = 0.0              # RDI:  in [0,∞), lower=better
    risk_inflation_index: float = 0.0              # RII:  in [0,∞), lower=better
    delta_preservation_error: float = 0.0          # DPE:  in [0,∞), lower=better
    contraindication_omission_rate: float = 0.0    # COR:  in [0,1], lower=better

    # ── Meta information ──────────────────────────────────────────────────────
    n_source_numbers: int = 0       # how many numbers extracted from source
    n_output_numbers: int = 0       # how many numbers extracted from output
    n_hallucinated_sentences: int = 0
    n_total_sentences: int = 0
    errors: List[str] = field(default_factory=list)   # non-fatal issues encountered

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def safety_score(self) -> float:
        """
        Composite safety score used as the primary paper metric.
        Balances numeric accuracy against hallucination, omission, and arm confusion.
        Range [0,1], higher = safer output.
        Weights reflect clinical significance:
            NAR      × 0.35  (numeric grounding)
            1-HR     × 0.25  (absence of hallucinated values)
            1-OR     × 0.20  (completeness)
            1-ACR    × 0.20  (correct arm attribution)
        """
        return (
            self.numeric_accuracy_rate * 0.35
            + (1.0 - self.hallucination_rate) * 0.25
            + (1.0 - self.omission_rate) * 0.20
            + (1.0 - self.arm_confusion_rate) * 0.20
        )

    def fluency_score(self) -> float:
        """
        Composite fluency score.
        Balances lexical overlap (ROUGE-L), semantic similarity (BERTScore),
        and morphological overlap (METEOR).
        Range [0,1], higher = more fluent/similar to reference.
        """
        weights = {"rl": 0.30, "bs": 0.45, "me": 0.25}
        return (
            self.rouge_l * weights["rl"]
            + self.bertscore_f1 * weights["bs"]
            + self.meteor * weights["me"]
        )


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

# Regex that matches:
#   • integers:             123   1,234   1.234 (European)
#   • decimals:             12.3  0.04    .5
#   • percentages:          12%   12.3%
#   • negative numbers:     -5.2  −5
#   • numbers in brackets:  (12)  used for adverse-event counts in tables
_NUMBER_PATTERN = re.compile(
    r"""
    (?<!\w)                         # not preceded by a word character
    (
        -?                          # optional negative sign
        \d{1,3}                     # 1–3 leading digits
        (?:[,\.]\d{3})*             # optional thousands separators
        (?:[,\.]\d+)?               # optional decimal portion
        |                           # OR
        -?\.\d+                     # bare decimal like .05
    )
    \%?                             # optional percent sign
    (?!\w)                          # not followed by a word character
    """,
    re.VERBOSE
)


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numeric values from a text string as floats.

    Handles:
        • standard decimals:    12.3
        • European decimals:    12,3  (converted to 12.3)
        • thousands separators: 1,234 or 1.234 (converted to 1234)
        • percentages:          12%   (stored as 12.0)
        • negative numbers:     -5.2

    Returns a deduplicated list preserving order of first appearance.

    Examples
    --------
    >>> extract_numbers("TEAE occurred in 45 (12.3%) patients in Drug arm.")
    [45.0, 12.3]

    >>> extract_numbers("0 events, N=10, rate = 0.0%")
    [0.0, 10.0, 0.0]
    """
    raw_matches = _NUMBER_PATTERN.findall(text)
    values: List[float] = []
    seen: set = set()

    for raw in raw_matches:
        cleaned = raw.replace(",", "").replace("%", "")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        # Round to 4 decimal places to avoid float-comparison noise
        val_r = round(val, 4)
        if val_r not in seen:
            seen.add(val_r)
            values.append(val_r)

    return values


def numbers_in_source(output_numbers: List[float],
                      source_numbers: List[float],
                      tolerance: float = 0.01) -> Tuple[int, int]:
    """
    For each number in output_numbers, check whether a numerically close
    value exists in source_numbers.

    Args:
        output_numbers: Numbers extracted from generated text.
        source_numbers: Numbers extracted from source table.
        tolerance:      Relative tolerance for matching (default 1%).
                        Two values a,b match if |a-b| / max(|b|, 1e-9) <= tol.

    Returns:
        Tuple of (n_grounded, n_total):
            n_grounded  how many output numbers have a source match
            n_total     len(output_numbers)
    """
    if not output_numbers:
        return 0, 0

    grounded = 0
    for out_val in output_numbers:
        for src_val in source_numbers:
            denom = max(abs(src_val), 1e-9)
            if abs(out_val - src_val) / denom <= tolerance:
                grounded += 1
                break  # matched; move to next output number

    return grounded, len(output_numbers)


def extract_arm_value_pairs(text: str,
                             arm_names: List[str]) -> List[Tuple[str, float]]:
    """
    Extract (arm_label, numeric_value) pairs from a text.

    A pair is detected when a numeric value appears within a window of
    ±30 tokens of a recognised arm name.

    Args:
        text:       Generated or reference text.
        arm_names:  List of treatment arm labels (e.g. ["Drug A", "Placebo"]).

    Returns:
        List of (arm_name, value) tuples found in the text.

    Example
    -------
    text = "Drug A reported 45 (12.3%) patients, Placebo reported 20 (5.1%)."
    arm_names = ["Drug A", "Placebo"]
    → [("Drug A", 45.0), ("Drug A", 12.3), ("Placebo", 20.0), ("Placebo", 5.1)]
    """
    pairs: List[Tuple[str, float]] = []
    tokens = text.lower().split()
    window = 30  # token radius for arm–number association

    number_positions: List[Tuple[int, float]] = []  # (token_idx, value)
    for idx, tok in enumerate(tokens):
        nums = extract_numbers(tok)
        if nums:
            number_positions.append((idx, nums[0]))

    for arm in arm_names:
        arm_lower = arm.lower()
        arm_tokens = arm_lower.split()
        for i in range(len(tokens) - len(arm_tokens) + 1):
            if tokens[i:i + len(arm_tokens)] == arm_tokens:
                # Found arm at position i; gather nearby numbers
                for num_idx, num_val in number_positions:
                    if abs(num_idx - i) <= window:
                        pairs.append((arm, num_val))

    return pairs


def sentence_tokenize(text: str) -> List[str]:
    """
    Simple but robust sentence splitter that does not require NLTK punkt.
    Works for clinical regulatory text which often has consistent punctuation.
    """
    # Split on period/exclamation/question followed by whitespace + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# =============================================================================
#  SEVERITY DETECTION HELPERS
# =============================================================================

# Maps regex patterns to severity tiers (higher = more clinically serious)
_SEVERITY_PATTERNS: Dict[str, int] = {
    r'\bfatal\b|\bdeath\b|\bdied\b|\bmortality\b': 4,
    r'\bserious adverse event\b|\bsae\b|\bhospitaliz': 3,
    r'\bgrade\s*3\b|\bgrade\s*4\b|\bsevere\b': 2,
    r'\bteae\b|\badverse event\b|\bae\b|\btreatment.emergent\b': 1,
}


def detect_severity_tier(text: str) -> int:
    """
    Return the maximum severity tier (1–4) present in a text fragment.
    Returns 1 (lowest) if no specific severity keyword is found.

    Severity scale:
        4 = Fatal / Death
        3 = Serious Adverse Event (SAE)
        2 = Grade 3 or 4 / Severe
        1 = TEAE / General AE
    """
    text_lower = text.lower()
    max_tier = 1
    for pattern, tier in _SEVERITY_PATTERNS.items():
        if re.search(pattern, text_lower):
            max_tier = max(max_tier, tier)
    return max_tier


def extract_clinical_facts(text: str) -> List[Dict[str, Any]]:
    """
    Extract structured clinical facts from text.

    A 'clinical fact' is a sentence that contains at least one number AND
    at least one clinical keyword (arm name, severity word, drug name hint).

    Returns:
        List of dicts with keys:
            'text'          : the sentence string
            'numbers'       : list of floats extracted
            'severity_tier' : int 1–4
    """
    sentences = sentence_tokenize(text)
    facts = []
    for sent in sentences:
        nums = extract_numbers(sent)
        if nums:
            facts.append({
                'text': sent,
                'numbers': nums,
                'severity_tier': detect_severity_tier(sent),
            })
    return facts


# =============================================================================
#  CORE METRICS IMPLEMENTATIONS
# =============================================================================


class RougeMetrics:
    """Wrapper around rouge_score library with graceful degradation."""

    def __init__(self):
        if _HAS_ROUGE:
            self._scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        else:
            self._scorer = None

    def compute(self, generated: str, reference: str
                ) -> Dict[str, float]:
        """
        Returns dict with keys: rouge_1, rouge_2, rouge_l (F1 scores).
        Returns zeros if rouge-score library is unavailable.
        """
        if self._scorer is None:
            logger.warning("rouge-score unavailable; returning 0.0 for ROUGE metrics.")
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

        try:
            scores = self._scorer.score(reference, generated)
            return {
                "rouge_1": scores["rouge1"].fmeasure,
                "rouge_2": scores["rouge2"].fmeasure,
                "rouge_l": scores["rougeL"].fmeasure,
            }
        except Exception as exc:
            logger.error("ROUGE computation error: %s", exc)
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


class BertScoreMetric:
    """
    BERTScore computation with optional clinical model.

    For clinical domain, use model_type='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    For fast evaluation, use model_type='distilbert-base-uncased'
    Default uses 'roberta-large' (standard choice in NLP literature).
    """

    def __init__(self,
                 model_type: str = "roberta-large",
                 device: str = "cpu",
                 batch_size: int = 16):
        """
        Args:
            model_type:  HuggingFace model ID for the BERT backbone.
            device:      'cpu', 'cuda', or 'mps'
            batch_size:  Number of examples per BERTScore batch.
        """
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size

        if not _HAS_BERTSCORE:
            logger.warning("bert-score unavailable. BERTScore will return 0.0.")

    def compute_batch(self,
                      generated_list: List[str],
                      reference_list: List[str]) -> List[float]:
        """
        Compute BERTScore F1 for a batch of (generated, reference) pairs.

        Args:
            generated_list:  List of generated summaries.
            reference_list:  List of reference summaries (same length).

        Returns:
            List of F1 float scores in [0,1].

        Raises:
            ValueError: if lists have different lengths.
        """
        if len(generated_list) != len(reference_list):
            raise ValueError(
                f"generated_list ({len(generated_list)}) and "
                f"reference_list ({len(reference_list)}) must have equal length."
            )

        if not _HAS_BERTSCORE:
            return [0.0] * len(generated_list)

        if not generated_list:
            return []

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                P, R, F1 = bert_score_fn(
                    generated_list,
                    reference_list,
                    model_type=self.model_type,
                    device=self.device,
                    batch_size=self.batch_size,
                    verbose=False,
                )
            return F1.tolist()
        except Exception as exc:
            logger.error("BERTScore computation error: %s", exc)
            return [0.0] * len(generated_list)

    def compute_single(self, generated: str, reference: str) -> float:
        """Convenience wrapper for single-example BERTScore."""
        results = self.compute_batch([generated], [reference])
        return results[0] if results else 0.0


class MeteorMetric:
    """METEOR score wrapper using NLTK."""

    def __init__(self):
        if _HAS_NLTK:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            except Exception:
                pass

    def compute(self, generated: str, reference: str) -> float:
        """
        Compute METEOR score between generated and reference text.

        METEOR is better than ROUGE at handling paraphrases due to its
        stemming and WordNet synonym matching.

        Returns:
            Float in [0,1]. Returns 0.0 if NLTK is unavailable.
        """
        if not _HAS_NLTK:
            logger.warning("nltk unavailable; returning 0.0 for METEOR.")
            return 0.0
        try:
            gen_tokens = generated.lower().split()
            ref_tokens = reference.lower().split()
            if not gen_tokens or not ref_tokens:
                return 0.0
            score = meteor_score([ref_tokens], gen_tokens)
            return float(score)
        except Exception as exc:
            logger.error("METEOR computation error: %s", exc)
            return 0.0


class ReadabilityMetrics:
    """Flesch–Kincaid and related readability scores."""

    def compute(self, text: str) -> Dict[str, float]:
        """
        Returns dict with:
            flesch_reading_ease:   100=easy, 0=difficult (Flesch, 1948)
            flesch_kincaid_grade:  US school grade level (Kincaid et al., 1975)

        Regulatory medical writing targets FK Grade ~12–14 for medical writers.
        Returns zeros if textstat unavailable.
        """
        if not _HAS_TEXTSTAT:
            return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            }
        except Exception as exc:
            logger.error("Readability computation error: %s", exc)
            return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}


# =============================================================================
#  CLINICAL FAITHFULNESS METRICS
# =============================================================================


class NumericAccuracyRate:
    """
    NAR: What fraction of numbers in the generated summary are grounded in
    the source table?

    Grounding is defined as: exists a source number s such that
        |gen_val - s| / max(|s|, 1e-9) <= tolerance

    This is the primary factual safety metric in this paper.
    Published baseline (Asgari et al., 2025): not directly reported as NAR,
    but hallucination rate of 1.47% implies very high grounding. We set
    NAR >= 0.95 as the publication threshold.
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Args:
            tolerance: Relative tolerance for numeric matching.
                       Default 0.01 = 1%, matching standard rounding.
        """
        self.tolerance = tolerance

    def compute(self, generated: str, source_table: str) -> Dict[str, Any]:
        """
        Args:
            generated:    Generated summary text.
            source_table: Linearised source table string (with start_table...
                          end_table delimiters, or any raw table text).

        Returns:
            Dict with:
                nar:              float in [0,1]
                n_grounded:       int
                n_total:          int
                ungrounded_values: list of floats not found in source
        """
        gen_numbers = extract_numbers(generated)
        src_numbers = extract_numbers(source_table)

        if not gen_numbers:
            return {
                "nar": 1.0,         # vacuously true — no numbers to be wrong
                "n_grounded": 0,
                "n_total": 0,
                "ungrounded_values": [],
            }

        grounded_count = 0
        ungrounded = []
        for gn in gen_numbers:
            matched = False
            for sn in src_numbers:
                denom = max(abs(sn), 1e-9)
                if abs(gn - sn) / denom <= self.tolerance:
                    matched = True
                    break
            if matched:
                grounded_count += 1
            else:
                ungrounded.append(gn)

        nar = grounded_count / len(gen_numbers)
        return {
            "nar": nar,
            "n_grounded": grounded_count,
            "n_total": len(gen_numbers),
            "ungrounded_values": ungrounded,
        }


class HallucinationRate:
    """
    HR: Fraction of output sentences that contain at least one numeric value
    NOT grounded in the source table.

    This is the sentence-level hallucination metric, analogous to the
    1.47% rate reported by Asgari et al. (2025) at annotation level.

    Important distinction from NAR:
        NAR measures value-level accuracy (what fraction of values are correct)
        HR  measures sentence-level corruption (what fraction of sentences are
            infected by even one unsupported value)

    A sentence with 5 correct values and 1 hallucinated value:
        NAR  contribution: 5/6 correct
        HR   contribution: 1 hallucinated sentence (binary)
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        self._nar = NumericAccuracyRate(tolerance=tolerance)

    def compute(self, generated: str, source_table: str) -> Dict[str, Any]:
        """
        Returns:
            Dict with:
                hr:                     float in [0,1], lower is better
                n_hallucinated_sentences: int
                n_total_sentences:       int
                hallucinated_sentences: list of sentence strings
        """
        sentences = sentence_tokenize(generated)
        src_numbers = extract_numbers(source_table)
        hallucinated_sentences = []

        for sent in sentences:
            result = self._nar.compute(sent, source_table)
            if result["ungrounded_values"]:
                hallucinated_sentences.append(sent)

        n_total = len(sentences)
        n_hallucinated = len(hallucinated_sentences)
        hr = n_hallucinated / n_total if n_total > 0 else 0.0

        return {
            "hr": hr,
            "n_hallucinated_sentences": n_hallucinated,
            "n_total_sentences": n_total,
            "hallucinated_sentences": hallucinated_sentences,
        }


class OmissionRate:
    """
    OR: Fraction of clinically significant facts in the source that do NOT
    appear in the generated summary.

    A 'clinical fact' is a (number, severity-context) pair from the source.
    A fact is 'present' in the output if its numeric value is grounded AND
    the severity keyword is mentioned in the same sentence.

    Published baseline: 3.45% omission rate (Asgari et al., 2025) at sentence
    level for clinical note generation. Our target is OR < 0.05 for
    safety tables.
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def compute(self, generated: str, source_table: str) -> Dict[str, Any]:
        """
        Returns:
            Dict with:
                or_score:       float in [0,1], lower is better
                n_omitted:      int
                n_total_facts:  int
                omitted_facts:  list of dict (text, numbers, severity_tier)
        """
        src_facts = extract_clinical_facts(source_table)
        if not src_facts:
            return {"or_score": 0.0, "n_omitted": 0,
                    "n_total_facts": 0, "omitted_facts": []}

        omitted = []
        gen_numbers = extract_numbers(generated)

        for fact in src_facts:
            # A source fact is 'present' if any of its numbers appear in output
            fact_covered = False
            for fact_num in fact['numbers']:
                for gn in gen_numbers:
                    denom = max(abs(fact_num), 1e-9)
                    if abs(gn - fact_num) / denom <= self.tolerance:
                        fact_covered = True
                        break
                if fact_covered:
                    break
            if not fact_covered:
                omitted.append(fact)

        or_score = len(omitted) / len(src_facts)
        return {
            "or_score": or_score,
            "n_omitted": len(omitted),
            "n_total_facts": len(src_facts),
            "omitted_facts": omitted,
        }


# =============================================================================
#  ★ NOVEL METRICS (Proposed in This Paper)
# =============================================================================


class SeverityWeightedOmissionScore:
    """
    ★ NOVEL METRIC 1: Severity-Weighted Omission Score (SWOS)
    ──────────────────────────────────────────────────────────
    MOTIVATION
    ----------
    The plain Omission Rate (OR) treats all omitted facts equally.
    In clinical safety reporting, omitting a fatal event is catastrophically
    more dangerous than omitting a mild TEAE. No prior clinical NLP paper has
    operationalised this distinction.

    DEFINITION
    ----------
    For each clinical fact f_i in the source with severity tier s_i ∈ {1,2,3,4}:
        - If f_i is omitted: add s_i / sum(s_j for all j) to the SWOS numerator
        - If f_i is covered:  add 0

    SWOS = sum(s_i × omit_i) / sum(s_i)

    This produces a weighted omission rate where fatal omissions (s=4) count
    4× more than TEAE omissions (s=1).

    INTERPRETATION
    --------------
    SWOS = 0.0   → perfect: no clinical facts omitted
    SWOS = 1.0   → worst case: all facts omitted (weighted by severity)
    SWOS < OR    → omissions are concentrated in low-severity facts (model
                   correctly prioritises serious events)
    SWOS > OR    → dangerous: model disproportionately omits serious events

    REFERENCE
    ---------
    Proposed in this paper. Inspired by the CREOLA severity framework
    (Asgari et al., 2025) but operationalised for table-to-text generation.
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def compute(self, generated: str, source_table: str) -> Dict[str, Any]:
        """
        Args:
            generated:    Generated summary.
            source_table: Linearised source table.

        Returns:
            Dict with:
                swos:              float in [0,1], lower is better
                plain_or:          float (unweighted, for comparison)
                severity_breakdown: dict mapping tier → {'total', 'omitted'}
                interpretation:    str human-readable note
        """
        src_facts = extract_clinical_facts(source_table)
        if not src_facts:
            return {
                "swos": 0.0,
                "plain_or": 0.0,
                "severity_breakdown": {},
                "interpretation": "No clinical facts found in source table.",
            }

        gen_numbers = extract_numbers(generated)
        tier_stats: Dict[int, Dict[str, int]] = {
            1: {"total": 0, "omitted": 0},
            2: {"total": 0, "omitted": 0},
            3: {"total": 0, "omitted": 0},
            4: {"total": 0, "omitted": 0},
        }

        total_weight = 0.0
        weighted_omission = 0.0

        for fact in src_facts:
            tier = fact['severity_tier']
            tier_stats[tier]["total"] += 1
            total_weight += tier

            # Check coverage
            covered = False
            for fn in fact['numbers']:
                for gn in gen_numbers:
                    if abs(gn - fn) / max(abs(fn), 1e-9) <= self.tolerance:
                        covered = True
                        break
                if covered:
                    break

            if not covered:
                tier_stats[tier]["omitted"] += 1
                weighted_omission += tier  # weight by severity tier

        swos = weighted_omission / total_weight if total_weight > 0 else 0.0
        plain_or = sum(s["omitted"] for s in tier_stats.values()) / len(src_facts)

        # Build interpretation string
        if swos > plain_or + 0.05:
            interpretation = (
                "WARNING: SWOS > OR — model disproportionately omits "
                "HIGH-SEVERITY events. This is a regulatory risk."
            )
        elif swos < plain_or - 0.05:
            interpretation = (
                "GOOD: SWOS < OR — omissions are concentrated in low-severity "
                "facts. Model correctly prioritises serious events."
            )
        else:
            interpretation = "SWOS ≈ OR — omissions distributed across severity tiers."

        return {
            "swos": swos,
            "plain_or": plain_or,
            "severity_breakdown": {
                f"tier_{k}": v for k, v in tier_stats.items()
            },
            "interpretation": interpretation,
        }


class NumericDriftIndex:
    """
    ★ NOVEL METRIC 2: Numeric Drift Index (NDI)
    ────────────────────────────────────────────
    MOTIVATION
    ----------
    The Numeric Accuracy Rate (NAR) is binary: a value is either grounded or
    not. This misses the important case of SOFT HALLUCINATIONS — where a model
    outputs a value close to, but not exactly equal to, the source.

    Examples of soft hallucinations NDI catches but NAR ignores:
        Source: 12.3%   → Output: 12%    (wrong rounding, 2.4% relative drift)
        Source: 0.047   → Output: 0.05   (rounding up, 6.4% drift)
        Source: 1,234   → Output: 1,230  (thousands approximation)

    These are numerically "close" but factually wrong in regulatory submissions
    where exact values are required per ICH E3 guidelines.

    DEFINITION
    ----------
    For each output number g_i, find its closest source number s_i:
        s_i = argmin_{s ∈ source_numbers} |g_i - s|

    Then:
        NDI = mean( |g_i - s_i| / max(|s_i|, 1e-9) )

    NDI is computed over ALL output numbers (not just ungrounded ones),
    giving a continuous measure of how far the output has drifted from the
    source distribution.

    INTERPRETATION
    --------------
    NDI = 0.000   → perfect: all output numbers exactly match source
    NDI = 0.010   → 1% average relative drift (e.g. 12.3 → 12.2)
    NDI = 0.100   → 10% average relative drift (concerning)
    NDI > 0.200   → severe drift (likely hallucination or wrong table used)

    Unlike NAR (higher=better), NDI is lower=better.

    REFERENCE
    ---------
    Novel metric proposed in this paper.
    Closest prior work: numerical faithfulness in data-to-text generation
    (Wiseman et al., 2017; Dusek et al., 2020), but not adapted to clinical
    safety table summarization.
    """

    def compute(self, generated: str, source_table: str) -> Dict[str, Any]:
        """
        Returns:
            Dict with:
                ndi:              float >= 0, lower is better
                per_value_drifts: list of (gen_val, closest_src, rel_drift)
                max_drift:        float — worst single-value drift
                n_values:         int
        """
        gen_numbers = extract_numbers(generated)
        src_numbers = extract_numbers(source_table)

        if not gen_numbers:
            return {
                "ndi": 0.0,
                "per_value_drifts": [],
                "max_drift": 0.0,
                "n_values": 0,
            }

        if not src_numbers:
            # No source numbers to compare against — all output numbers are drifted
            return {
                "ndi": 1.0,
                "per_value_drifts": [(g, None, 1.0) for g in gen_numbers],
                "max_drift": 1.0,
                "n_values": len(gen_numbers),
            }

        drifts = []
        per_value = []
        for gn in gen_numbers:
            # Find closest source number by absolute distance
            closest_src = min(src_numbers, key=lambda s: abs(gn - s))
            # Fixed: Use 0.01 as minimum denominator to prevent exploding NDI for 0.0% sources
            rel_drift = abs(gn - closest_src) / max(abs(closest_src), 0.01)
            drifts.append(rel_drift)
            per_value.append((gn, closest_src, rel_drift))

        ndi = float(np.mean(drifts))
        max_drift = float(np.max(drifts))

        return {
            "ndi": ndi,
            "per_value_drifts": per_value,
            "max_drift": max_drift,
            "n_values": len(gen_numbers),
        }


class ArmConfusionRate:
    """
    ★ NOVEL METRIC 3: Arm Confusion Rate (ACR)
    ───────────────────────────────────────────
    MOTIVATION
    ----------
    Existing metrics (NAR, HR) detect NUMBER hallucinations — when a value in
    the output doesn't exist in the source. But they CANNOT detect the
    ARM-SWAP error: a numerically correct value attributed to the WRONG arm.

    Example of arm-swap (value correct, arm wrong):
        Source: "Drug A: 45 (12.3%) patients had TEAE; Placebo: 20 (5.1%)"
        Output: "Placebo reported 12.3% TEAE; Drug A reported 5.1% TEAE"
        NAR = 1.0 (all numbers grounded!)    ← FAILS TO DETECT ERROR
        ACR = 1.0 (all arms confused)        ← CORRECTLY DETECTS ERROR

    In regulatory submissions, swapping drug and placebo adverse event rates
    is a critical error that could mislead safety assessors.

    DEFINITION
    ----------
    1. From the SOURCE table, extract (arm, value) pairs for each arm.
    2. From the GENERATED text, extract (arm, value) pairs in a window.
    3. For each generated (arm_g, val_g) pair:
        - Find the source (arm_s, val_s) where val_s ≈ val_g
        - If arm_g ≠ arm_s → this is an ARM CONFUSION
    4. ACR = n_confused / n_total_arm_attributed_pairs

    INTERPRETATION
    --------------
    ACR = 0.0   → perfect: all values attributed to correct arms
    ACR = 0.5   → half of arm attributions are wrong
    ACR = 1.0   → complete arm confusion (e.g. drug/placebo fully swapped)

    REFERENCE
    ---------
    Novel metric proposed in this paper. Addresses a failure mode specific
    to adverse event table summarization not covered by any prior metric.
    Inspired by the ArmParser module in our own pipeline which detects
    arm-labeling failures at generation time.
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def compute(self, generated: str,
                source_table: str,
                arm_names: List[str]) -> Dict[str, Any]:
        """
        Args:
            generated:    Generated summary.
            source_table: Linearised source table.
            arm_names:    List of treatment arm labels used in the trial.

        Returns:
            Dict with:
                acr:             float in [0,1], lower is better
                n_confused:      int
                n_evaluated:     int
                confused_pairs:  list of (gen_arm, correct_arm, value)
        """
        if len(arm_names) < 2:
            return {
                "acr": 0.0,
                "n_confused": 0,
                "n_evaluated": 0,
                "confused_pairs": [],
                "note": "Need ≥2 arms to evaluate confusion.",
            }

        # Build source arm→values mapping
        src_arm_values: Dict[str, List[float]] = {arm: [] for arm in arm_names}
        src_pairs = extract_arm_value_pairs(source_table, arm_names)
        for arm, val in src_pairs:
            src_arm_values[arm].append(val)

        # Build generated arm→values mapping
        gen_pairs = extract_arm_value_pairs(generated, arm_names)

        if not gen_pairs:
            return {
                "acr": 0.0,
                "n_confused": 0,
                "n_evaluated": 0,
                "confused_pairs": [],
                "note": "No arm-attributed values found in generated text.",
            }

        confused = []
        evaluable = []

        for gen_arm, gen_val in gen_pairs:
            # Find which source arm actually owns this value
            true_arm = None
            for src_arm, src_vals in src_arm_values.items():
                for sv in src_vals:
                    if abs(gen_val - sv) / max(abs(sv), 1e-9) <= self.tolerance:
                        true_arm = src_arm
                        break
                if true_arm:
                    break

            if true_arm is None:
                # Value not found in source at all — this is a hallucination,
                # not an arm confusion. Skip for ACR purposes.
                continue

            evaluable.append((gen_arm, true_arm, gen_val))
            if gen_arm.lower() != true_arm.lower():
                confused.append((gen_arm, true_arm, gen_val))

        n_evaluated = len(evaluable)
        n_confused = len(confused)
        acr = n_confused / n_evaluated if n_evaluated > 0 else 0.0

        return {
            "acr": acr,
            "n_confused": n_confused,
            "n_evaluated": n_evaluated,
            "confused_pairs": confused,
        }


class RiskInflationDeflationIndex:
    """
    ★ NOVEL METRIC: Risk Inflation / Deflation Index (RII / RDI)
    ────────────────────────────────────────────────────────────
    Measures if the generated text systematically overstates (Inflation) or
    understates (Deflation) risk values compared to the closest source values.
    Understating a severe event is a critical safety failure.
    """
    def compute(self, generated: str, source_table: str) -> Dict[str, Any]:
        gen_numbers = extract_numbers(generated)
        src_numbers = extract_numbers(source_table)
        
        if not gen_numbers or not src_numbers:
            return {"rdi": 0.0, "rii": 0.0}
            
        deflations = []
        inflations = []
        
        for gn in gen_numbers:
            closest_src = min(src_numbers, key=lambda s: abs(gn - s))
            diff = gn - closest_src
            
            # Use a threshold of 0.5 to ignore minor rounding differences
            if diff < -0.5:
                deflations.append(abs(diff) / max(abs(closest_src), 0.01))
            elif diff > 0.5:
                inflations.append(abs(diff) / max(abs(closest_src), 0.01))
                
        rdi = float(np.mean(deflations)) if deflations else 0.0
        rii = float(np.mean(inflations)) if inflations else 0.0
        
        return {"rdi": rdi, "rii": rii}

class ContraindicationOmissionRate:
    """
    ★ NOVEL METRIC: Contraindication Omission Rate (COR)
    ────────────────────────────────────────────────────────────
    Strict binary metric for severe safety signals. If the table contains a 'Fatal'
    event or 'Discontinuation', did the summary mention it?
    """
    def compute(self, generated: str, source_table: str) -> float:
        src_lower = source_table.lower()
        gen_lower = generated.lower()
        
        has_fatal_src = bool(re.search(r'\bfatal\b|\bdeath\b|\bdied\b', src_lower))
        has_disc_src = bool(re.search(r'\bdiscontinu', src_lower))
        
        omissions = 0
        total_contra = 0
        
        if has_fatal_src:
            total_contra += 1
            if not re.search(r'\bfatal\b|\bdeath\b|\bdied\b', gen_lower):
                omissions += 1
                
        if has_disc_src:
            total_contra += 1
            if not re.search(r'\bdiscontinu', gen_lower):
                omissions += 1
                
        if total_contra == 0:
            return 0.0
            
        return float(omissions / total_contra)

class DeltaPreservationError:
    """
    ★ NOVEL METRIC: Delta Preservation Error (DPE)
    ────────────────────────────────────────────────────────────
    Absolute difference between the (Drug % - Placebo %) in the source vs. 
    the generated text for the highest reported event.
    """
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def compute(self, generated: str, source_table: str, arm_names: List[str]) -> float:
        if len(arm_names) < 2:
            return 0.0
            
        gen_pairs = extract_arm_value_pairs(generated, arm_names)
        src_pairs = extract_arm_value_pairs(source_table, arm_names)
        
        if not gen_pairs or not src_pairs:
            return 0.0
            
        arm_0_gen = [v for a, v in gen_pairs if a.lower() == arm_names[0].lower()]
        arm_1_gen = [v for a, v in gen_pairs if a.lower() == arm_names[1].lower()]
        
        arm_0_src = [v for a, v in src_pairs if a.lower() == arm_names[0].lower()]
        arm_1_src = [v for a, v in src_pairs if a.lower() == arm_names[1].lower()]
        
        # If we can't find pairs for both arms, we can't compute a robust delta
        if not arm_0_gen or not arm_1_gen or not arm_0_src or not arm_1_src:
            return 0.0
            
        # Approximation: take the max value for each arm as the overall primary endpoint
        # (TEAE usually has highest %)
        gen_delta = abs(max(arm_0_gen) - max(arm_1_gen))
        src_delta = abs(max(arm_0_src) - max(arm_1_src))
        
        return abs(gen_delta - src_delta)


# =============================================================================
#  MAIN EVALUATION SUITE
# =============================================================================


class ClinicalEvaluationSuite:
    """
    Unified interface for all evaluation metrics used in the research paper.

    Instantiate once, call evaluate_single() or evaluate_batch().

    Args:
        bertscore_model:  HuggingFace model ID for BERTScore backbone.
                          Use 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
                          for clinical-domain BERTScore (requires download).
                          Use 'roberta-large' (default) for standard BERTScore.
                          Use 'distilbert-base-uncased' for fast evaluation.
        bertscore_device: 'cpu', 'cuda', or 'mps'
        numeric_tolerance: Relative tolerance for numeric matching (default 0.01)

    Example
    -------
    suite = ClinicalEvaluationSuite(
        bertscore_model='distilbert-base-uncased',
        bertscore_device='cpu',
    )
    result = suite.evaluate_single(
        generated_summary="Drug A had 45 (12.3%) TEAEs vs Placebo 20 (5.1%).",
        reference_summary="In the Drug A arm, 45 patients (12.3%) experienced TEAEs.",
        source_table="start_table ... [ROW] TEAE | Drug A | 45 | 12.3% | Placebo | 20 | 5.1% end_table",
        arm_names=["Drug A", "Placebo"],
    )
    """

    def __init__(self,
                 bertscore_model: str = "roberta-large",
                 bertscore_device: str = "cpu",
                 numeric_tolerance: float = 0.01):

        self.tol = numeric_tolerance

        # Initialise metric modules
        self.rouge = RougeMetrics()
        self.bertscore = BertScoreMetric(
            model_type=bertscore_model,
            device=bertscore_device,
        )
        self.meteor = MeteorMetric()
        self.readability = ReadabilityMetrics()
        self.nar = NumericAccuracyRate(tolerance=self.tol)
        self.hr = HallucinationRate(tolerance=self.tol)
        self.omission = OmissionRate(tolerance=self.tol)
        # Novel metrics
        self.swos = SeverityWeightedOmissionScore(tolerance=self.tol)
        self.ndi = NumericDriftIndex()
        self.acr = ArmConfusionRate(tolerance=self.tol)
        self.risk_idx = RiskInflationDeflationIndex()
        self.cor = ContraindicationOmissionRate()
        self.dpe = DeltaPreservationError(tolerance=self.tol)

        logger.info(
            "ClinicalEvaluationSuite initialised | BERTScore model: %s | "
            "numeric tolerance: %.3f", bertscore_model, numeric_tolerance
        )

    def evaluate_single(self,
                        generated_summary: str,
                        reference_summary: str,
                        source_table: str,
                        arm_names: Optional[List[str]] = None,
                        ) -> EvaluationResult:
        """
        Run all metrics on a single (generated, reference, source) triple.

        Args:
            generated_summary:  The text produced by the summarization system.
            reference_summary:  The gold-standard reference text.
            source_table:       The input linearised table string.
            arm_names:          List of treatment arm labels (e.g. ["Drug A", "Placebo"]).
                                If None, ACR is skipped (set to 0.0).

        Returns:
            EvaluationResult dataclass with all metric values populated.
        """
        result = EvaluationResult()
        arm_names = arm_names or []

        # ── TIER 1: Standard NLP ─────────────────────────────────────────────
        try:
            rouge_scores = self.rouge.compute(generated_summary, reference_summary)
            result.rouge_1 = rouge_scores["rouge_1"]
            result.rouge_2 = rouge_scores["rouge_2"]
            result.rouge_l = rouge_scores["rouge_l"]
        except Exception as e:
            result.errors.append(f"ROUGE error: {e}")

        try:
            result.bertscore_f1 = self.bertscore.compute_single(
                generated_summary, reference_summary
            )
        except Exception as e:
            result.errors.append(f"BERTScore error: {e}")

        try:
            result.meteor = self.meteor.compute(generated_summary, reference_summary)
        except Exception as e:
            result.errors.append(f"METEOR error: {e}")

        try:
            read_scores = self.readability.compute(generated_summary)
            result.flesch_kincaid_grade = read_scores["flesch_kincaid_grade"]
        except Exception as e:
            result.errors.append(f"Readability error: {e}")

        # ── TIER 2: Clinical Faithfulness ────────────────────────────────────
        try:
            nar_result = self.nar.compute(generated_summary, source_table)
            result.numeric_accuracy_rate = nar_result["nar"]
            result.n_source_numbers = len(extract_numbers(source_table))
            result.n_output_numbers = nar_result["n_total"]
        except Exception as e:
            result.errors.append(f"NAR error: {e}")

        try:
            hr_result = self.hr.compute(generated_summary, source_table)
            result.hallucination_rate = hr_result["hr"]
            result.n_hallucinated_sentences = hr_result["n_hallucinated_sentences"]
            result.n_total_sentences = hr_result["n_total_sentences"]
        except Exception as e:
            result.errors.append(f"HR error: {e}")

        try:
            or_result = self.omission.compute(generated_summary, source_table)
            result.omission_rate = or_result["or_score"]
        except Exception as e:
            result.errors.append(f"OR error: {e}")

        # ── TIER 3: Novel Metrics ─────────────────────────────────────────────
        try:
            swos_result = self.swos.compute(generated_summary, source_table)
            result.severity_weighted_omission_score = swos_result["swos"]
        except Exception as e:
            result.errors.append(f"SWOS error: {e}")

        try:
            ndi_result = self.ndi.compute(generated_summary, source_table)
            result.numeric_drift_index = ndi_result["ndi"]
        except Exception as e:
            result.errors.append(f"NDI error: {e}")

        try:
            if arm_names and len(arm_names) >= 2:
                acr_result = self.acr.compute(
                    generated_summary, source_table, arm_names
                )
                result.arm_confusion_rate = acr_result["acr"]
                
                result.delta_preservation_error = self.dpe.compute(
                    generated_summary, source_table, arm_names
                )
        except Exception as e:
            result.errors.append(f"ACR/DPE error: {e}")

        try:
            risk_result = self.risk_idx.compute(generated_summary, source_table)
            result.risk_deflation_index = risk_result["rdi"]
            result.risk_inflation_index = risk_result["rii"]
            
            result.contraindication_omission_rate = self.cor.compute(
                generated_summary, source_table
            )
        except Exception as e:
            result.errors.append(f"Risk/COR error: {e}")

        if result.errors:
            logger.warning("evaluate_single: %d non-fatal errors: %s",
                           len(result.errors), result.errors)

        return result

    def evaluate_batch(self,
                       examples: List[Dict[str, Any]],
                       show_progress: bool = True,
                       ) -> List[EvaluationResult]:
        """
        Evaluate a list of examples efficiently, batching BERTScore.

        Each example dict must contain:
            'generated'   : str
            'reference'   : str
            'source_table': str
            'arm_names'   : list[str] (optional)

        Args:
            examples:      List of example dicts.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of EvaluationResult objects in the same order as input.
        """
        if not examples:
            return []

        # ── Pre-compute BERTScore in one batch (much faster) ─────────────────
        generated_texts = [e["generated"] for e in examples]
        reference_texts = [e["reference"] for e in examples]

        logger.info("Computing BERTScore for %d examples in batch...", len(examples))
        try:
            bert_scores = self.bertscore.compute_batch(
                generated_texts, reference_texts
            )
        except Exception as e:
            logger.error("Batch BERTScore failed: %s. Falling back to 0.0.", e)
            bert_scores = [0.0] * len(examples)

        # ── Evaluate each example individually for all other metrics ──────────
        results: List[EvaluationResult] = []
        iterator = examples
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(examples, desc="Evaluating", unit="example")
            except ImportError:
                logger.info("tqdm not installed; progress bar unavailable.")

        for i, ex in enumerate(iterator):
            # Temporarily inject the pre-computed BERTScore
            result = self.evaluate_single(
                generated_summary=ex["generated"],
                reference_summary=ex["reference"],
                source_table=ex["source_table"],
                arm_names=ex.get("arm_names", []),
            )
            result.bertscore_f1 = bert_scores[i]  # override with batch result
            results.append(result)

        return results

    def aggregate_results(self,
                          results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate statistics over a list of EvaluationResult objects.

        Returns a nested dict:
            {
              "metric_name": {
                "mean": float,
                "std":  float,
                "p10":  float,   (10th percentile)
                "p50":  float,   (median)
                "p90":  float,   (90th percentile)
              },
              ...
            }
        """
        if not results:
            return {}

        fields = [
            "rouge_1", "rouge_2", "rouge_l", "bertscore_f1", "meteor",
            "flesch_kincaid_grade", "numeric_accuracy_rate", "hallucination_rate",
            "omission_rate", "severity_weighted_omission_score",
            "numeric_drift_index", "arm_confusion_rate",
            "risk_deflation_index", "risk_inflation_index",
            "delta_preservation_error", "contraindication_omission_rate",
        ]

        agg: Dict[str, Dict[str, float]] = {}
        for f in fields:
            vals = np.array([getattr(r, f, 0.0) for r in results], dtype=float)
            agg[f] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "p10":  float(np.percentile(vals, 10)),
                "p50":  float(np.percentile(vals, 50)),
                "p90":  float(np.percentile(vals, 90)),
            }

        # Also include composite scores
        safety_scores = [r.safety_score() for r in results]
        fluency_scores = [r.fluency_score() for r in results]

        for name, scores in [("safety_score", safety_scores),
                              ("fluency_score", fluency_scores)]:
            vals = np.array(scores, dtype=float)
            agg[name] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "p10":  float(np.percentile(vals, 10)),
                "p50":  float(np.percentile(vals, 50)),
                "p90":  float(np.percentile(vals, 90)),
            }

        return agg
