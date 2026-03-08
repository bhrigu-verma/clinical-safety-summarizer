"""
Stage 1 (Upgraded): Statistical Feature Extractor
Expanded from 24 → 32 features as per Roadmap v2.0, Section 3.1.

New clinical signal features added:
  - arm_count
  - has_dual_phase
  - discontinuation_delta
  - grade_3_4_ratio
  - sae_drug_related_flag
  - data_completeness_score
  - footnote_count
  - n_subjects_imbalance
"""
import re
import numpy as np
from typing import Dict, List


class StatisticalFeatureExtractor:
    """
    Extracts 32 numerical + clinical signal features from a linearized table.
    These features feed the KNN retrieval engine and the XGB ensemble.
    """

    CLINICAL_THRESHOLDS = {
        'high_severity_ae':   0.50,   # >50% incidence = high severity
        'sae_threshold':      0.20,   # >20% SAE = significant
        'discontinuation':    0.10,   # >10% discontinuation = clinically notable
        'grade34_threshold':  0.30,   # >30% Grade 3-4 = severe
        'sae_drug_related':   0.05,   # >5% drug-related SAE = flag
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(self, text: str) -> Dict[str, float]:
        """Return a dict of all 32 features for one linearized table string."""
        features = {}

        # ── Original 24 features ──────────────────────────────────────────────

        all_numbers = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', text)]
        features['num_count']   = len(all_numbers)
        features['max_number']  = max(all_numbers) if all_numbers else 0
        features['min_number']  = min(all_numbers) if all_numbers else 0
        features['mean_number'] = float(np.mean(all_numbers)) if all_numbers else 0
        features['std_number']  = float(np.std(all_numbers))  if all_numbers else 0

        percentages = [float(p) for p in re.findall(r'(\d+\.?\d*)\s*\)', text)]
        features['pct_count'] = len(percentages)
        features['max_pct']   = max(percentages) if percentages else 0
        features['min_pct']   = min(percentages) if percentages else 0
        features['mean_pct']  = float(np.mean(percentages)) if percentages else 0

        n_matches = re.findall(r'N\s*=\s*(\d+)', text)
        arm_sizes = [int(n) for n in n_matches]
        features['num_arms']       = len(arm_sizes)
        features['total_patients'] = sum(arm_sizes)
        features['arm_size_ratio'] = (
            max(arm_sizes) / min(arm_sizes) if len(arm_sizes) >= 2 and min(arm_sizes) > 0 else 1.0
        )

        features['has_high_teae']       = 1 if any(p > 50 for p in percentages) else 0
        features['has_sae']             = 1 if ('SAE' in text or 'serious adverse' in text.lower()) else 0
        features['has_grade34']         = 1 if ('Grade 3' in text or 'Grade 4' in text) else 0
        features['has_discontinuation'] = 1 if 'discontinu' in text.lower() else 0
        features['has_death']           = 1 if ('death' in text.lower() or 'fatal' in text.lower()) else 0

        features['arm_pct_difference']    = self._compute_arm_difference(text)
        features['num_comparative_rows']  = len(re.findall(r'\[ROW\]', text))

        features['has_title']          = 1 if 'TABLE_TITLE' in text else 0
        features['has_headers']        = 1 if 'HEADERS' in text else 0
        features['table_length_chars'] = len(text)
        features['num_rows']           = text.count('[ROW]')
        features['num_columns']        = self._estimate_columns(text)

        # ── NEW: 8 additional clinical signal features (Roadmap v2, §3.1) ─────

        # 25. arm_count — Number of treatment arms detected
        features['arm_count'] = len(arm_sizes) if arm_sizes else self._infer_arm_count(text)

        # 26. has_dual_phase — Double-Blind AND Open-Label in same table
        features['has_dual_phase'] = 1 if (
            ('double-blind' in text.lower() or 'double blind' in text.lower()) and
            ('open-label' in text.lower() or 'open label' in text.lower())
        ) else 0

        # 27. discontinuation_delta — Absolute difference in discontinuation % between arms
        features['discontinuation_delta'] = self._compute_discontinuation_delta(text)

        # 28. grade_3_4_ratio — Grade 3-4 % ÷ total TEAE % (severity index)
        features['grade_3_4_ratio'] = self._compute_grade34_ratio(text)

        # 29. sae_drug_related_flag — Any drug-related SAE > 5%
        features['sae_drug_related_flag'] = self._flag_sae_drug_related(text)

        # 30. data_completeness_score — Fraction of [ROW] cells that are non-empty
        features['data_completeness_score'] = self._compute_completeness(text)

        # 31. footnote_count — Number of footnote markers (a, b, c…)
        features['footnote_count'] = len(re.findall(r'\b[a-d]\b', text))

        # 32. n_subjects_imbalance — Ratio of largest arm N to smallest arm N
        features['n_subjects_imbalance'] = (
            max(arm_sizes) / min(arm_sizes)
            if len(arm_sizes) >= 2 and min(arm_sizes) > 0 else 1.0
        )

        return features

    def transform_batch(self, texts: List[str]) -> np.ndarray:
        """Transform a batch of table strings into a 2-D feature matrix."""
        all_features = [self.extract(t) for t in texts]
        if not all_features:
            return np.array([])
        keys = self.get_feature_names()
        return np.array([[f.get(k, 0) for k in keys] for f in all_features])

    def get_feature_names(self) -> List[str]:
        """Return all 32 feature names in canonical order."""
        return [
            # Original 24
            'num_count', 'max_number', 'min_number', 'mean_number', 'std_number',
            'pct_count', 'max_pct', 'min_pct', 'mean_pct',
            'num_arms', 'total_patients', 'arm_size_ratio',
            'has_high_teae', 'has_sae', 'has_grade34', 'has_discontinuation', 'has_death',
            'arm_pct_difference', 'num_comparative_rows',
            'has_title', 'has_headers', 'table_length_chars', 'num_rows', 'num_columns',
            # New 8 (Roadmap v2)
            'arm_count', 'has_dual_phase', 'discontinuation_delta', 'grade_3_4_ratio',
            'sae_drug_related_flag', 'data_completeness_score', 'footnote_count',
            'n_subjects_imbalance',
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_arm_difference(self, text: str) -> float:
        rows = re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
        max_diff = 0.0
        for row in rows:
            pcts = [float(p) for p in re.findall(r'(\d+\.?\d*)\s*\)', row)]
            if len(pcts) >= 2:
                diff = abs(pcts[0] - pcts[1])
                max_diff = max(max_diff, diff)
        return max_diff

    def _estimate_columns(self, text: str) -> int:
        header_match = re.search(r'\[HEADERS:(.*?)\]', text)
        if header_match:
            return header_match.group(1).count('|') + 1
        return 0

    def _infer_arm_count(self, text: str) -> int:
        """Fallback arm count from '|' delimiters in the first [ROW]."""
        row_match = re.search(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
        if row_match:
            return max(1, row_match.group(1).count('|'))
        return 2  # reasonable clinical default

    def _compute_discontinuation_delta(self, text: str) -> float:
        """Absolute difference in discontinuation % between arms."""
        disc_rows = [
            row for row in re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
            if 'discontinu' in row.lower()
        ]
        if not disc_rows:
            return 0.0
        pcts = [float(p) for p in re.findall(r'(\d+\.?\d*)\s*\)', disc_rows[0])]
        return abs(pcts[0] - pcts[1]) if len(pcts) >= 2 else 0.0

    def _compute_grade34_ratio(self, text: str) -> float:
        """Grade 3-4 TEAE % ÷ overall TEAE % for the first arm."""
        grade_rows = [
            row for row in re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
            if 'grade 3' in row.lower() and 'teae' in row.lower()
        ]
        teae_rows = [
            row for row in re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
            if re.search(r'\bteae\b', row, re.IGNORECASE) and 'grade' not in row.lower()
        ]
        if grade_rows and teae_rows:
            g34_pcts  = [float(p) for p in re.findall(r'(\d+\.?\d*)\s*\)', grade_rows[0])]
            teae_pcts = [float(p) for p in re.findall(r'(\d+\.?\d*)\s*\)', teae_rows[0])]
            if g34_pcts and teae_pcts and teae_pcts[0] > 0:
                return round(g34_pcts[0] / teae_pcts[0], 4)
        return 0.0

    def _flag_sae_drug_related(self, text: str) -> int:
        """1 if any drug-related SAE percentage > 5%."""
        sae_rows = [
            row for row in re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
            if 'drug-related' in row.lower() and 'sae' in row.lower()
        ]
        for row in sae_rows:
            pcts = [float(p) for p in re.findall(r'(\d+\.?\d*)\s*\)', row)]
            if any(p > self.CLINICAL_THRESHOLDS['sae_drug_related'] * 100 for p in pcts):
                return 1
        return 0

    def _compute_completeness(self, text: str) -> float:
        """Fraction of expected cells (rows × columns) that are filled."""
        rows = re.findall(r'\[ROW\](.*?)(?=\[ROW\]|end_table)', text, re.DOTALL)
        n_cols = self._estimate_columns(text) or 2
        if not rows:
            return 1.0
        filled = sum(
            1 for row in rows
            for cell in row.split('|')
            if cell.strip()
        )
        expected = len(rows) * n_cols
        return round(filled / expected, 4) if expected > 0 else 1.0
