"""
src/models/content_selector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: Content Selection — decides WHICH individual AE rows are
clinically significant enough to include in the narrative.

Two modes:
  1. LightGBM (trained) — loads from models/content_selector.pkl
  2. Rule-based fallback — used when no model checkpoint exists

Selection logic:
  ALWAYS include:
    • is_sae = True   (SAE / fatal — regulatory override)
    • is_severe = True (Grade 3-4 — regulatory override)
    • abs_risk_diff > DIFF_THRESHOLD (notable between-arm difference)
    • drug_pct > DRUG_THRESHOLD      (common in experimental arm)

  LightGBM (when loaded):
    • Also includes events where P(select) > threshold (default 0.5)
    • Threshold can be tuned via set_threshold()

Training:
  See scripts/train_content_selector.py for the training pipeline.
  Input: list of {row_features, label=1/0} from annotated examples or
         synthetic data (generate_synthetic_data.py Tier 2 labels).

Saved artefacts:
  models/content_selector.pkl — {model: LGBMClassifier, scaler: StandardScaler}
"""

import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.feature_engineering.row_feature_extractor import (
    RowFeatureExtractor,
    RowFeatureVector,
    ParsedRow,
)

# ── Thresholds (tunable) ───────────────────────────────────────────────────────

DIFF_THRESHOLD  = 5.0    # abs_risk_diff > 5pp → always include
DRUG_THRESHOLD  = 5.0    # drug_pct > 5% → consider for inclusion
LGBM_THRESHOLD  = 0.40   # LightGBM probability cutoff (< 0.5 for higher recall)
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "content_selector.pkl"


# ── Selection result ───────────────────────────────────────────────────────────

@dataclass
class SelectionResult:
    row:           ParsedRow
    selected:      bool
    score:         float        # LightGBM prob or rule-based score (0–1)
    reason:        str          # human-readable reason for selection/rejection
    is_override:   bool         # True if regulatory override forced inclusion


# ── Main Selector ──────────────────────────────────────────────────────────────

class ContentSelector:
    """
    LightGBM-based AE row selector with deterministic regulatory overrides.

    Usage (inference):
        selector = ContentSelector.load()               # loads trained model
        results  = selector.select(row_feature_vectors) # returns [SelectionResult]
        selected = [r.row for r in results if r.selected]

    Usage (fallback — no model):
        selector = ContentSelector()                    # rule-based only
        results  = selector.select(row_feature_vectors)
    """

    def __init__(
        self,
        lgbm_model=None,
        scaler=None,
        threshold: float = LGBM_THRESHOLD,
    ):
        self._model    = lgbm_model   # LGBMClassifier or None
        self._scaler   = scaler       # StandardScaler or None
        self.threshold = threshold
        self._using_lgbm = lgbm_model is not None

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_path: Optional[str] = None) -> "ContentSelector":
        """Load trained LightGBM model from disk. Falls back to rules if not found."""
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            return cls()   # rule-based fallback

        try:
            with open(path, "rb") as f:
                artefacts = pickle.load(f)
            return cls(
                lgbm_model=artefacts["model"],
                scaler=artefacts.get("scaler"),
                threshold=artefacts.get("threshold", LGBM_THRESHOLD),
            )
        except Exception as e:
            print(f"[ContentSelector] Warning: could not load model ({e}). Using rules.")
            return cls()

    def save(self, model_path: Optional[str] = None):
        """Save trained model to disk."""
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":     self._model,
                "scaler":    self._scaler,
                "threshold": self.threshold,
            }, f)

    def set_threshold(self, threshold: float):
        """Tune selection threshold post-training."""
        self.threshold = threshold

    # ── Inference ─────────────────────────────────────────────────────────────

    def select(self, row_vectors: List[RowFeatureVector]) -> List[SelectionResult]:
        """
        Decide which rows to include.

        Args:
            row_vectors: Output of RowFeatureExtractor.extract_all()

        Returns:
            List of SelectionResult (one per input row, with .selected flag).
        """
        if not row_vectors:
            return []

        results = []
        for rv in row_vectors:
            selected, score, reason, override = self._evaluate(rv)
            results.append(SelectionResult(
                row=rv.row,
                selected=selected,
                score=score,
                reason=reason,
                is_override=override,
            ))

        return results

    def _evaluate(
        self, rv: RowFeatureVector
    ) -> Tuple[bool, float, str, bool]:
        """Returns (selected, score, reason, is_override)."""
        row = rv.row

        # ── Regulatory overrides (always select) ──────────────────────────────
        if row.is_sae:
            return True, 1.0, "regulatory_override:SAE/fatal", True

        if row.is_severe:
            return True, 1.0, "regulatory_override:Grade3-4/severe", True

        if row.abs_risk_diff >= DIFF_THRESHOLD:
            return True, 1.0, f"rule:abs_diff={row.abs_risk_diff:.1f}pp>={DIFF_THRESHOLD}pp", True

        # ── LightGBM (if available) ───────────────────────────────────────────
        if self._using_lgbm:
            score = self._lgbm_score(rv.features)
            if score >= self.threshold:
                return True, score, f"lgbm:score={score:.3f}", False
            # Even if LightGBM says no, apply drug threshold as safety net
            if row.drug_pct >= DRUG_THRESHOLD:
                return True, score, f"lgbm_below_thresh_but_drug_pct={row.drug_pct:.1f}%", False
            return False, score, f"lgbm:score={score:.3f}<threshold", False

        # ── Rule-based fallback (no LightGBM) ────────────────────────────────
        score = self._rule_score(row)
        selected = (
            row.drug_pct >= DRUG_THRESHOLD or
            row.abs_risk_diff >= 3.0 or
            row.is_sae or
            row.is_severe
        )
        reason = (
            f"rule:drug_pct={row.drug_pct:.1f}%,"
            f"abs_diff={row.abs_risk_diff:.1f}pp"
        )
        return selected, score, reason, False

    def _lgbm_score(self, features: np.ndarray) -> float:
        """Run LightGBM inference."""
        X = features.reshape(1, -1)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        proba = self._model.predict_proba(X)[0]
        return float(proba[1])   # probability of class 1 (select)

    def _rule_score(self, row: ParsedRow) -> float:
        """
        Heuristic 0–1 score for rule-based fallback.
        Higher = more likely to be clinically significant.
        """
        score = 0.0
        if row.drug_pct > 0:    score += min(row.drug_pct / 30.0, 0.4)
        if row.abs_risk_diff > 0: score += min(row.abs_risk_diff / 20.0, 0.4)
        if row.is_sae:            score += 0.15
        if row.is_severe:         score += 0.10
        if row.relative_risk > 2: score += 0.05
        return round(min(score, 1.0), 4)

    # ── Training interface ────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,       # (n_samples, 12) feature matrix
        y: np.ndarray,       # (n_samples,) binary labels 0/1
        class_weight: str = "balanced",
    ) -> "ContentSelector":
        """
        Train a LightGBM binary classifier.

        Args:
            X: feature matrix (use RowFeatureExtractor.extract_all() to build)
            y: binary labels (1 = include in narrative, 0 = skip)
            class_weight: 'balanced' recommended (selected events are minority)
        """
        try:
            from lightgbm import LGBMClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.utils.class_weight import compute_sample_weight

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            # Compute sample weights for class imbalance
            sample_weights = compute_sample_weight("balanced", y)

            self._model = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            self._model.fit(X_scaled, y, sample_weight=sample_weights)
            self._using_lgbm = True

            # Print feature importances
            importances = self._model.feature_importances_
            names = RowFeatureExtractor.FEATURE_NAMES
            sorted_idx = np.argsort(importances)[::-1]
            print("\nLightGBM Feature Importances:")
            for idx in sorted_idx:
                print(f"  {names[idx]:20s}: {importances[idx]:.4f}")

        except ImportError:
            raise ImportError(
                "lightgbm is required for training. Install with: pip install lightgbm"
            )

        return self

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Evaluate classifier on held-out set."""
        from sklearn.metrics import (
            classification_report, roc_auc_score, average_precision_score
        )
        X_s = self._scaler.transform(X) if self._scaler else X
        y_pred  = self._model.predict(X_s)
        y_proba = self._model.predict_proba(X_s)[:, 1]

        report = classification_report(y, y_pred, output_dict=True)
        return {
            "classification_report": report,
            "roc_auc":   round(roc_auc_score(y, y_proba), 4),
            "avg_prec":  round(average_precision_score(y, y_proba), 4),
            "accuracy":  round(report["accuracy"], 4),
            "precision_class1": round(report["1"]["precision"], 4),
            "recall_class1":    round(report["1"]["recall"], 4),
            "f1_class1":        round(report["1"]["f1-score"], 4),
        }
