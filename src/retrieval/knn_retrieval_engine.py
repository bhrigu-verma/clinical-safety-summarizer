"""
Stage 2 (NEW): KNN Retrieval Engine
Roadmap v2.0, Section 3.2

Instead of generic templates, retrieve the most similar historical writeup
from the 187-pair corpus and adapt it. Research shows KNN retrieval on
domain-specific corpora achieves ROUGE-1 improvements of 10–15 points over
template-only approaches.

Core logic:
  1. Embed all 187 pairs into a 32-dim clinical feature vector.
  2. At query time, find the K nearest neighbors by cosine distance.
  3. Return the top-K clinical writeups (references) for slot-filling.
"""
import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.feature_engineering.statistical_features import StatisticalFeatureExtractor


class KNNRetrievalEngine:
    """
    KNN-based retrieval engine using clinical feature vectors.

    Usage:
        engine = KNNRetrievalEngine()
        engine.build(pairs)            # pairs: list of {table_text, writeup}
        engine.save('data/retrieval_index.pkl')

        # At inference time:
        engine = KNNRetrievalEngine.load('data/retrieval_index.pkl')
        results = engine.retrieve(query_table_text, k=3)
    """

    def __init__(self, k: int = 5, metric: str = 'cosine'):
        self.k = k
        self.metric = metric
        self.feature_extractor = StatisticalFeatureExtractor()
        self.scaler = StandardScaler()
        self.nn_model: Optional[NearestNeighbors] = None
        self.corpus_pairs: List[Dict] = []   # [{table_text, writeup, features}]
        self.feature_matrix: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Index Building
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, pairs: List[Dict]) -> None:
        """
        Build the retrieval index from a list of table-writeup pairs.

        Args:
            pairs: List of dicts, each with keys:
                   'table_text' (str) — linearized table
                   'writeup'    (str) — reference clinical narrative
        """
        if not pairs:
            raise ValueError("Cannot build index from empty corpus.")

        print(f"Building KNN index for {len(pairs)} pairs...")

        # 1. Extract features for every pair
        table_texts = [p['table_text'] for p in pairs]
        raw_features = self.feature_extractor.transform_batch(table_texts)

        # 2. Normalise (StandardScaler — critical for cosine distance to work well)
        scaled_features = self.scaler.fit_transform(raw_features)
        self.feature_matrix = scaled_features

        # 3. Fit KNN model
        effective_k = min(self.k, len(pairs))
        self.nn_model = NearestNeighbors(
            n_neighbors=effective_k,
            metric=self.metric,
            algorithm='brute'       # exact; small corpus
        )
        self.nn_model.fit(scaled_features)

        # 4. Store corpus (features + text)
        self.corpus_pairs = []
        for i, pair in enumerate(pairs):
            self.corpus_pairs.append({
                'table_text': pair['table_text'],
                'writeup':    pair['writeup'],
                'features':   raw_features[i].tolist()
            })

        print(f"✅ KNN index built. {len(self.corpus_pairs)} examples indexed.")

    def build_loo(self, pairs: List[Dict], exclude_idx: int) -> 'KNNRetrievalEngine':
        """
        Build a leave-one-out index excluding the example at exclude_idx.
        Used by the LOO evaluation script.
        """
        loo_pairs = [p for i, p in enumerate(pairs) if i != exclude_idx]
        engine = KNNRetrievalEngine(k=self.k, metric=self.metric)
        engine.build(loo_pairs)
        return engine

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_table_text: str,
        k: Optional[int] = None
    ) -> List[Dict]:
        """
        Find the K most similar historical pairs for a query table.

        Returns:
            List of dicts sorted by similarity (best first):
                {
                  'table_text': str,
                  'writeup':    str,
                  'distance':   float,   # cosine distance (lower = better)
                  'similarity': float,   # 1 - distance (higher = better)
                  'rank':       int
                }
        """
        if self.nn_model is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        k = k or self.k
        effective_k = min(k, len(self.corpus_pairs))

        # Extract and scale query features
        raw_feat = self.feature_extractor.transform_batch([query_table_text])
        scaled_feat = self.scaler.transform(raw_feat)

        # KNN search
        distances, indices = self.nn_model.kneighbors(scaled_feat, n_neighbors=effective_k)
        distances = distances[0]
        indices   = indices[0]

        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances)):
            results.append({
                'table_text': self.corpus_pairs[idx]['table_text'],
                'writeup':    self.corpus_pairs[idx]['writeup'],
                'distance':   float(dist),
                'similarity': float(1 - dist),
                'rank':       rank + 1
            })

        return results

    def retrieve_best(self, query_table_text: str) -> Dict:
        """Convenience: return only the single best match."""
        results = self.retrieve(query_table_text, k=1)
        return results[0] if results else {}

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the full index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'k':             self.k,
                'metric':        self.metric,
                'scaler':        self.scaler,
                'nn_model':      self.nn_model,
                'corpus_pairs':  self.corpus_pairs,
                'feature_matrix': self.feature_matrix,
            }, f)
        print(f"✅ Index saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'KNNRetrievalEngine':
        """Load a persisted index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        engine = cls(k=data['k'], metric=data['metric'])
        engine.scaler         = data['scaler']
        engine.nn_model       = data['nn_model']
        engine.corpus_pairs   = data['corpus_pairs']
        engine.feature_matrix = data['feature_matrix']
        print(f"✅ Index loaded from {path} ({len(engine.corpus_pairs)} examples)")
        return engine


# ── CLI builder ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build KNN retrieval index')
    parser.add_argument('--pairs',  required=True, help='Path to 187_pairs.json')
    parser.add_argument('--output', required=True, help='Output index path (.pkl)')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)

    engine = KNNRetrievalEngine(k=args.k)
    engine.build(pairs)
    engine.save(args.output)
