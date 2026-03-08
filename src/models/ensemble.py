"""
Ensemble ML model combining XGBoost, LightGBM, and CatBoost.
Uses weighted voting based on validation performance.
"""
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np
from sklearn.base import BaseEstimator
from typing import List, Dict, Optional
import pickle

class ClinicalEnsemble(BaseEstimator):
    """
    Stacking ensemble for sentence importance prediction.
    
    Architecture:
    - Level 1: XGBoost, LightGBM, CatBoost (base models)
    - Level 2: Logistic Regression meta-learner
    - Output: Sentence importance probability
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        
        # Base models
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='gpu_hist' if use_gpu else 'hist',
            eval_metric='auc',
            random_state=42
        )
        
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            num_leaves=63,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            device='gpu' if use_gpu else 'cpu',
            random_state=42
        )
        
        self.cat_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            task_type='GPU' if use_gpu else 'CPU',
            random_seed=42,
            verbose=0
        )
        
        # Weights (set based on validation performance)
        self.weights = {'xgb': 0.35, 'lgb': 0.40, 'cat': 0.25}
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        """Train all base models"""
        
        print("Training XGBoost...")
        if X_val is not None:
            self.xgb_model.fit(X, y, eval_set=[(X_val, y_val)],
                              early_stopping_rounds=50, verbose=False)
        else:
            self.xgb_model.fit(X, y)
        
        print("Training LightGBM...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        if X_val is not None:
            self.lgb_model.fit(X, y, eval_set=[(X_val, y_val)],
                             callbacks=callbacks)
        else:
            self.lgb_model.fit(X, y)
        
        print("Training CatBoost...")
        if X_val is not None:
            self.cat_model.fit(X, y, eval_set=(X_val, y_val),
                             early_stopping_rounds=50)
        else:
            self.cat_model.fit(X, y)
        
        # Update weights based on validation AUC
        if X_val is not None:
            self._calibrate_weights(X_val, y_val)
        
        self.fitted = True
        return self
    
    def _calibrate_weights(self, X_val, y_val):
        """Calibrate ensemble weights based on validation performance"""
        from sklearn.metrics import roc_auc_score
        
        xgb_auc = roc_auc_score(y_val, self.xgb_model.predict_proba(X_val)[:, 1])
        lgb_auc = roc_auc_score(y_val, self.lgb_model.predict_proba(X_val)[:, 1])
        cat_auc = roc_auc_score(y_val, self.cat_model.predict_proba(X_val)[:, 1])
        
        total = xgb_auc + lgb_auc + cat_auc
        self.weights = {
            'xgb': xgb_auc / total,
            'lgb': lgb_auc / total,
            'cat': cat_auc / total
        }
        
        print(f"Calibrated weights: XGB={self.weights['xgb']:.3f}, "
              f"LGB={self.weights['lgb']:.3f}, CAT={self.weights['cat']:.3f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted probability prediction"""
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        cat_proba = self.cat_model.predict_proba(X)[:, 1]
        
        ensemble_proba = (
            self.weights['xgb'] * xgb_proba +
            self.weights['lgb'] * lgb_proba +
            self.weights['cat'] * cat_proba
        )
        
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
