"""Compatibility wrapper for src.evaluation.metrics_suite.

This project currently stores the canonical implementation at repo root:
    metrics_suite.py

This module loads and re-exports that implementation so imports like
`from src.evaluation.metrics_suite import ClinicalEvaluationSuite`
continue to work.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT_IMPL = Path(__file__).resolve().parents[2] / "metrics_suite.py"

if not _ROOT_IMPL.exists():
    raise FileNotFoundError(f"Root metrics suite not found: {_ROOT_IMPL}")

_spec = importlib.util.spec_from_file_location("_root_metrics_suite", str(_ROOT_IMPL))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec from {_ROOT_IMPL}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

# Re-export all public names.
for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

__all__ = [name for name in globals().keys() if not name.startswith("_")]
