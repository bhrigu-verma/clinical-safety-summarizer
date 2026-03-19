"""Compatibility wrapper for src/evaluation/ablation_analysis.py.

The canonical implementation currently lives at repo root:
    ablation_analysis.py

This wrapper preserves CLI usage:
    python src/evaluation/ablation_analysis.py ...
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT_IMPL = Path(__file__).resolve().parents[2] / "ablation_analysis.py"

if not _ROOT_IMPL.exists():
    raise FileNotFoundError(f"Root ablation analyzer not found: {_ROOT_IMPL}")

_spec = importlib.util.spec_from_file_location("_root_ablation_analysis", str(_ROOT_IMPL))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec from {_ROOT_IMPL}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


if __name__ == "__main__":
    _mod.main()
