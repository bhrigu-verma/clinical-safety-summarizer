"""Compatibility wrapper for src/evaluation/generate_figures.py.

The canonical implementation currently lives at repo root:
    generate_figures.py

This wrapper preserves documented CLI usage:
    python src/evaluation/generate_figures.py ...
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT_IMPL = Path(__file__).resolve().parents[2] / "generate_figures.py"

if not _ROOT_IMPL.exists():
    raise FileNotFoundError(f"Root figure generator not found: {_ROOT_IMPL}")

_spec = importlib.util.spec_from_file_location("_root_generate_figures", str(_ROOT_IMPL))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec from {_ROOT_IMPL}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


if __name__ == "__main__":
    args = _mod.parse_args()
    _mod.generate_all_figures(
        results_dir=args.results_dir,
        output_dir=args.figures_dir,
        tier=args.tier,
    )
