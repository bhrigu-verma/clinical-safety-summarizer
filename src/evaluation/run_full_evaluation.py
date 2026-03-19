"""Compatibility wrapper for src/evaluation/run_full_evaluation.py.

The canonical implementation currently lives at repo root:
    run_full_evaluation.py

This wrapper preserves documented CLI usage:
    python src/evaluation/run_full_evaluation.py ...
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT_IMPL = Path(__file__).resolve().parents[2] / "run_full_evaluation.py"

if not _ROOT_IMPL.exists():
    raise FileNotFoundError(f"Root evaluation runner not found: {_ROOT_IMPL}")

_spec = importlib.util.spec_from_file_location("_root_run_full_evaluation", str(_ROOT_IMPL))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec from {_ROOT_IMPL}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


if __name__ == "__main__":
    args = _mod.parse_args()
    _mod.run_evaluation(
        tier=args.tier,
        modes=args.modes,
        ml_url=args.ml_url,
        dl_url=args.dl_url,
        output_dir=args.output_dir,
        benchmark_dir=args.benchmark_dir,
        bertscore_model=args.bertscore_model,
        bertscore_device=args.bertscore_device,
        numeric_tolerance=args.numeric_tolerance,
        profile_name=args.profile_name,
        n_max=args.n_max,
    )
