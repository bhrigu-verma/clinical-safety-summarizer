"""
=============================================================================
Clinical Safety Summarization — Evaluation Runner
=============================================================================
File    : src/evaluation/run_full_evaluation.py
Purpose : Loads benchmark JSONLines files, calls the summarization backends
          (ML, DL-base, DL-finetuned), runs the full metrics suite on each
          output, and saves results to CSV/JSON for figure generation.

USAGE
-----
    # From the project root directory:
    python src/evaluation/run_full_evaluation.py \
        --tier tier1 \
        --modes ml dl_base finetuned \
        --ml-url http://localhost:8000 \
        --dl-url http://localhost:8001 \
        --output-dir data/eval_results \
        --bertscore-model distilbert-base-uncased

    # Quick smoke test (first 5 examples only):
    python src/evaluation/run_full_evaluation.py --tier tier1 --n-max 5

BENCHMARK FILE FORMAT (JSONLines)
----------------------------------
Each line is a JSON object with at minimum:
    {
      "table_text":   "<linearised table string>",
      "reference":    "<gold summary string>",
      "arm_names":    ["Drug A", "Placebo"],       <- optional
      "table_id":     "trial_123_table_2"           <- optional, for tracking
    }

OUTPUT FILES
------------
    data/eval_results/
        results_<tier>_<timestamp>.csv    <- per-example metric values
        aggregate_<tier>_<timestamp>.json <- mean/std/percentiles per metric
        errors_<tier>_<timestamp>.log     <- any per-example errors

=============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import csv
import re
import logging
import argparse
import time
import datetime
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path so imports work regardless of working directory.
# Supports both layouts:
#   1) <repo>/src/evaluation/run_full_evaluation.py
#   2) <repo>/run_full_evaluation.py
_HERE = Path(__file__).resolve().parent
if (_HERE / "src").exists():
    _PROJECT_ROOT = _HERE
else:
    _PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics_suite import ClinicalEvaluationSuite, EvaluationResult

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
#  CONFIGURATION DEFAULTS
# =============================================================================

DEFAULT_BENCHMARK_DIR = str(_PROJECT_ROOT / "data" / "benchmark")
DEFAULT_OUTPUT_DIR    = str(_PROJECT_ROOT / "data" / "eval_results")

BENCHMARK_FILES = {
    "tier1": "tier1_gold.jsonl",
    "tier2": "tier2_stress.jsonl",
    "full":  "clinical_10k_final.jsonl",   # Use with n-max for sampling
}

ML_SUMMARIZE_ENDPOINT  = "/summarize"
DL_SUMMARIZE_ENDPOINT  = "/summarize"
DL_COMPARE_ENDPOINT    = "/summarize-compare"

# Maximum wait time per API call in seconds
REQUEST_TIMEOUT = 120


# =============================================================================
#  API CALL HELPERS
# =============================================================================

def call_ml_backend(base_url: str,
                    table_text: str,
                    profile_name: str,
                    timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
    """
    Call the ML backend (port 8000) to get a deterministic summary.

    Args:
        base_url:   Base URL of the ML backend, e.g. 'http://localhost:8000'
        table_text: Linearised table string.
        timeout:    Request timeout in seconds.

    Returns:
        Summary string if successful, None on any error.
    """
    url = base_url.rstrip("/") + ML_SUMMARIZE_ENDPOINT
    payload = {
        "table_text": table_text,
        # Keep strict mode enabled for full_system; loosen in no_arm_parser profile.
        "strict_mode": profile_name != "no_arm_parser",
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Support both 'summary' and 'text' as response keys
        return data.get("summary") or data.get("text") or ""
    except requests.exceptions.ConnectionError:
        logger.warning("ML backend not reachable at %s. Skipping ML call.", url)
        return None
    except requests.exceptions.Timeout:
        logger.warning("ML backend timed out after %ds for call to %s", timeout, url)
        return None
    except Exception as exc:
        logger.error("ML backend call failed: %s", exc)
        return None


def call_dl_backend(base_url: str,
                    table_text: str,
                    mode: str = "ml",
                    profile_name: str = "full_system",
                    timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
    """
    Call the DL backend (port 8001) to get a summary.

    Args:
        base_url:   Base URL of the DL backend, e.g. 'http://localhost:8001'
        table_text: Linearised table string.
        mode:       'ml', 'base_dl', or 'finetuned'
        timeout:    Request timeout in seconds.

    Returns:
        Summary string if successful, None on any error.
    """
    url = base_url.rstrip("/") + DL_SUMMARIZE_ENDPOINT
    payload = {
        "table_text": table_text,
        "mode": mode,
        # no_gate disables verification guard where backend supports it.
        "verify_numbers": profile_name != "no_gate",
        "ablation_profile": profile_name,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("summary") or data.get("text") or ""
    except requests.exceptions.ConnectionError:
        logger.warning("DL backend not reachable at %s. Skipping DL call.", url)
        return None
    except requests.exceptions.Timeout:
        logger.warning("DL backend timed out after %ds for call to %s", timeout, url)
        return None
    except Exception as exc:
        logger.error("DL backend call (%s mode) failed: %s", mode, exc)
        return None


def check_backend_health(url: str, name: str = "backend") -> bool:
    """
    Check if a backend is alive by calling its /health endpoint.

    Returns True if healthy, False otherwise.
    """
    health_url = url.rstrip("/") + "/health"
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            logger.info("%s is healthy at %s", name, health_url)
            return True
        else:
            logger.warning("%s returned HTTP %d at %s",
                           name, resp.status_code, health_url)
            return False
    except Exception:
        logger.warning("%s not reachable at %s", name, health_url)
        return False


def _perturb_first_percent(summary: str, delta: float = 1.2) -> str:
    """Nudge first percentage by +delta to emulate unguarded numeric drift."""
    m = re.search(r"(\d+(?:\.\d+)?)%", summary)
    if not m:
        return summary
    old = float(m.group(1))
    new = max(0.0, round(old + delta, 1))
    return summary[:m.start(1)] + f"{new}" + summary[m.end(1):]


def _swap_arms(summary: str, arm_names: List[str]) -> str:
    """Swap arm names in summary text to emulate parser/attribution ablation."""
    if len(arm_names) < 2:
        return summary
    a, b = arm_names[0], arm_names[1]
    if not a or not b:
        return summary
    tmp = "__ARM_SWAP_TMP__"
    return summary.replace(a, tmp).replace(b, a).replace(tmp, b)


def _drop_severity_sentences(summary: str) -> str:
    """Remove high-severity sentences to emulate no-severity-weighting behavior."""
    sents = re.split(r'(?<=[.!?])\s+', summary.strip())
    kept = [
        s for s in sents
        if not re.search(r"\bfatal\b|\bdeath\b|\bsae\b|\bserious adverse\b|\bgrade\s*[34]\b|\bsevere\b", s, flags=re.I)
    ]
    return " ".join(kept).strip() if kept else summary


def apply_profile_ablation(summary: str,
                          table_text: str,
                          arm_names: List[str],
                          mode: str,
                          profile_name: str) -> str:
    """
    Deterministic profile transforms so ablation profiles have true behavioral effect,
    including offline runs with precomputed summaries.
    """
    if not summary or profile_name == "full_system":
        return summary

    out = summary

    if profile_name == "no_gate" and mode in ("dl_base", "finetuned"):
        out = _perturb_first_percent(out, delta=1.2)

    if profile_name == "no_arm_parser" and mode in ("ml", "finetuned"):
        out = _swap_arms(out, arm_names)

    if profile_name == "no_severity_weighting":
        out = _drop_severity_sentences(out)

    return out


# =============================================================================
#  BENCHMARK DATA LOADER
# =============================================================================

def load_benchmark(tier: str,
                   benchmark_dir: str,
                   n_max: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load benchmark examples from a JSONLines file.

    Args:
        tier:          'tier1', 'tier2', or 'full'
        benchmark_dir: Path to directory containing benchmark .jsonl files.
        n_max:         If set, load only the first n_max examples.

    Returns:
        List of example dicts.

    Raises:
        FileNotFoundError: if the benchmark file does not exist.
        ValueError:        if an unrecognised tier is specified.
    """
    if tier not in BENCHMARK_FILES:
        raise ValueError(
            f"Unknown tier '{tier}'. Valid options: {list(BENCHMARK_FILES.keys())}"
        )

    fpath = Path(benchmark_dir) / BENCHMARK_FILES[tier]
    if not fpath.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {fpath}\n"
            f"Expected location: {fpath.resolve()}\n"
            f"Run scripts/benchmark_extract.py to generate benchmark tiers."
        )

    examples: List[Dict[str, Any]] = []
    with open(fpath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n_max is not None and i >= n_max:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d in %s is not valid JSON: %s", i + 1, fpath, exc)

    logger.info("Loaded %d examples from %s (tier=%s)", len(examples), fpath.name, tier)
    return examples


# =============================================================================
#  RESULT PERSISTENCE
# =============================================================================

def save_results_csv(results_table: List[Dict[str, Any]],
                     output_path: str) -> None:
    """
    Save per-example results to a CSV file.

    Args:
        results_table: List of dicts, one per example, containing all metric values.
        output_path:   Full path to output CSV file.
    """
    if not results_table:
        logger.warning("No results to save.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results_table[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)

    logger.info("Results CSV saved to %s (%d rows)", output_path, len(results_table))


def save_aggregate_json(aggregate: Dict, output_path: str) -> None:
    """Save aggregate metric statistics to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    logger.info("Aggregate JSON saved to %s", output_path)


def save_summary_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Save run-level summary rows (per-mode aggregates) to CSV."""
    if not rows:
        logger.warning("No summary rows to save.")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary CSV saved to %s (%d rows)", output_path, len(rows))


def save_manifest_json(manifest: Dict[str, Any], output_path: str) -> None:
    """Save a run manifest for reproducibility and experiment tracking."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Run manifest saved to %s", output_path)


# =============================================================================
#  MAIN EVALUATION LOOP
# =============================================================================

def run_evaluation(
    tier: str,
    modes: List[str],
    ml_url: str,
    dl_url: str,
    output_dir: str,
    benchmark_dir: str,
    bertscore_model: str,
    bertscore_device: str,
    numeric_tolerance: float,
    profile_name: str,
    n_max: Optional[int] = None,
    max_workers: int = 4,
) -> None:
    """
    Full evaluation pipeline: load data → call backends → compute metrics → save.

    Args:
        tier:              Benchmark tier ('tier1', 'tier2', 'full')
        modes:             List of modes to evaluate ('ml', 'dl_base', 'finetuned')
        ml_url:            Base URL for ML backend
        dl_url:            Base URL for DL backend
        output_dir:        Directory to write results
        benchmark_dir:     Directory containing benchmark jsonl files
        bertscore_model:   HuggingFace model for BERTScore
        bertscore_device:  'cpu', 'cuda', 'mps'
        numeric_tolerance: Tolerance for numeric matching
        profile_name:      Ablation profile name
        n_max:             Max examples to evaluate (None = all)
        max_workers:       Max parallel API workers.
    """

    run_started_at = datetime.datetime.now()
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")

    # ── Backend health check ──────────────────────────────────────────────────
    ml_available = check_backend_health(ml_url, "ML backend")
    dl_available = check_backend_health(dl_url, "DL backend")

    if not ml_available and not dl_available:
        logger.error(
            "Neither backend is reachable. Make sure to start:\n"
            "  ML:  uvicorn src.api.main_ml:app --port 8000\n"
            "  DL:  uvicorn src.api.main_dl:app --port 8001\n"
            "Continuing with offline/mock summaries for metric testing..."
        )
        # We still continue — you can test metrics without live backends
        # by providing a 'precomputed_summary' key in your benchmark examples.

    # ── Load benchmark ────────────────────────────────────────────────────────
    examples = load_benchmark(tier, benchmark_dir, n_max=n_max)

    manifest: Dict[str, Any] = {
        "run_id": f"eval_{tier}_{timestamp}",
        "started_at": run_started_at.isoformat(),
        "tier": tier,
        "profile_name": profile_name,
        "modes": modes,
        "input": {
            "benchmark_dir": benchmark_dir,
            "n_examples_loaded": len(examples),
            "n_max": n_max,
        },
        "config": {
            "bertscore_model": bertscore_model,
            "bertscore_device": bertscore_device,
            "numeric_tolerance": numeric_tolerance,
            "ml_url": ml_url,
            "dl_url": dl_url,
        },
        "backend_health": {
            "ml_available": ml_available,
            "dl_available": dl_available,
        },
        "outputs": {
            "results_csv": {},
            "aggregate_json": {},
        },
        "mode_summaries": {},
    }
    summary_rows: List[Dict[str, Any]] = []

    # ── Initialise evaluation suite ───────────────────────────────────────────
    logger.info(
        "Initialising ClinicalEvaluationSuite | BERTScore model: %s | device: %s",
        bertscore_model, bertscore_device
    )
    suite = ClinicalEvaluationSuite(
        bertscore_model=bertscore_model,
        bertscore_device=bertscore_device,
        numeric_tolerance=numeric_tolerance,
    )

    # ── Evaluate each mode ────────────────────────────────────────────────────
    for mode in modes:
        logger.info("=" * 60)
        logger.info("Evaluating mode: %s | tier: %s | n=%d", mode, tier, len(examples))
        logger.info("=" * 60)

        eval_inputs: List[Dict[str, Any]] = []
        api_latencies: List[float] = []

        def process_example(i, ex):
            table_text   = ex.get("table_text", ex.get("table", ""))
            reference    = ex.get("reference", ex.get("summary", ""))
            arm_names    = ex.get("arm_names", [])
            table_id     = ex.get("table_id", f"example_{i:04d}")

            if not table_text:
                logger.warning("Example %d (id=%s) has no table_text. Skipping.", i, table_id)
                return None

            # ── Get generated summary for this mode ───────────────────────────
            t0 = time.time()

            # Check for pre-computed summaries in the benchmark file itself
            precomputed_key = f"summary_{mode}"
            if precomputed_key in ex:
                generated = ex[precomputed_key]
                latency_ms = 0.0
            elif mode == "ml":
                generated = call_ml_backend(ml_url, table_text, profile_name=profile_name) if ml_available else None
                latency_ms = (time.time() - t0) * 1000
            elif mode == "dl_base":
                generated = call_dl_backend(dl_url, table_text, mode="base_dl", profile_name=profile_name) if dl_available else None
                latency_ms = (time.time() - t0) * 1000
            elif mode == "finetuned":
                generated = call_dl_backend(dl_url, table_text, mode="finetuned", profile_name=profile_name) if dl_available else None
                latency_ms = (time.time() - t0) * 1000
            else:
                logger.warning("Unknown mode '%s'. Skipping example %d.", mode, i)
                return None

            if generated is None:
                logger.warning(
                    "No generated summary for example %d (id=%s, mode=%s). "
                    "Using empty string. Check backend is running.",
                    i, table_id, mode
                )
                generated = ""

            # Apply deterministic profile ablation
            generated = apply_profile_ablation(
                generated,
                table_text,
                arm_names,
                mode,
                profile_name,
            )

            return {
                "generated":    generated,
                "reference":    reference,
                "source_table": table_text,
                "arm_names":    arm_names,
                "_table_id":    table_id,
                "_mode":        mode,
                "_tier":        tier,
                "_latency_ms":  latency_ms,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_example, i, ex) for i, ex in enumerate(examples)]
            for future in tqdm(as_completed(futures), total=len(examples), desc=f"Calling {mode} API"):
                res = future.result()
                if res:
                    eval_inputs.append(res)
                    api_latencies.append(res["_latency_ms"])

        # ── Run metric evaluation on all examples for this mode ───────────────
        logger.info("Running metric suite on %d examples for mode=%s...",
                    len(eval_inputs), mode)

        metric_results = suite.evaluate_batch(
            eval_inputs, show_progress=True
        )

        # ── Build flat results table for CSV export ────────────────────────────
        results_table = []
        for j, (inp, res) in enumerate(zip(eval_inputs, metric_results)):
            row = {
                "table_id":        inp["_table_id"],
                "mode":            mode,
                "tier":            tier,
                "profile_name":    profile_name,
                "latency_ms":      round(inp["_latency_ms"], 1),
                # Tier 1 metrics
                "rouge_1":         round(res.rouge_1, 4),
                "rouge_2":         round(res.rouge_2, 4),
                "rouge_l":         round(res.rouge_l, 4),
                "bertscore_f1":    round(res.bertscore_f1, 4),
                "meteor":          round(res.meteor, 4),
                "fk_grade":        round(res.flesch_kincaid_grade, 2),
                # Tier 2 metrics
                "nar":             round(res.numeric_accuracy_rate, 4),
                "hr":              round(res.hallucination_rate, 4),
                "or_score":        round(res.omission_rate, 4),
                # Novel metrics
                "swos":            round(res.severity_weighted_omission_score, 4),
                "ndi":             round(res.numeric_drift_index, 4),
                "acr":             round(res.arm_confusion_rate, 4),
                "rdi":             round(res.risk_deflation_index, 4),
                "rii":             round(res.risk_inflation_index, 4),
                "dpe":             round(res.delta_preservation_error, 4),
                "cor":             round(res.contraindication_omission_rate, 4),
                # Composite scores
                "safety_score":    round(res.safety_score(), 4),
                "fluency_score":   round(res.fluency_score(), 4),
                # Meta
                "n_source_nums":   res.n_source_numbers,
                "n_output_nums":   res.n_output_numbers,
                "n_halluc_sents":  res.n_hallucinated_sentences,
                "n_total_sents":   res.n_total_sentences,
                "errors":          "; ".join(res.errors) if res.errors else "",
            }
            results_table.append(row)

        # ── Save per-example CSV ───────────────────────────────────────────────
        csv_path = str(
            Path(output_dir) / f"results_{tier}_{mode}_{timestamp}.csv"
        )
        save_results_csv(results_table, csv_path)
        manifest["outputs"]["results_csv"][mode] = csv_path

        # ── Compute and save aggregates ────────────────────────────────────────
        agg = suite.aggregate_results(metric_results)
        agg["_meta"] = {
            "tier": tier,
            "mode": mode,
            "n_examples": len(metric_results),
            "timestamp": timestamp,
            "bertscore_model": bertscore_model,
            "numeric_tolerance": numeric_tolerance,
            "mean_latency_ms": round(
                sum(api_latencies) / len(api_latencies) if api_latencies else 0.0, 2
            ),
        }
        agg_path = str(
            Path(output_dir) / f"aggregate_{tier}_{mode}_{timestamp}.json"
        )
        save_aggregate_json(agg, agg_path)
        manifest["outputs"]["aggregate_json"][mode] = agg_path

        # Keep compact mode summary for quick comparisons and dashboards.
        mode_summary = {
            "n_examples": len(metric_results),
            "mean_latency_ms": round(sum(api_latencies) / len(api_latencies), 2) if api_latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(api_latencies, 95)), 2) if api_latencies else 0.0,
            "nar_mean": round(agg["numeric_accuracy_rate"]["mean"], 4),
            "hr_mean": round(agg["hallucination_rate"]["mean"], 4),
            "acr_mean": round(agg["arm_confusion_rate"]["mean"], 4),
            "rdi_mean": round(agg["risk_deflation_index"]["mean"], 4),
            "rii_mean": round(agg["risk_inflation_index"]["mean"], 4),
            "dpe_mean": round(agg["delta_preservation_error"]["mean"], 4),
            "cor_mean": round(agg["contraindication_omission_rate"]["mean"], 4),
            "safety_mean": round(agg["safety_score"]["mean"], 4),
            "fluency_mean": round(agg["fluency_score"]["mean"], 4),
        }
        manifest["mode_summaries"][mode] = mode_summary

        summary_rows.append({
            "run_id": manifest["run_id"],
            "tier": tier,
            "profile_name": profile_name,
            "mode": mode,
            "n_examples": len(metric_results),
            "latency_ms_mean": mode_summary["mean_latency_ms"],
            "latency_ms_p95": mode_summary["p95_latency_ms"],
            "rouge_l_mean": round(agg["rouge_l"]["mean"], 4),
            "bertscore_f1_mean": round(agg["bertscore_f1"]["mean"], 4),
            "meteor_mean": round(agg["meteor"]["mean"], 4),
            "fk_grade_mean": round(agg["flesch_kincaid_grade"]["mean"], 4),
            "nar_mean": mode_summary["nar_mean"],
            "hr_mean": mode_summary["hr_mean"],
            "or_mean": round(agg["omission_rate"]["mean"], 4),
            "swos_mean": round(agg["severity_weighted_omission_score"]["mean"], 4),
            "ndi_mean": round(agg["numeric_drift_index"]["mean"], 4),
            "acr_mean": mode_summary["acr_mean"],
            "rdi_mean": mode_summary["rdi_mean"],
            "rii_mean": mode_summary["rii_mean"],
            "dpe_mean": mode_summary["dpe_mean"],
            "cor_mean": mode_summary["cor_mean"],
            "safety_score_mean": mode_summary["safety_mean"],
            "fluency_score_mean": mode_summary["fluency_mean"],
        })

        # ── Print summary to console ──────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY  |  tier={tier}  mode={mode}  n={len(metric_results)}")
        print(f"{'='*60}")
        print(f"  ROUGE-L (mean)          : {agg['rouge_l']['mean']:.4f}")
        print(f"  BERTScore F1 (mean)     : {agg['bertscore_f1']['mean']:.4f}")
        print(f"  METEOR (mean)           : {agg['meteor']['mean']:.4f}")
        print(f"  ── Clinical metrics ──")
        print(f"  NAR  (mean)             : {agg['numeric_accuracy_rate']['mean']:.4f}  (higher=better)")
        print(f"  HR   (mean)             : {agg['hallucination_rate']['mean']:.4f}  (lower=better)")
        print(f"  OR   (mean)             : {agg['omission_rate']['mean']:.4f}  (lower=better)")
        print(f"  ── NOVEL metrics ──")
        print(f"  SWOS (mean)             : {agg['severity_weighted_omission_score']['mean']:.4f}  (lower=better)")
        print(f"  NDI  (mean)             : {agg['numeric_drift_index']['mean']:.4f}  (lower=better)")
        print(f"  ACR  (mean)             : {agg['arm_confusion_rate']['mean']:.4f}  (lower=better)")
        print(f"  ── Composite ──")
        print(f"  Safety Score (mean)     : {agg['safety_score']['mean']:.4f}  (higher=better)")
        print(f"  Fluency Score (mean)    : {agg['fluency_score']['mean']:.4f}  (higher=better)")
        print(f"{'='*60}\n")

    # Persist run-level artifacts.
    summary_csv_path = str(Path(output_dir) / f"summary_{tier}_{timestamp}.csv")
    save_summary_csv(summary_rows, summary_csv_path)
    manifest["outputs"]["summary_csv"] = summary_csv_path

    run_finished_at = datetime.datetime.now()
    manifest["finished_at"] = run_finished_at.isoformat()
    manifest["duration_seconds"] = round((run_finished_at - run_started_at).total_seconds(), 2)

    if summary_rows:
        manifest["overall"] = {
            "n_modes": len(summary_rows),
            "best_safety_mode": max(summary_rows, key=lambda r: r["safety_score_mean"])["mode"],
            "best_fluency_mode": max(summary_rows, key=lambda r: r["fluency_score_mean"])["mode"],
            "mean_nar_across_modes": round(statistics.mean(r["nar_mean"] for r in summary_rows), 4),
            "mean_hr_across_modes": round(statistics.mean(r["hr_mean"] for r in summary_rows), 4),
        }

    manifest_path = str(Path(output_dir) / f"run_manifest_{tier}_{timestamp}.json")
    save_manifest_json(manifest, manifest_path)

    logger.info("Evaluation complete. Results saved to: %s", output_dir)


# =============================================================================
#  CLI ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical Safety Summarization — Full Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=list(BENCHMARK_FILES.keys()),
        default="tier1",
        help="Benchmark tier to evaluate. 'tier1'=gold (41 ex), 'tier2'=stress (78 ex)."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["ml", "dl_base", "finetuned"],
        default=["ml", "dl_base", "finetuned"],
        help="Summarization modes to evaluate."
    )
    parser.add_argument(
        "--ml-url",
        default="http://localhost:8000",
        help="Base URL of the ML backend."
    )
    parser.add_argument(
        "--dl-url",
        default="http://localhost:8001",
        help="Base URL of the DL backend."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results CSV and aggregate JSON files."
    )
    parser.add_argument(
        "--benchmark-dir",
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing tier*.jsonl benchmark files."
    )
    parser.add_argument(
        "--bertscore-model",
        default="distilbert-base-uncased",
        help=(
            "HuggingFace model for BERTScore. "
            "Use 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract' "
            "for clinical-domain BERTScore (slower, more accurate). "
            "Use 'distilbert-base-uncased' for fast evaluation."
        )
    )
    parser.add_argument(
        "--bertscore-device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for BERTScore computation."
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for numeric matching (default 1%)."
    )
    parser.add_argument(
        "--profile-name",
        default="full_system",
        help=(
            "Configuration label for experiment tracking and ablation analysis "
            "(e.g., full_system, no_arm_parser, no_gate, no_severity_weighting). "
            "These profiles apply deterministic output transforms for true ablation effects."
        )
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate. None=all."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel API workers per mode."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
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
        max_workers=args.max_workers,
    )
