"""
Feature Ablation Analysis for Clinical Safety Summarization
===========================================================

Purpose
-------
Analyze multiple run summary CSV files produced by run_full_evaluation.py
(using different --profile-name values), compute deltas vs baseline profile,
and generate ablation-focused tables and figures.

Inputs
------
Summary CSV files in data/eval_results named like:
    summary_<tier>_<timestamp>.csv
Each row should include:
    tier, profile_name, mode, nar_mean, hr_mean, acr_mean,
    safety_score_mean, fluency_score_mean, latency_ms_mean

Usage
-----
python ablation_analysis.py \
  --results-dir data/eval_results \
  --figures-dir data/figures \
  --tier tier1 \
  --mode finetuned \
  --baseline-profile full_system
"""

from __future__ import annotations

import argparse
import datetime
import glob
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

COLORS = {
    "baseline": "#1D3557",
    "gain": "#2D6A4F",
    "loss": "#D62828",
    "neutral": "#ADB5BD",
}


def _save(fig: plt.Figure, output_dir: str, stem: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for fmt in ("png", "svg"):
        out = str(Path(output_dir) / f"{stem}.{fmt}")
        fig.savefig(
            out,
            dpi=180,
            bbox_inches=None,
            facecolor="white",
            edgecolor="none",
        )
        log.info("Saved ablation figure: %s", out)
    plt.close(fig)


def load_summary_rows(results_dir: str, tier: str) -> pd.DataFrame:
    pattern = str(Path(results_dir) / f"summary_{tier}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No summary files found for pattern: {pattern}")

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d["_source_file"] = Path(f).name
            dfs.append(d)
        except Exception as exc:
            log.warning("Failed to read %s: %s", f, exc)

    if not dfs:
        raise RuntimeError("No readable summary CSV files found.")

    all_df = pd.concat(dfs, ignore_index=True)

    # Backward compatibility with summary files generated before profile tracking.
    if "profile_name" not in all_df.columns:
        all_df["profile_name"] = "full_system"

    required = {
        "tier", "profile_name", "mode", "nar_mean", "hr_mean", "acr_mean",
        "safety_score_mean", "fluency_score_mean", "latency_ms_mean"
    }
    missing = required - set(all_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary files: {sorted(missing)}")

    return all_df


def dedupe_latest_per_profile_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Use latest row per (profile_name, mode) based on source file timestamp suffix."""
    d = df.copy()
    d["_ts"] = d["_source_file"].str.extract(r"_(\d{8}_\d{6})\.csv$")[0].fillna("00000000_000000")
    d = d.sort_values("_ts")
    d = d.groupby(["profile_name", "mode"], as_index=False).tail(1)
    return d.drop(columns=["_ts"])


def build_ablation_overview(df: pd.DataFrame, baseline_profile: str, mode: str) -> pd.DataFrame:
    view = df[df["mode"] == mode].copy()
    if view.empty:
        raise ValueError(f"No rows found for mode={mode}")

    if baseline_profile not in set(view["profile_name"]):
        raise ValueError(
            f"Baseline profile '{baseline_profile}' not found. Available: {sorted(view['profile_name'].unique())}"
        )

    base = view[view["profile_name"] == baseline_profile].iloc[0]

    rows: List[Dict] = []
    for _, r in view.iterrows():
        rows.append({
            "profile_name": r["profile_name"],
            "mode": mode,
            "tier": r["tier"],
            "n_examples": int(r["n_examples"]),
            "latency_ms_mean": float(r["latency_ms_mean"]),
            "nar_mean": float(r["nar_mean"]),
            "hr_mean": float(r["hr_mean"]),
            "acr_mean": float(r["acr_mean"]),
            "safety_score_mean": float(r["safety_score_mean"]),
            "fluency_score_mean": float(r["fluency_score_mean"]),
            "delta_safety": float(r["safety_score_mean"] - base["safety_score_mean"]),
            "delta_fluency": float(r["fluency_score_mean"] - base["fluency_score_mean"]),
            "delta_nar": float(r["nar_mean"] - base["nar_mean"]),
            "delta_hr": float(r["hr_mean"] - base["hr_mean"]),
            "delta_acr": float(r["acr_mean"] - base["acr_mean"]),
            "delta_latency_ms": float(r["latency_ms_mean"] - base["latency_ms_mean"]),
            "baseline_profile": baseline_profile,
        })

    out = pd.DataFrame(rows).sort_values("delta_safety", ascending=False)
    return out


def build_long_delta_table(overview: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("safety_score_mean", "delta_safety", "higher_better"),
        ("fluency_score_mean", "delta_fluency", "higher_better"),
        ("nar_mean", "delta_nar", "higher_better"),
        ("hr_mean", "delta_hr", "lower_better"),
        ("acr_mean", "delta_acr", "lower_better"),
        ("latency_ms_mean", "delta_latency_ms", "lower_better"),
    ]

    rows: List[Dict] = []
    for _, r in overview.iterrows():
        for metric, delta_col, direction in metrics:
            base = overview[overview["profile_name"] == r["baseline_profile"]][metric].iloc[0]
            val = r[metric]
            delta = r[delta_col]
            pct = (delta / base * 100.0) if abs(base) > 1e-12 else 0.0
            rows.append({
                "profile_name": r["profile_name"],
                "metric": metric,
                "direction": direction,
                "value": float(val),
                "delta_vs_baseline": float(delta),
                "pct_delta_vs_baseline": float(pct),
            })

    return pd.DataFrame(rows)


def figure11_safety_delta(overview: pd.DataFrame, output_dir: str, tier: str, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    d = overview.copy()
    x = np.arange(len(d))
    vals = d["delta_safety"].to_numpy()
    colors = [COLORS["neutral"] if abs(v) < 1e-12 else (COLORS["gain"] if v > 0 else COLORS["loss"]) for v in vals]

    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.axhline(0.0, color="#666", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(d["profile_name"].tolist(), rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Delta Safety Score vs Baseline")
    ax.set_title(f"Figure 11: Ablation Impact on Safety | tier={tier} mode={mode}")

    for i, v in enumerate(vals):
        ax.text(i, v + (0.003 if v >= 0 else -0.005), f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    _save(fig, output_dir, "figure11_ablation_safety_delta")


def figure12_ablation_metric_deltas(overview: pd.DataFrame, output_dir: str, tier: str, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))

    d = overview[overview["profile_name"] != overview["baseline_profile"]].copy()
    if d.empty:
        log.warning("Only baseline profile available; rendering placeholder figure12.")
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No ablation profiles found beyond baseline.\n"
            "Run additional evaluations with --profile-name values\n"
            "(e.g., no_arm_parser, no_gate, no_severity_weighting)\n"
            "then rerun ablation_analysis.py.",
            ha="center",
            va="center",
            fontsize=11,
            color="#444",
        )
        ax.set_title(f"Figure 12: Ablation Metric Deltas | tier={tier} mode={mode}")
        _save(fig, output_dir, "figure12_ablation_metric_deltas")
        return

    profiles = d["profile_name"].tolist()
    x = np.arange(len(profiles))
    w = 0.24

    ax.bar(x - w, d["delta_nar"], width=w, color="#2D6A4F", label="Δ NAR")
    ax.bar(x, d["delta_hr"], width=w, color="#E76F51", label="Δ HR")
    ax.bar(x + w, d["delta_acr"], width=w, color="#8338EC", label="Δ ACR")

    ax.axhline(0.0, color="#666", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Delta vs Baseline")
    ax.set_title(f"Figure 12: Ablation Metric Deltas | tier={tier} mode={mode}")
    ax.legend(fontsize=8, framealpha=0.7)

    _save(fig, output_dir, "figure12_ablation_metric_deltas")


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature ablation analyzer")
    ap.add_argument("--results-dir", default="data/eval_results")
    ap.add_argument("--figures-dir", default="data/figures")
    ap.add_argument("--tier", default="tier1", choices=["tier1", "tier2", "full"])
    ap.add_argument("--mode", default="finetuned", choices=["ml", "dl_base", "finetuned"])
    ap.add_argument("--baseline-profile", default="full_system")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_summary_rows(args.results_dir, args.tier)
    df = dedupe_latest_per_profile_mode(df)

    overview = build_ablation_overview(df, args.baseline_profile, args.mode)
    long_delta = build_long_delta_table(overview)

    out_dir = Path(args.results_dir)
    overview_path = out_dir / f"ablation_overview_{args.tier}_{args.mode}_{ts}.csv"
    delta_path = out_dir / f"ablation_deltas_{args.tier}_{args.mode}_{ts}.csv"

    overview.to_csv(overview_path, index=False)
    long_delta.to_csv(delta_path, index=False)

    log.info("Ablation overview saved: %s", overview_path)
    log.info("Ablation deltas saved: %s", delta_path)

    figure11_safety_delta(overview, args.figures_dir, args.tier, args.mode)
    figure12_ablation_metric_deltas(overview, args.figures_dir, args.tier, args.mode)

    log.info("Ablation analysis complete.")


if __name__ == "__main__":
    main()
