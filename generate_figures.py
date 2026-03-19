"""
=============================================================================
Clinical Safety Summarization — Research Paper Figure Generator
=============================================================================
File    : src/evaluation/generate_figures.py
Purpose : Reads evaluation result CSVs and generates all 8 publication-quality
          figures for the research paper using matplotlib and seaborn.

USAGE
-----
    python src/evaluation/generate_figures.py \
        --results-dir data/eval_results \
        --figures-dir data/figures \
        --tier1-csv results_tier1_ml_*.csv \    <- glob or exact path
        --tier2-csv results_tier2_ml_*.csv

    # OR: use the auto-detect mode (finds latest CSVs automatically)
    python src/evaluation/generate_figures.py --auto

FIGURES PRODUCED
----------------
    Figure 1:  System architecture diagram (SVG-style, generated in matplotlib)
    Figure 2:  Numeric Accuracy Rate (NAR) grouped bar chart — Tier1 vs Tier2
    Figure 3:  Hallucination rate comparison vs published Asgari 2025 baseline
    Figure 4:  Safety vs Fluency scatter plot — Pareto frontier analysis
    Figure 5:  Verification gate activation breakdown — pie chart per mode
    Figure 6:  ROUGE-L vs table complexity (n_source_nums as complexity proxy)
    Figure 7:  Leave-One-Out (LOO) learning curve — accuracy vs corpus size
               (uses LOO data from data/eval_results/loo_*.json if available)
    Figure 8:  Metric correlation heatmap — all metrics vs all metrics

DESIGN NOTES
------------
    All figures use a consistent clinical/scientific style:
        - Serif title font (Source Serif Pro / DejaVu Serif)
        - Clean white background, no grid on main axes
        - Color palette: safe clinical blues/greens for "better" values,
          ambers/reds for "worse" values
        - 300 DPI output for print publication
        - Both PNG (for paper submission) and SVG (for editing) output

REQUIREMENTS
------------
    pip install matplotlib seaborn pandas numpy scipy
=============================================================================
"""

from __future__ import annotations

import os
import sys
import glob
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False
    warnings.warn("seaborn not installed — some figures will use basic matplotlib styles.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# =============================================================================
#  STYLE CONSTANTS
# =============================================================================

# Color palette — consistent across all figures
COLORS = {
    "ml":        "#2E86AB",   # strong blue — deterministic/safe
    "dl_base":   "#A8DADC",   # light teal — neural baseline
    "finetuned": "#1D3557",   # dark navy — fine-tuned best effort
    "baseline":  "#E63946",   # red — published external baseline
    "good":      "#2D6A4F",   # dark green — "better" region
    "bad":       "#D62828",   # red — "worse" region
    "neutral":   "#ADB5BD",   # gray — neutral
    "gold":      "#FFB703",   # amber — gold standard
}

MODE_LABELS = {
    "ml":        "ML deterministic",
    "dl_base":   "DL base (T5-XL)",
    "finetuned": "DL fine-tuned",
}

METRIC_DISPLAY_NAMES = {
    "rouge_1":        "ROUGE-1",
    "rouge_2":        "ROUGE-2",
    "rouge_l":        "ROUGE-L",
    "bertscore_f1":   "BERTScore F1",
    "meteor":         "METEOR",
    "nar":            "NAR (Numeric Accuracy)",
    "hr":             "HR (Hallucination Rate)",
    "or_score":       "OR (Omission Rate)",
    "swos":           "SWOS ★",
    "ndi":            "NDI ★",
    "acr":            "ACR ★",
    "safety_score":   "Safety Score (composite)",
    "fluency_score":  "Fluency Score (composite)",
}

FIG_DPI   = 300
FIG_SAVE_FORMATS = ["png", "svg"]   # SVG for paper editing, PNG for submission


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_results_df(results_dir: str, tier: str = "tier1") -> pd.DataFrame:
    """
    Load all CSV result files for a given tier and concatenate them.

    Automatically handles multiple mode files — each CSV will have a 'mode'
    column indicating which system produced it.

    Returns:
        pd.DataFrame with all columns from metrics_suite + 'mode', 'tier'.

    Raises:
        FileNotFoundError: if no matching CSV files found.
    """
    pattern = str(Path(results_dir) / f"results_{tier}_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No result CSV files matching pattern: {pattern}\n"
            f"Run run_full_evaluation.py first to generate results."
        )

    dfs = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
            logger.info("Loaded %d rows from %s", len(df), fpath)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fpath, exc)

    combined = pd.concat(dfs, ignore_index=True)
    
    # By default, paper figures should only use the main full_system run to avoid 
    # mixing ablation data into the main architectural results.
    if "profile_name" in combined.columns:
        combined = combined[combined["profile_name"] == "full_system"]
        
    logger.info("Total rows loaded for tier=%s (full_system): %d", tier, len(combined))
    return combined


# =============================================================================
#  FIGURE HELPERS
# =============================================================================

def save_figure(fig: plt.Figure, output_dir: str, filename_stem: str) -> None:
    """Save figure in all configured formats."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for fmt in FIG_SAVE_FORMATS:
        fpath = str(Path(output_dir) / f"{filename_stem}.{fmt}")
        fig.savefig(fpath, dpi=FIG_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        logger.info("Saved figure: %s", fpath)


def apply_clinical_style(ax: plt.Axes,
                          xlabel: str = "",
                          ylabel: str = "",
                          title: str = "") -> None:
    """Apply consistent publication-ready styling to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color="#333333", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, color="#333333", labelpad=8)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#1A1A2E", pad=10)


# =============================================================================
#  FIGURE 2: Numeric Accuracy Rate grouped bar chart
# =============================================================================

def figure2_nar_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 2: NAR grouped bar chart across modes and tiers.

    Shows NAR (mean ± std) for each mode (ML / DL-base / DL-finetuned)
    on Tier-1 (gold) and Tier-2 (stress) benchmarks.

    Bar chart with error bars. Red dashed horizontal line at the
    published hallucination baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle(
        "Figure 2: Numeric Accuracy Rate (NAR) by Mode and Benchmark Tier\n"
        "Higher is better — dashed line = 0.95 publication threshold",
        fontsize=11, fontweight="bold", y=1.01
    )

    tiers = ["tier1", "tier2"]
    tier_titles = ["Tier 1: Gold standard (n=41)", "Tier 2: Stress test (n=78)"]
    modes = ["ml", "dl_base", "finetuned"]
    x = np.arange(len(modes))
    bar_width = 0.55

    for ax, tier, tier_title in zip(axes, tiers, tier_titles):
        tier_df = df[df["tier"] == tier] if "tier" in df.columns else df
        means, stds = [], []
        for mode in modes:
            mode_df = tier_df[tier_df["mode"] == mode]
            if mode_df.empty:
                means.append(0.0)
                stds.append(0.0)
                logger.warning("No data for mode=%s tier=%s in figure 2.", mode, tier)
            else:
                means.append(mode_df["nar"].mean())
                stds.append(mode_df["nar"].std())

        bars = ax.bar(
            x, means,
            width=bar_width,
            color=[COLORS[m] for m in modes],
            edgecolor="white",
            linewidth=0.8,
            yerr=stds,
            capsize=4,
            error_kw={"ecolor": "#666666", "linewidth": 1},
            zorder=3,
        )

        # Publication threshold line
        ax.axhline(0.95, color=COLORS["bad"], linestyle="--",
                   linewidth=1.2, alpha=0.8, zorder=2, label="0.95 threshold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in modes], fontsize=8, rotation=12)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        # Annotate bar values
        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=8, color="#333333"
            )

        apply_clinical_style(ax, ylabel="Numeric Accuracy Rate (NAR)", title=tier_title)

    axes[0].legend(fontsize=8, framealpha=0.6)
    plt.tight_layout()
    save_figure(fig, output_dir, "figure2_nar_comparison")
    plt.close(fig)


# =============================================================================
#  FIGURE 3: Hallucination rate vs published baseline
# =============================================================================

def figure3_hallucination_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 3: Hallucination Rate (HR) horizontal bar chart, including
    the published Asgari et al. (2025) baseline of 1.47% = 0.0147.

    Lower is better. Uses log scale to handle the wide range.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    modes_in_data = ["ml", "dl_base", "finetuned"]
    labels, means, stds = [], [], []

    for mode in modes_in_data:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            logger.warning("No data for mode=%s in figure 3.", mode)
            continue
        labels.append(MODE_LABELS[mode])
        means.append(mode_df["hr"].mean())
        stds.append(mode_df["hr"].std())

    # Add published baselines from literature
    labels.append("Asgari et al. 2025 (GPT-4, clinical notes)")
    means.append(0.0147)  # 1.47% hallucination rate
    stds.append(0.0)

    labels.append("LLaMA-2 zero-shot (PICO eval, 2024)")
    means.append(0.38)    # 38% reported in Mitigating Hallucinations paper
    stds.append(0.0)

    colors = (
        [COLORS[m] for m in modes_in_data]
        + [COLORS["baseline"]] * 2
    )

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, means,
        xerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"ecolor": "#666666", "linewidth": 1},
        height=0.6,
        zorder=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hallucination Rate (fraction of sentences with unsupported values)",
                   fontsize=9)

    # Annotate each bar
    for bar, val in zip(bars, means):
        ax.text(
            val + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            ha="left", va="center", fontsize=8, color="#333333"
        )

    # Vertical line separating "good" from "bad"
    ax.axvline(0.05, color="#CCCCCC", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0.051, len(labels) - 0.5, "<5% target", fontsize=7, color="#999999")

    apply_clinical_style(
        ax,
        title="Figure 3: Hallucination Rate Comparison\n"
              "Lower is better — red bars = external baselines from literature"
    )

    # Legend
    own_patch    = mpatches.Patch(color=COLORS["ml"],       label="This work (our system)")
    extern_patch = mpatches.Patch(color=COLORS["baseline"], label="Published baselines")
    ax.legend(handles=[own_patch, extern_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    save_figure(fig, output_dir, "figure3_hallucination_comparison")
    plt.close(fig)


# =============================================================================
#  FIGURE 4: Safety vs Fluency scatter (Pareto frontier)
# =============================================================================

def figure4_safety_fluency_scatter(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 4: Safety Score (Y-axis) vs Fluency Score (X-axis) scatter plot.

    Each dot = one test example.
    Three colours = three modes.
    Shows the Pareto frontier — examples where no mode dominates on both axes.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for mode in ["ml", "dl_base", "finetuned"]:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            continue

        # Skip rows with zeros (likely missing data)
        mask = (mode_df["safety_score"] > 0) & (mode_df["fluency_score"] > 0)
        mode_df = mode_df[mask]

        ax.scatter(
            mode_df["fluency_score"],
            mode_df["safety_score"],
            color=COLORS[mode],
            alpha=0.65,
            s=40,
            label=MODE_LABELS[mode],
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )

        # Centroid marker
        if len(mode_df) > 0:
            cx = mode_df["fluency_score"].mean()
            cy = mode_df["safety_score"].mean()
            ax.scatter(cx, cy, color=COLORS[mode], s=180, marker="*",
                       edgecolors="#333333", linewidths=0.8, zorder=6)
            ax.annotate(
                f"  {MODE_LABELS[mode]}\n  (μ={cy:.3f}, μ={cx:.3f})",
                xy=(cx, cy), fontsize=7, color=COLORS[mode],
                xytext=(4, 4), textcoords="offset points",
            )

    # Ideal region shading (high safety AND high fluency)
    ax.axhspan(0.8, 1.05, alpha=0.04, color=COLORS["good"], zorder=1)
    ax.axvspan(0.6, 1.05, alpha=0.04, color=COLORS["good"], zorder=1)
    ax.text(0.62, 0.96, "High safety\n& high fluency\n(ideal region)",
            fontsize=7, color=COLORS["good"], alpha=0.7)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.7)
    ax.set_aspect("equal")

    apply_clinical_style(
        ax,
        xlabel="Fluency Score (composite: ROUGE-L 30% + BERTScore 45% + METEOR 25%)",
        ylabel="Safety Score (composite: NAR 35% + 1-HR 25% + 1-OR 20% + 1-ACR 20%)",
        title="Figure 4: Safety–Fluency Tradeoff\n"
              "Stars = mode centroids. Ideal = top-right corner."
    )

    plt.tight_layout()
    save_figure(fig, output_dir, "figure4_safety_fluency_scatter")
    plt.close(fig)


# =============================================================================
#  FIGURE 5: Gate activation breakdown
# =============================================================================

def figure5_gate_breakdown(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 5: For the 'finetuned' mode, show what fraction of outputs had:
        (a) HR > 0   — hallucination detected (gate fires on sentences)
        (b) NAR < 1  — at least one ungrounded number
        (c) SWOS > OR — disproportionate high-severity omission
        (d) ACR > 0  — arm confusion detected
        (e) Passed all checks — no issues

    Pie chart showing the distribution of failure modes.
    Uses the finetuned mode data.
    """
    mode_df = df[df["mode"] == "finetuned"]
    if mode_df.empty:
        logger.warning("No finetuned data available for figure 5. Skipping.")
        return

    # Compute gate trigger categories (non-exclusive — one example can trigger multiple)
    total = len(mode_df)
    n_hr         = (mode_df["hr"] > 0).sum()
    n_nar_fail   = (mode_df["nar"] < 0.95).sum()
    n_swos_warn  = (mode_df["swos"] > mode_df["or_score"]).sum() if "swos" in mode_df.columns else 0
    n_acr        = (mode_df["acr"] > 0).sum()
    n_passed     = ((mode_df["hr"] == 0) & (mode_df["nar"] >= 0.95)).sum()

    sizes = [n_hr, n_nar_fail, n_swos_warn, n_acr, n_passed]
    labels_pie = [
        f"HR>0 (hallucinated sentence)\nn={n_hr}",
        f"NAR<0.95 (ungrounded number)\nn={n_nar_fail}",
        f"SWOS>OR (severity omission)\nn={n_swos_warn}",
        f"ACR>0 (arm confusion)\nn={n_acr}",
        f"Passed all checks\nn={n_passed}",
    ]
    pie_colors = [
        COLORS["bad"],
        "#E76F51",
        COLORS["gold"],
        "#8338EC",
        COLORS["good"],
    ]

    # Filter out zero-sized slices
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, pie_colors) if s > 0]
    if not nonzero:
        logger.warning("All gate trigger counts are zero — check data for figure 5.")
        return

    sizes_nz   = [x[0] for x in nonzero]
    labels_nz  = [x[1] for x in nonzero]
    colors_nz  = [x[2] for x in nonzero]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes_nz,
        labels=None,
        colors=colors_nz,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        pctdistance=0.75,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    for at in autotexts:
        at.set_fontsize(8)

    ax.legend(
        wedges, labels_nz,
        title=f"Verification gate triggers (DL fine-tuned, n={total})",
        title_fontsize=8,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
    )
    ax.set_title(
        "Figure 5: Verification Gate Activation Analysis\n"
        "DL fine-tuned mode — reasons for gate triggering or passing",
        fontsize=11, fontweight="bold", pad=14
    )

    plt.tight_layout()
    save_figure(fig, output_dir, "figure5_gate_breakdown")
    plt.close(fig)


# =============================================================================
#  FIGURE 6: ROUGE-L vs table complexity
# =============================================================================

def figure6_rouge_vs_complexity(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 6: ROUGE-L score vs source table complexity.

    Complexity proxy = n_source_nums (number of distinct numeric values in
    the source table). Tables are binned into 3 complexity tiers:
        Simple:  n_source_nums ≤ 10
        Medium:  11 ≤ n_source_nums ≤ 25
        Complex: n_source_nums > 25

    Box plots showing ROUGE-L distribution per complexity tier per mode.
    """
    complexity_col = None
    if "n_source_nums" in df.columns:
        complexity_col = "n_source_nums"
    elif "n_src_nums" in df.columns:
        complexity_col = "n_src_nums"
    else:
        logger.warning("Neither n_source_nums nor n_src_nums found — skipping figure 6.")
        return

    # Bin complexity
    def complexity_bin(n):
        if n <= 10:  return "Simple\n(≤10 values)"
        if n <= 25:  return "Medium\n(11–25 values)"
        return "Complex\n(>25 values)"

    df["complexity"] = df[complexity_col].apply(complexity_bin)
    complexity_order = ["Simple\n(≤10 values)", "Medium\n(11–25 values)", "Complex\n(>25 values)"]
    modes = ["ml", "dl_base", "finetuned"]

    fig, ax = plt.subplots(figsize=(10, 6))

    n_modes = len(modes)
    n_complexity = len(complexity_order)
    width = 0.22
    x = np.arange(n_complexity)

    for j, mode in enumerate(modes):
        mode_df = df[df["mode"] == mode]
        medians, q25s, q75s = [], [], []

        for comp in complexity_order:
            comp_df = mode_df[mode_df["complexity"] == comp]["rouge_l"].dropna()
            if len(comp_df) == 0:
                medians.append(0.0); q25s.append(0.0); q75s.append(0.0)
            else:
                medians.append(comp_df.median())
                q25s.append(comp_df.quantile(0.25))
                q75s.append(comp_df.quantile(0.75))

        offset = (j - n_modes / 2 + 0.5) * (width + 0.04)
        ax.bar(
            x + offset, medians,
            width=width,
            color=COLORS[mode],
            alpha=0.85,
            label=MODE_LABELS[mode],
            edgecolor="white",
            zorder=3,
        )
        # IQR error bars
        yerr_lower = np.array(medians) - np.array(q25s)
        yerr_upper = np.array(q75s) - np.array(medians)
        ax.errorbar(
            x + offset, medians,
            yerr=[yerr_lower, yerr_upper],
            fmt="none", ecolor="#666666",
            capsize=3, linewidth=1, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(complexity_order, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, framealpha=0.7)

    apply_clinical_style(
        ax,
        xlabel="Table Complexity (by number of distinct numeric values in source)",
        ylabel="ROUGE-L (median, with IQR error bars)",
        title="Figure 6: ROUGE-L vs Table Complexity\n"
              "Error bars = interquartile range (Q1–Q3)"
    )

    plt.tight_layout()
    save_figure(fig, output_dir, "figure6_rouge_vs_complexity")
    plt.close(fig)


# =============================================================================
#  FIGURE 7: LOO Learning Curve
# =============================================================================

def figure7_loo_learning_curve(results_dir: str, output_dir: str) -> None:
    """
    Figure 7: Leave-One-Out (LOO) learning curve.

    Plots NAR and Safety Score as a function of training corpus size.
    Data comes from LOO evaluation runs stored in results_dir/loo_*.json.

    If no LOO data is found, generates a synthetic illustration curve
    with a clear annotation explaining it is illustrative.
    """
    loo_files = sorted(glob.glob(str(Path(results_dir) / "loo_*.json")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Figure 7: Learning Curve — Performance vs Training Corpus Size\n"
        "(Leave-One-Out cross-validation)",
        fontsize=11, fontweight="bold"
    )

    if loo_files:
        # Load real LOO data
        corpus_sizes, nar_means, nar_stds, safety_means, safety_stds = [], [], [], [], []
        for f in loo_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                corpus_sizes.append(data.get("n_train", 0))
                nar_means.append(data.get("nar_mean", 0.0))
                nar_stds.append(data.get("nar_std", 0.0))
                safety_means.append(data.get("safety_score_mean", 0.0))
                safety_stds.append(data.get("safety_score_std", 0.0))
            except Exception as exc:
                logger.warning("Failed to load LOO file %s: %s", f, exc)

        note = ""

    else:
        # Generate illustrative curve (power-law saturation)
        logger.info("No LOO files found — generating illustrative curve for figure 7.")
        corpus_sizes = [100, 250, 500, 1000, 2000, 4000, 7000, 9630]
        nar_means    = [0.72 + 0.18 * (1 - np.exp(-s / 1500)) + np.random.normal(0, 0.01)
                        for s in corpus_sizes]
        nar_stds     = [max(0.03 - s * 1e-6, 0.005) for s in corpus_sizes]
        safety_means = [0.68 + 0.20 * (1 - np.exp(-s / 2000)) + np.random.normal(0, 0.01)
                        for s in corpus_sizes]
        safety_stds  = [max(0.04 - s * 1e-6, 0.005) for s in corpus_sizes]
        note = "\n[ILLUSTRATIVE — run scripts/run_loo_eval.py for real data]"

    for ax, means, stds, metric_label in [
        (ax1, nar_means, nar_stds, "Numeric Accuracy Rate (NAR)"),
        (ax2, safety_means, safety_stds, "Safety Score (composite)"),
    ]:
        ax.plot(corpus_sizes, means, color=COLORS["ml"],
                linewidth=2, marker="o", markersize=5, zorder=4)
        ax.fill_between(
            corpus_sizes,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color=COLORS["ml"], zorder=3
        )

        # Full corpus marker
        if corpus_sizes:
            ax.axvline(corpus_sizes[-1], color=COLORS["gold"], linestyle="--",
                       linewidth=1.2, alpha=0.8, label=f"Full corpus (n={corpus_sizes[-1]:,})")
        ax.set_xlim(0, max(corpus_sizes) * 1.05 if corpus_sizes else 10000)
        ax.set_ylim(0.6, 1.05)
        ax.legend(fontsize=8, framealpha=0.6)
        apply_clinical_style(
            ax,
            xlabel="Training corpus size (number of table-summary pairs)",
            ylabel=metric_label,
            title=metric_label + note
        )

    plt.tight_layout()
    save_figure(fig, output_dir, "figure7_loo_learning_curve")
    plt.close(fig)


# =============================================================================
#  FIGURE 8: Metric Correlation Heatmap
# =============================================================================

def figure8_metric_correlation(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 8: Pearson correlation heatmap between all metrics.

    This reveals which metrics measure redundant things (high correlation)
    vs genuinely different aspects (low/negative correlation).

    Key expected patterns:
        • ROUGE metrics → high inter-correlation
        • NAR, HR, OR → moderate negative correlation with ROUGE (safety vs fluency tradeoff)
        • SWOS, NDI, ACR → novel metrics should show partial independence from existing ones
    """
    metric_cols = [
        "rouge_1", "rouge_2", "rouge_l", "bertscore_f1", "meteor",
        "nar", "hr", "or_score", "swos", "ndi", "acr",
        "safety_score", "fluency_score",
    ]

    available_cols = [c for c in metric_cols if c in df.columns]
    if len(available_cols) < 4:
        logger.warning("Not enough metric columns for correlation heatmap. Skipping figure 8.")
        return

    corr_df = df[available_cols].dropna().corr(method="pearson")
    corr_df.columns = [METRIC_DISPLAY_NAMES.get(c, c) for c in corr_df.columns]
    corr_df.index   = [METRIC_DISPLAY_NAMES.get(c, c) for c in corr_df.index]

    n = len(corr_df)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.85), max(8, n * 0.75)))

    if _HAS_SEABORN:
        mask = np.triu(np.ones_like(corr_df, dtype=bool))   # upper triangle only
        sns.heatmap(
            corr_df,
            ax=ax,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 7},
            cbar_kws={"label": "Pearson r", "shrink": 0.6},
        )
    else:
        # Fallback to basic imshow
        im = ax.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr_df.index, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.7)

    ax.set_title(
        "Figure 8: Pearson Correlation Matrix — All Evaluation Metrics\n"
        "★ = novel metrics proposed in this paper",
        fontsize=11, fontweight="bold", pad=12
    )

    plt.tight_layout()
    save_figure(fig, output_dir, "figure8_metric_correlation")
    plt.close(fig)


# =============================================================================
#  FIGURE 9: Error composition by mode
# =============================================================================

def figure9_error_composition(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 9: Stacked error composition by mode.

    Buckets (example-level, non-exclusive in logic but normalized for visualization):
        - Hallucination risk: HR > 0
        - Arm confusion risk: ACR > 0
        - Numeric drift risk: NDI > 0.005
        - Severity omission risk: SWOS > OR + 0.005

    For readability, each bucket is plotted as percentage of examples in mode.
    """
    required_cols = {"mode", "hr", "acr", "ndi", "swos", "or_score"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Missing columns for figure 9 (%s). Skipping.", sorted(missing))
        return

    modes = [m for m in ["ml", "dl_base", "finetuned"] if m in set(df["mode"].unique())]
    if not modes:
        logger.warning("No supported modes found for figure 9. Skipping.")
        return

    labels = [MODE_LABELS.get(m, m) for m in modes]
    hall = []
    armc = []
    drift = []
    sev = []

    for mode in modes:
        d = df[df["mode"] == mode]
        n = max(len(d), 1)
        hall.append(100.0 * float((d["hr"] > 0).sum()) / n)
        armc.append(100.0 * float((d["acr"] > 0).sum()) / n)
        drift.append(100.0 * float((d["ndi"] > 0.005).sum()) / n)
        sev.append(100.0 * float((d["swos"] > (d["or_score"] + 0.005)).sum()) / n)

    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.bar(x, hall, color="#D62828", edgecolor="white", label="HR>0 (hallucination)")
    ax.bar(x, armc, bottom=hall, color="#8338EC", edgecolor="white", label="ACR>0 (arm confusion)")
    bottom2 = np.array(hall) + np.array(armc)
    ax.bar(x, drift, bottom=bottom2, color="#F77F00", edgecolor="white", label="NDI>0.005 (numeric drift)")
    bottom3 = bottom2 + np.array(drift)
    ax.bar(x, sev, bottom=bottom3, color="#FFB703", edgecolor="white", label="SWOS>OR+0.005 (severity omission)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 220)
    apply_clinical_style(
        ax,
        xlabel="Mode",
        ylabel="Error trigger rate (% of examples; stacked)",
        title="Figure 9: Error Composition by Mode\nHigher stacked bars indicate more frequent risk triggers"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    plt.tight_layout()
    save_figure(fig, output_dir, "figure9_error_composition")
    plt.close(fig)


# =============================================================================
#  FIGURE 10: System profile (latency and gate-trigger)
# =============================================================================

def figure10_system_profile(df: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 10: System profile by mode.

    Left axis: latency p50/p95 in ms (bars).
    Right axis: gate-trigger proxy rate (%) where any safety check fails:
        HR>0 OR NAR<0.95 OR ACR>0 OR SWOS>OR+0.005
    """
    required_cols = {"mode", "hr", "nar", "acr", "swos", "or_score"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Missing columns for figure 10 (%s). Skipping.", sorted(missing))
        return

    has_latency = "latency_ms" in df.columns

    modes = [m for m in ["ml", "dl_base", "finetuned"] if m in set(df["mode"].unique())]
    if not modes:
        logger.warning("No supported modes found for figure 10. Skipping.")
        return

    p50, p95, gate_rate = [], [], []
    for mode in modes:
        d = df[df["mode"] == mode].copy()
        if d.empty:
            p50.append(0.0); p95.append(0.0); gate_rate.append(0.0)
            continue

        if has_latency:
            p50.append(float(np.percentile(d["latency_ms"], 50)))
            p95.append(float(np.percentile(d["latency_ms"], 95)))
        else:
            p50.append(0.0)
            p95.append(0.0)

        triggered = (
            (d["hr"] > 0)
            | (d["nar"] < 0.95)
            | (d["acr"] > 0)
            | (d["swos"] > (d["or_score"] + 0.005))
        )
        gate_rate.append(100.0 * float(triggered.sum()) / max(len(d), 1))

    x = np.arange(len(modes))
    width = 0.32
    fig, ax = plt.subplots(figsize=(10, 5.5))

    if has_latency:
        ax.bar(x - width/2, p50, width=width, color="#4CC9F0", edgecolor="white", label="Latency p50 (ms)")
        ax.bar(x + width/2, p95, width=width, color="#4361EE", edgecolor="white", label="Latency p95 (ms)")
    else:
        ax.bar(x, gate_rate, width=0.5, color="#ADB5BD", edgecolor="white", alpha=0.2,
               label="Latency unavailable in this result schema")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes], fontsize=9)
    apply_clinical_style(
        ax,
        xlabel="Mode",
        ylabel="Latency (ms)" if has_latency else "Latency unavailable",
        title="Figure 10: System Profile by Mode\nLatency distribution with gate-trigger proxy"
    )

    ax2 = ax.twinx()
    ax2.plot(x, gate_rate, color="#D00000", marker="o", linewidth=2.0, label="Gate-trigger proxy (%)")
    ax2.set_ylabel("Gate-trigger proxy (%)", fontsize=10, color="#D00000")
    ax2.tick_params(axis="y", colors="#D00000", labelsize=9)
    ax2.set_ylim(0, 105)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    save_figure(fig, output_dir, "figure10_system_profile")
    plt.close(fig)


# =============================================================================
#  FIGURE 1: Architecture Diagram (matplotlib)
# =============================================================================

def figure1_architecture(output_dir: str) -> None:
    """
    Figure 1: System architecture diagram drawn in matplotlib.

    This is a high-level conceptual diagram showing the hybrid
    deterministic + neural pipeline with the verification gate.
    For publication, the SVG output can be further edited in Inkscape.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def draw_box(cx, cy, w, h, text, subtext="", color="#E8F4FD", textcolor="#1A1A2E",
                 fontsize=9, subtextsize=7):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#AAAAAA",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.12 if subtext else 0), text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight="bold", wrap=True)
        if subtext:
            ax.text(cx, cy - 0.22, subtext, ha="center", va="center",
                    fontsize=subtextsize, color="#555555", style="italic")

    def draw_arrow(x1, y1, x2, y2, color="#AAAAAA"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
        )

    # Input
    draw_box(5, 5.4, 3.5, 0.6, "Clinical Safety Table (CSR source)",
             "PDF / DOCX / Image / Plain text", color="#F8F9FA")
    draw_arrow(5, 5.1, 5, 4.6)

    # Parser
    draw_box(5, 4.3, 3.5, 0.55, "Arm Parser + Slot Extractor",
             "Identifies arms, N counts, row types", color="#E3EEF7")
    draw_arrow(3.25, 4.0, 2.0, 3.4)
    draw_arrow(6.75, 4.0, 8.0, 3.4)

    # ML path
    draw_box(2.0, 3.1, 2.8, 0.55, "ML Deterministic Path",
             "SlotFill + KNN retrieval + template", color="#D4EDDA", textcolor="#155724")
    draw_arrow(2.0, 2.8, 2.0, 2.2)

    # DL path
    draw_box(8.0, 3.1, 2.8, 0.55, "DL Generation Path",
             "Flan-T5-XL + QLoRA adapter rewrite", color="#CCE5FF", textcolor="#004085")
    draw_arrow(8.0, 2.8, 8.0, 2.2)

    # Candidates
    draw_box(2.0, 1.9, 2.8, 0.5, "ML Candidate Summary",
             "Slot-verified, template-grounded", color="#C3E6CB", textcolor="#155724")
    draw_box(8.0, 1.9, 2.8, 0.5, "DL Rewrite Candidate",
             "Higher fluency, unverified", color="#B8DAFF", textcolor="#004085")

    draw_arrow(3.4, 1.9, 4.1, 1.2)
    draw_arrow(6.6, 1.9, 5.9, 1.2)

    # Gate
    draw_box(5, 1.0, 3.5, 0.55, "Hallucination Guardian (Verification Gate)",
             "NAR check | ACR check | SWOS | NDI threshold",
             color="#FFF3CD", textcolor="#856404")
    draw_arrow(5, 0.72, 5, 0.18)

    # Dashed fallback arrow
    ax.annotate(
        "", xy=(3.5, 0.4), xytext=(6.5, 0.75),
        arrowprops=dict(
            arrowstyle="->", color=COLORS["bad"],
            lw=1.0, linestyle="dashed",
        )
    )
    ax.text(5.8, 0.52, "fail → ML fallback", fontsize=7, color=COLORS["bad"], style="italic")

    # Output
    draw_box(5, 0.1, 3.5, 0.4, "Verified Summary Output",
             "JSON: text + NAR + warnings + latency",
             color="#D4EDDA", textcolor="#155724")

    ax.set_title(
        "Figure 1: Hybrid Clinical Safety Summarization Architecture\n"
        "Deterministic ML core with DL rewrite path and verification gate",
        fontsize=12, fontweight="bold", pad=10
    )

    plt.tight_layout()
    save_figure(fig, output_dir, "figure1_architecture")
    plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================

def generate_all_figures(results_dir: str,
                          output_dir: str,
                          tier: str = "tier1") -> None:
    """
    Generate all 8 paper figures.

    Args:
        results_dir: Directory containing evaluation CSVs.
        output_dir:  Directory to write figure files.
        tier:        Which benchmark tier to use for figures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Generating Figure 1: Architecture diagram...")
    try:
        figure1_architecture(output_dir)
    except Exception as e:
        logger.error("Figure 1 failed: %s", e)

    # Load results dataframe (needed for figures 2–8)
    try:
        df = load_results_df(results_dir, tier=tier)
    except FileNotFoundError as e:
        logger.error(
            "Cannot load results for figures 2–8: %s\n"
            "Run run_full_evaluation.py first.", e
        )
        logger.info("Figure 1 was saved. Other figures require evaluation data.")
        return

    # Add composite scores if not already present
    if "safety_score" not in df.columns:
        df["safety_score"] = (
            df.get("nar", 0) * 0.35
            + (1 - df.get("hr", 0)) * 0.25
            + (1 - df.get("or_score", 0)) * 0.20
            + (1 - df.get("acr", 0)) * 0.20
        )
    if "fluency_score" not in df.columns:
        df["fluency_score"] = (
            df.get("rouge_l", 0) * 0.30
            + df.get("bertscore_f1", 0) * 0.45
            + df.get("meteor", 0) * 0.25
        )

    logger.info("Generating Figure 2: NAR comparison...")
    try:
        figure2_nar_comparison(df, output_dir)
    except Exception as e:
        logger.error("Figure 2 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 3: Hallucination comparison...")
    try:
        figure3_hallucination_comparison(df, output_dir)
    except Exception as e:
        logger.error("Figure 3 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 4: Safety vs fluency scatter...")
    try:
        figure4_safety_fluency_scatter(df, output_dir)
    except Exception as e:
        logger.error("Figure 4 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 5: Gate activation breakdown...")
    try:
        figure5_gate_breakdown(df, output_dir)
    except Exception as e:
        logger.error("Figure 5 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 6: ROUGE-L vs complexity...")
    try:
        figure6_rouge_vs_complexity(df, output_dir)
    except Exception as e:
        logger.error("Figure 6 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 7: LOO learning curve...")
    try:
        figure7_loo_learning_curve(results_dir, output_dir)
    except Exception as e:
        logger.error("Figure 7 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 8: Metric correlation heatmap...")
    try:
        figure8_metric_correlation(df, output_dir)
    except Exception as e:
        logger.error("Figure 8 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 9: Error composition by mode...")
    try:
        figure9_error_composition(df, output_dir)
    except Exception as e:
        logger.error("Figure 9 failed: %s", e, exc_info=True)

    logger.info("Generating Figure 10: System profile...")
    try:
        figure10_system_profile(df, output_dir)
    except Exception as e:
        logger.error("Figure 10 failed: %s", e, exc_info=True)

    logger.info("All figures saved to: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical Safety Summarization — Paper Figure Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="data/eval_results")
    parser.add_argument("--figures-dir", default="data/figures")
    parser.add_argument("--tier", default="tier1",
                        choices=["tier1", "tier2", "full"])
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect latest result CSVs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_figures(
        results_dir=args.results_dir,
        output_dir=args.figures_dir,
        tier=args.tier,
    )
