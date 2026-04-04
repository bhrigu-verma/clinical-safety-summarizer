"""
script3_analyze.py  —  HinglishMath Results Analyser & Visualiser
==================================================================

PURPOSE
-------
Takes the raw evaluation results from script2 and produces:

  1. Full statistical analysis proving (or disproving) the hypothesis
  2. Publication-quality tables in multiple formats (console, CSV, JSON)
  3. A beautiful self-contained HTML report with interactive charts
  4. A LaTeX-ready results table for the paper
  5. Statistical significance tests (Fisher's exact test)

HYPOTHESIS BEING TESTED
-----------------------
H1: Accuracy(HG_065) < Accuracy(EN)  by ≥ 15 percentage points
H2: Language switching rate in HG_065 > HI by ≥ 2× 
H3: Off-target language rate in HG_065 > EN by ≥ 10pp
H4: Accuracy degradation is non-uniform across topics
     (some topics degrade more than others — the error taxonomy)

USAGE
-----
  python script3_analyze.py \
    --input hm_results/results_raw.jsonl \
    --output-dir hm_analysis

OUTPUT
------
  hm_analysis/
    report.html            beautiful interactive HTML report
    tables/
      accuracy_matrix.csv
      per_topic_accuracy.csv
      error_taxonomy.csv
      significance_tests.json
    paper_table.tex        LaTeX table ready to paste into paper
    full_stats.json        machine-readable full statistics
"""

from __future__ import annotations

import argparse
import json
import math
import csv
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ─── Utilities ───────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def pct(n: float, total: float) -> str:
    if total == 0:
        return "—"
    return f"{100*n/total:.1f}%"

def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """
    2×2 Fisher's exact test.
    Table:    correct  wrong
    group1:   a        b
    group2:   c        d
    Returns two-sided p-value approximation.
    """
    # Use scipy if available, else simple approximation
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return float(p)
    except ImportError:
        pass

    # Fallback: normal approximation for large samples
    n1 = a + b
    n2 = c + d
    N  = n1 + n2
    if N == 0 or n1 == 0 or n2 == 0:
        return 1.0
    p1 = a / n1
    p2 = c / n2
    p_pool = (a + c) / N
    denom  = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if denom == 0:
        return 1.0
    z  = (p1 - p2) / denom
    # Approximation: p ≈ 2 * (1 - Φ(|z|))
    def phi(x):
        t = 1 / (1 + 0.2316419 * abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        return 1 - (1/math.sqrt(2*math.pi)) * math.exp(-x*x/2) * poly
    return 2 * (1 - phi(abs(z)))


# ─── Data loading ────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    results = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ─── Core statistical engine ─────────────────────────────────────────────────

VARIANTS_ORDER = ["EN", "HI", "HG_030", "HG_065"]
VARIANT_LABELS = {
    "EN":     "English (CM=0.0)",
    "HI":     "Pure Hindi (CM=0.0)",
    "HG_030": "Light Hinglish (CM=0.3)",
    "HG_065": "Heavy Hinglish (CM=0.65)",
}

def compute_stats(results: list[dict]) -> dict:
    """Compute all statistics needed for paper."""

    # ── Per-variant aggregates ──────────────────────────────────────────────
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r)

    variant_stats = {}
    for v in VARIANTS_ORDER:
        rs = by_variant.get(v, [])
        if not rs:
            continue
        n         = len(rs)
        n_correct = sum(1 for r in rs if r["is_correct"])
        n_off     = sum(1 for r in rs if r.get("off_target_lang"))
        switches  = [r.get("language_switches", 0) for r in rs]
        lengths   = [r.get("response_length", 0) for r in rs]
        dist_en   = [r.get("lang_dist_en", 0) for r in rs]
        dist_hi   = [r.get("lang_dist_hi", 0) for r in rs]

        variant_stats[v] = {
            "n":                n,
            "n_correct":        n_correct,
            "accuracy":         n_correct / n,
            "n_off_target":     n_off,
            "off_target_rate":  n_off / n,
            "avg_switches":     sum(switches) / n,
            "avg_response_len": sum(lengths) / n,
            "avg_en_fraction":  sum(dist_en) / n,
            "avg_hi_fraction":  sum(dist_hi) / n,
        }

    # ── Accuracy degradation (vs EN baseline) ──────────────────────────────
    en_acc = variant_stats.get("EN", {}).get("accuracy", 1.0)
    for v, s in variant_stats.items():
        s["accuracy_drop_vs_en"] = en_acc - s["accuracy"]
        s["relative_drop_pct"]   = (en_acc - s["accuracy"]) / en_acc * 100 if en_acc > 0 else 0

    # ── Statistical significance (EN vs HG_065) ────────────────────────────
    en_rs   = by_variant.get("EN", [])
    hg65_rs = by_variant.get("HG_065", [])

    sig_tests = {}
    if en_rs and hg65_rs:
        en_c   = sum(1 for r in en_rs   if r["is_correct"])
        hg65_c = sum(1 for r in hg65_rs if r["is_correct"])
        p_val  = fisher_exact_p(
            en_c,            len(en_rs)   - en_c,
            hg65_c,          len(hg65_rs) - hg65_c,
        )
        sig_tests["EN_vs_HG065"] = {
            "en_correct":    en_c,
            "en_total":      len(en_rs),
            "hg65_correct":  hg65_c,
            "hg65_total":    len(hg65_rs),
            "p_value":       round(p_val, 6),
            "significant":   p_val < 0.05,
            "highly_sig":    p_val < 0.01,
        }

    # ── Per-topic breakdown ─────────────────────────────────────────────────
    by_topic_variant: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        by_topic_variant[(r.get("topic", "Unknown"), r["variant"])].append(r)

    topics = sorted(set(r.get("topic", "Unknown") for r in results))
    topic_matrix: dict[str, dict] = {}
    for topic in topics:
        topic_matrix[topic] = {}
        for v in VARIANTS_ORDER:
            rs = by_topic_variant.get((topic, v), [])
            if rs:
                n = len(rs)
                c = sum(1 for r in rs if r["is_correct"])
                topic_matrix[topic][v] = {"n": n, "correct": c, "accuracy": c / n}

    # ── Difficulty breakdown ────────────────────────────────────────────────
    by_diff_variant: dict[tuple, list] = defaultdict(list)
    for r in results:
        by_diff_variant[(r.get("difficulty", "?"), r["variant"])].append(r)

    diffs = sorted(set(r.get("difficulty", "?") for r in results))
    diff_matrix: dict[str, dict] = {}
    for diff in diffs:
        diff_matrix[diff] = {}
        for v in VARIANTS_ORDER:
            rs = by_diff_variant.get((diff, v), [])
            if rs:
                n = len(rs)
                c = sum(1 for r in rs if r["is_correct"])
                diff_matrix[diff][v] = {"n": n, "correct": c, "accuracy": c / n}

    # ── Error taxonomy ─────────────────────────────────────────────────────
    # For incorrect HG_065 responses, classify the failure type
    hg65_wrong = [r for r in hg65_rs if not r["is_correct"]]
    error_taxonomy: dict[str, int] = {
        "off_target_language":    sum(1 for r in hg65_wrong if r.get("off_target_lang")),
        "high_switch_count":      sum(1 for r in hg65_wrong if r.get("language_switches", 0) > 5),
        "answer_not_found":       sum(1 for r in hg65_wrong if r.get("answer_extraction") == "not_found"),
        "confidence_hedged":      sum(1 for r in hg65_wrong if r.get("confidence_markers")),
        "other_math_error":       0,  # remainder
    }
    classified = sum(v for k, v in error_taxonomy.items() if k != "other_math_error")
    error_taxonomy["other_math_error"] = max(0, len(hg65_wrong) - classified)

    # ── Language reasoning distribution ────────────────────────────────────
    lang_dist_by_variant: dict[str, dict[str, int]] = {}
    for v in VARIANTS_ORDER:
        rs  = by_variant.get(v, [])
        cnt: dict[str, int] = defaultdict(int)
        for r in rs:
            cnt[r.get("reasoning_language", "UNKNOWN")] += 1
        lang_dist_by_variant[v] = dict(cnt)

    # ── Linguistic trap analysis ───────────────────────────────────────────
    hg65_all   = by_variant.get("HG_065", [])
    trap_stats = {}
    traps_grouped: dict[str, list] = defaultdict(list)
    for r in hg65_all:
        trap = r.get("linguistic_trap", "")
        if trap:
            # Extract trap keyword (first 30 chars)
            key = trap[:40].strip()
            traps_grouped[key].append(r)
    for trap_key, trap_rs in list(traps_grouped.items())[:15]:  # top 15
        n = len(trap_rs)
        c = sum(1 for r in trap_rs if r["is_correct"])
        trap_stats[trap_key] = {"n": n, "correct": c, "accuracy": c / n if n else 0}

    return {
        "variant_stats":       variant_stats,
        "significance_tests":  sig_tests,
        "topic_matrix":        topic_matrix,
        "diff_matrix":         diff_matrix,
        "error_taxonomy":      error_taxonomy,
        "lang_dist_by_variant": lang_dist_by_variant,
        "trap_stats":          trap_stats,
        "hypothesis_verdict":  _verdict(variant_stats, sig_tests),
    }

def _verdict(variant_stats: dict, sig_tests: dict) -> dict:
    en_acc    = variant_stats.get("EN",     {}).get("accuracy", 0)
    hg65_acc  = variant_stats.get("HG_065", {}).get("accuracy", 0)
    drop      = en_acc - hg65_acc
    h1_passed = drop >= 0.15  # ≥15pp drop

    hg65_sw   = variant_stats.get("HG_065", {}).get("avg_switches", 0)
    hi_sw     = variant_stats.get("HI",     {}).get("avg_switches", 0)
    h2_passed = hg65_sw >= 2 * hi_sw if hi_sw > 0 else False

    hg65_off  = variant_stats.get("HG_065", {}).get("off_target_rate", 0)
    en_off    = variant_stats.get("EN",     {}).get("off_target_rate", 0)
    h3_passed = (hg65_off - en_off) >= 0.10

    sig       = sig_tests.get("EN_vs_HG065", {}).get("significant", False)

    return {
        "H1_accuracy_drop_>=15pp": h1_passed,
        "H2_switch_rate_2x":       h2_passed,
        "H3_off_target_+10pp":     h3_passed,
        "significant_p<0.05":      sig,
        "overall_hypothesis":      h1_passed and sig,
        "accuracy_drop_pp":        round(drop * 100, 2),
    }


# ─── Output generators ────────────────────────────────────────────────────────

def print_console_report(stats: dict) -> None:
    vs  = stats["variant_stats"]
    sig = stats["significance_tests"]
    ver = stats["hypothesis_verdict"]

    print("\n" + "═"*70)
    print("  HINGLISHMATH HYPOTHESIS VALIDATION — RESULTS")
    print("═"*70)

    # Main accuracy table
    print("\n  ACCURACY BY VARIANT\n  " + "─"*60)
    print(f"  {'Variant':<22} {'N':>6} {'Correct':>8} {'Accuracy':>9} {'Drop vs EN':>11}")
    print("  " + "─"*60)
    for v in VARIANTS_ORDER:
        s = vs.get(v)
        if not s:
            continue
        drop  = f"{'▼'}{s['accuracy_drop_vs_en']*100:.1f}pp" if s["accuracy_drop_vs_en"] > 0 else "—"
        print(f"  {VARIANT_LABELS[v]:<22} {s['n']:>6} {s['n_correct']:>8} "
              f"{s['accuracy']*100:>8.1f}% {drop:>11}")
    print("  " + "─"*60)

    # Language behavior table
    print("\n  LANGUAGE BEHAVIOR\n  " + "─"*70)
    print(f"  {'Variant':<22} {'Avg Switches':>14} {'Off-Target %':>13} {'EN frac':>8} {'HI frac':>8}")
    print("  " + "─"*70)
    for v in VARIANTS_ORDER:
        s = vs.get(v)
        if not s:
            continue
        print(f"  {VARIANT_LABELS[v]:<22} {s['avg_switches']:>14.2f} "
              f"{s['off_target_rate']*100:>12.1f}% "
              f"{s['avg_en_fraction']:>8.3f} {s['avg_hi_fraction']:>8.3f}")

    # Significance
    print("\n  STATISTICAL SIGNIFICANCE\n  " + "─"*50)
    for test_name, t in sig.items():
        sig_str = "*** (p<0.01)" if t["highly_sig"] else "* (p<0.05)" if t["significant"] else "NOT significant"
        print(f"  {test_name}: p={t['p_value']:.4f}  {sig_str}")

    # Verdict
    print("\n  HYPOTHESIS VERDICT\n  " + "─"*50)
    for h, passed in ver.items():
        mark = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {mark}  {h}: {passed}")

    print("\n" + "═"*70 + "\n")

def save_csv_tables(stats: dict, out_dir: Path) -> None:
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)
    vs = stats["variant_stats"]

    # Accuracy matrix CSV
    with (tables / "accuracy_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "N", "Correct", "Accuracy%", "Drop_vs_EN_pp",
                    "OffTarget%", "AvgSwitches", "AvgEN_frac", "AvgHI_frac"])
        for v in VARIANTS_ORDER:
            s = vs.get(v)
            if not s:
                continue
            w.writerow([
                VARIANT_LABELS[v], s["n"], s["n_correct"],
                round(s["accuracy"]*100, 2),
                round(s["accuracy_drop_vs_en"]*100, 2),
                round(s["off_target_rate"]*100, 2),
                round(s["avg_switches"], 2),
                round(s["avg_en_fraction"], 3),
                round(s["avg_hi_fraction"], 3),
            ])

    # Per-topic accuracy CSV
    with (tables / "per_topic_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Topic"] + [f"{v}_accuracy%" for v in VARIANTS_ORDER] + ["Max_Drop_pp"])
        for topic, tdata in stats["topic_matrix"].items():
            row = [topic]
            accs = []
            for v in VARIANTS_ORDER:
                acc = tdata.get(v, {}).get("accuracy")
                row.append(round(acc*100, 1) if acc is not None else "—")
                if acc is not None:
                    accs.append(acc)
            drop = (max(accs) - min(accs)) * 100 if len(accs) >= 2 else 0
            row.append(round(drop, 1))
            w.writerow(row)

    # Error taxonomy CSV
    with (tables / "error_taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Error_Type", "Count", "Percent_of_HG065_Errors"])
        hg65_wrong = sum(stats["error_taxonomy"].values())
        for err_type, count in stats["error_taxonomy"].items():
            pct_str = f"{count/hg65_wrong*100:.1f}" if hg65_wrong else "0"
            w.writerow([err_type, count, pct_str])

def save_latex_table(stats: dict, out_dir: Path) -> None:
    vs = stats["variant_stats"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Accuracy degradation in Hinglish code-mixed mathematical reasoning (HinglishMath-1K)}",
        r"\label{tab:hinglishmath_results}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Variant} & \textbf{N} & \textbf{Acc\%} & \textbf{Drop vs EN} & \textbf{Off-target\%} & \textbf{Avg Switches} \\",
        r"\hline",
    ]
    for v in VARIANTS_ORDER:
        s = vs.get(v)
        if not s:
            continue
        label = VARIANT_LABELS[v].replace("&", r"\&")
        drop  = f"{s['accuracy_drop_vs_en']*100:.1f}pp" if s["accuracy_drop_vs_en"] > 0 else "—"
        lines.append(
            f"{label} & {s['n']} & {s['accuracy']*100:.1f} & "
            f"{drop} & {s['off_target_rate']*100:.1f} & {s['avg_switches']:.2f} \\\\"
        )
    sig = stats["significance_tests"].get("EN_vs_HG065", {})
    lines += [
        r"\hline",
        r"\multicolumn{6}{l}{\small " +
        f"Fisher's exact test (EN vs HG\\_065): p={sig.get('p_value','—')}, "
        f"{'significant' if sig.get('significant') else 'not significant'}" +
        r"} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (out_dir / "paper_table.tex").write_text("\n".join(lines), encoding="utf-8")

def save_html_report(stats: dict, results: list[dict], out_dir: Path) -> None:
    vs  = stats["variant_stats"]
    ver = stats["hypothesis_verdict"]
    sig = stats["significance_tests"]

    # Build data for charts
    var_labels_js  = json.dumps([VARIANT_LABELS[v] for v in VARIANTS_ORDER if v in vs])
    acc_data_js    = json.dumps([round(vs[v]["accuracy"]*100, 2) for v in VARIANTS_ORDER if v in vs])
    switch_data_js = json.dumps([round(vs[v]["avg_switches"], 2) for v in VARIANTS_ORDER if v in vs])
    en_frac_js     = json.dumps([round(vs[v]["avg_en_fraction"]*100, 2) for v in VARIANTS_ORDER if v in vs])
    hi_frac_js     = json.dumps([round(vs[v]["avg_hi_fraction"]*100, 2) for v in VARIANTS_ORDER if v in vs])

    # Topic degradation data
    topic_names, topic_en_acc, topic_hg65_acc, topic_drops = [], [], [], []
    for topic, tdata in sorted(stats["topic_matrix"].items()):
        en_a  = tdata.get("EN",     {}).get("accuracy")
        hg65_a = tdata.get("HG_065", {}).get("accuracy")
        if en_a is not None and hg65_a is not None:
            topic_names.append(topic)
            topic_en_acc.append(round(en_a*100, 1))
            topic_hg65_acc.append(round(hg65_a*100, 1))
            topic_drops.append(round((en_a - hg65_a)*100, 1))
    # Sort by drop (descending)
    sorted_zip = sorted(zip(topic_drops, topic_names, topic_en_acc, topic_hg65_acc), reverse=True)
    if sorted_zip:
        topic_drops, topic_names, topic_en_acc, topic_hg65_acc = zip(*sorted_zip)
        topic_drops, topic_names = list(topic_drops), list(topic_names)
        topic_en_acc, topic_hg65_acc = list(topic_en_acc), list(topic_hg65_acc)

    err_labels = json.dumps(list(stats["error_taxonomy"].keys()))
    err_vals   = json.dumps(list(stats["error_taxonomy"].values()))

    verdict_color = "#16a34a" if ver.get("overall_hypothesis") else "#dc2626"
    verdict_text  = "HYPOTHESIS CONFIRMED" if ver.get("overall_hypothesis") else "HYPOTHESIS NOT YET CONFIRMED"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HinglishMath-1K — Hypothesis Validation Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; }}
  .header {{ background: #1e3a5f; color: white; padding: 2.5rem 2rem; }}
  .header h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.4rem; }}
  .header p  {{ color: #93c5fd; font-size: 0.95rem; }}
  .verdict  {{ background: {verdict_color}; color: white; padding: 1.2rem 2rem; font-size: 1.1rem; font-weight: 600; letter-spacing: 0.05em; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }}
  .card  {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; }}
  .card h2 {{ font-size: 1rem; font-weight: 600; color: #475569; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; text-align: center; }}
  .metric-card .val {{ font-size: 2rem; font-weight: 700; color: #1e3a5f; }}
  .metric-card .lbl {{ font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; }}
  .bad  {{ color: #dc2626 !important; }}
  .good {{ color: #16a34a !important; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  th    {{ background: #1e3a5f; color: white; padding: 0.6rem 0.8rem; text-align: left; font-weight: 500; }}
  td    {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #f1f5f9; }}
  tr:hover td {{ background: #f8fafc; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }}
  .badge-green  {{ background: #dcfce7; color: #166534; }}
  .badge-red    {{ background: #fee2e2; color: #991b1b; }}
  .badge-amber  {{ background: #fef3c7; color: #92400e; }}
  .section-title {{ font-size: 1.2rem; font-weight: 600; margin: 2rem 0 1rem; color: #1e293b; border-left: 4px solid #1e3a5f; padding-left: 0.75rem; }}
  .chart-wrap {{ position: relative; height: 300px; }}
  .chart-wrap-tall {{ position: relative; height: 420px; }}
  .hyp-row {{ display: flex; align-items: center; gap: 0.75rem; padding: 0.6rem 0; border-bottom: 1px solid #f1f5f9; }}
  .hyp-mark {{ font-size: 1.2rem; }}
  .hyp-text {{ font-size: 0.9rem; color: #334155; }}
  footer {{ text-align: center; padding: 2rem; color: #94a3b8; font-size: 0.8rem; }}
  @media (max-width: 768px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>HinglishMath-1K — Hypothesis Validation</h1>
  <p>Code-mixed Hindi-English inputs degrade mathematical reasoning in LLMs &nbsp;|&nbsp; Generated {utc_now()}</p>
</div>

<div class="verdict">{verdict_text} &nbsp;|&nbsp; EN→HG_065 accuracy drop: {ver.get('accuracy_drop_pp', 0):.1f} pp</div>

<div class="container">

  <!-- KPI row -->
  <div class="grid-3" style="margin-top:2rem">
    <div class="metric-card">
      <div class="val">{vs.get('EN', {{}}).get('accuracy', 0)*100:.1f}%</div>
      <div class="lbl">English Accuracy (baseline)</div>
    </div>
    <div class="metric-card">
      <div class="val bad">{vs.get('HG_065', {{}}).get('accuracy', 0)*100:.1f}%</div>
      <div class="lbl">Hinglish CM=0.65 Accuracy</div>
    </div>
    <div class="metric-card">
      <div class="val {'bad' if ver.get('accuracy_drop_pp', 0) >= 15 else 'good'}">{ver.get('accuracy_drop_pp', 0):.1f}pp</div>
      <div class="lbl">Accuracy Drop (EN→HG_065)</div>
    </div>
    <div class="metric-card">
      <div class="val">{sig.get('EN_vs_HG065', {{}}).get('p_value', 1):.4f}</div>
      <div class="lbl">p-value (Fisher's exact)</div>
    </div>
    <div class="metric-card">
      <div class="val">{vs.get('HG_065', {{}}).get('avg_switches', 0):.1f}</div>
      <div class="lbl">Avg Language Switches (HG_065)</div>
    </div>
    <div class="metric-card">
      <div class="val bad">{vs.get('HG_065', {{}}).get('off_target_rate', 0)*100:.1f}%</div>
      <div class="lbl">Off-Target Language Rate (HG_065)</div>
    </div>
  </div>

  <!-- Hypothesis checklist -->
  <div class="section-title">Hypothesis Checklist</div>
  <div class="card" style="margin-bottom:1.5rem">
    {"".join(
        f'<div class="hyp-row"><span class="hyp-mark">{"✅" if v else "❌"}</span>'
        f'<span class="hyp-text"><strong>{k}</strong>: {v}</span></div>'
        for k, v in ver.items()
    )}
  </div>

  <!-- Charts row 1 -->
  <div class="section-title">Accuracy by Variant</div>
  <div class="grid-2">
    <div class="card">
      <h2>Accuracy % by Input Variant</h2>
      <div class="chart-wrap">
        <canvas id="accChart"></canvas>
      </div>
    </div>
    <div class="card">
      <h2>Language Switches per Response</h2>
      <div class="chart-wrap">
        <canvas id="swChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Charts row 2 -->
  <div class="section-title">Language Composition of Reasoning</div>
  <div class="grid-2">
    <div class="card">
      <h2>English Fraction in Reasoning (by variant)</h2>
      <div class="chart-wrap">
        <canvas id="enFracChart"></canvas>
      </div>
    </div>
    <div class="card">
      <h2>Error Taxonomy (HG_065 wrong answers)</h2>
      <div class="chart-wrap">
        <canvas id="errChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Topic degradation -->
  <div class="section-title">Accuracy Degradation by Topic</div>
  <div class="card">
    <h2>EN vs HG_065 Accuracy — per topic (sorted by drop)</h2>
    <div class="chart-wrap-tall">
      <canvas id="topicChart"></canvas>
    </div>
  </div>

  <!-- Detailed table -->
  <div class="section-title">Full Accuracy Matrix</div>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Variant</th><th>N</th><th>Correct</th><th>Accuracy</th>
          <th>Drop vs EN</th><th>Off-Target %</th><th>Avg Switches</th>
        </tr>
      </thead>
      <tbody>
        {"".join(
            f'<tr><td>{VARIANT_LABELS[v]}</td>'
            f'<td>{vs[v]["n"]}</td>'
            f'<td>{vs[v]["n_correct"]}</td>'
            f'<td><strong>{vs[v]["accuracy"]*100:.1f}%</strong></td>'
            f'<td class="{"bad" if vs[v]["accuracy_drop_vs_en"] >= 0.15 else ""}">'
            f'{"▼"+str(round(vs[v]["accuracy_drop_vs_en"]*100,1))+"pp" if vs[v]["accuracy_drop_vs_en"]>0 else "—"}</td>'
            f'<td>{vs[v]["off_target_rate"]*100:.1f}%</td>'
            f'<td>{vs[v]["avg_switches"]:.2f}</td></tr>'
            for v in VARIANTS_ORDER if v in vs
        )}
      </tbody>
    </table>
  </div>

  <footer>HinglishMath-1K Research Benchmark &nbsp;|&nbsp; B.Tech CS Research Project &nbsp;|&nbsp; {utc_now()[:10]}</footer>
</div>

<script>
const COLORS = ['#1e3a5f', '#2563eb', '#f59e0b', '#dc2626'];
const LIGHT  = ['rgba(30,58,95,0.7)','rgba(37,99,235,0.7)','rgba(245,158,11,0.7)','rgba(220,38,38,0.7)'];

new Chart(document.getElementById('accChart'), {{
  type: 'bar',
  data: {{
    labels: {var_labels_js},
    datasets: [{{ label: 'Accuracy %', data: {acc_data_js}, backgroundColor: LIGHT, borderColor: COLORS, borderWidth: 1.5 }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ min: 0, max: 100, ticks: {{ callback: v => v+'%' }} }} }} }}
}});

new Chart(document.getElementById('swChart'), {{
  type: 'bar',
  data: {{
    labels: {var_labels_js},
    datasets: [{{ label: 'Avg Switches', data: {switch_data_js}, backgroundColor: LIGHT, borderColor: COLORS, borderWidth: 1.5 }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('enFracChart'), {{
  type: 'bar',
  data: {{
    labels: {var_labels_js},
    datasets: [
      {{ label: 'EN fraction %', data: {en_frac_js}, backgroundColor: 'rgba(30,58,95,0.7)', borderColor: '#1e3a5f', borderWidth: 1.5 }},
      {{ label: 'HI fraction %', data: {hi_frac_js}, backgroundColor: 'rgba(220,38,38,0.6)', borderColor: '#dc2626', borderWidth: 1.5 }}
    ]
  }},
  options: {{ responsive: true, maintainAspectRatio: false,
    scales: {{ y: {{ ticks: {{ callback: v => v+'%' }} }} }} }}
}});

new Chart(document.getElementById('errChart'), {{
  type: 'doughnut',
  data: {{
    labels: {err_labels},
    datasets: [{{ data: {err_vals}, backgroundColor: ['#dc2626','#f59e0b','#1e3a5f','#7c3aed','#059669'] }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false }}
}});

new Chart(document.getElementById('topicChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(topic_names[:20])},
    datasets: [
      {{ label: 'English acc%',    data: {json.dumps(topic_en_acc[:20])},   backgroundColor: 'rgba(30,58,95,0.7)' }},
      {{ label: 'HG_065 acc%',     data: {json.dumps(topic_hg65_acc[:20])}, backgroundColor: 'rgba(220,38,38,0.6)' }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    indexAxis: 'y',
    scales: {{ x: {{ min: 0, max: 100, ticks: {{ callback: v => v+'%' }} }} }}
  }}
}});
</script>
</body>
</html>"""

    (out_dir / "report.html").write_text(html, encoding="utf-8")
    print(f"  → HTML report: {out_dir / 'report.html'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="HinglishMath Results Analyser")
    p.add_argument("--input",      required=True, help="Path to results_raw.jsonl from script2")
    p.add_argument("--output-dir", default="hm_analysis")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from {args.input} …")
    results = load_results(Path(args.input))
    print(f"Loaded {len(results)} evaluation records.")

    print("Computing statistics …")
    stats = compute_stats(results)

    print_console_report(stats)

    print("Saving CSV tables …")
    save_csv_tables(stats, out)

    print("Saving LaTeX table …")
    save_latex_table(stats, out)

    print("Saving HTML report …")
    save_html_report(stats, results, out)

    print("Saving full stats JSON …")
    (out / "full_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(f"\n✓ All outputs in: {out.resolve()}")
    print(f"  Open {out / 'report.html'} in a browser to see the full visual report.\n")

if __name__ == "__main__":
    main()