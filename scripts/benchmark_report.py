"""
scripts/benchmark_report.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3 of the Benchmark Pipeline.

Reads all JSONL result files from data/benchmark/results/,
aggregates metrics, prints a terminal report, and generates
data/benchmark/benchmark_report.json + benchmark_report.pdf.

Usage:
    ./venv/bin/python3 scripts/benchmark_report.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "data" / "benchmark" / "results"
OUT_JSON    = ROOT / "data" / "benchmark" / "benchmark_report.json"
OUT_PDF     = ROOT / "data" / "benchmark" / "benchmark_report.pdf"


# ── Load + aggregate ───────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def stats(values: list) -> Dict:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=float)
    return {
        "n":    int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std":  round(float(np.std(arr)),  4),
        "min":  round(float(np.min(arr)),  4),
        "p10":  round(float(np.percentile(arr, 10)), 4),
        "p50":  round(float(np.percentile(arr, 50)), 4),
        "p90":  round(float(np.percentile(arr, 90)), 4),
        "max":  round(float(np.max(arr)),  4),
    }


def aggregate(rows: List[Dict]) -> Dict:
    if not rows:
        return {}

    def col(key, exclude_none=True):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return vals

    agg = {
        "n_total":           len(rows),
        "n_success":         sum(1 for r in rows if r.get("success")),
        "n_verified":        sum(1 for r in rows if r.get("verified")),
        "success_rate":      round(sum(1 for r in rows if r.get("success")) / len(rows), 4),
        "verified_rate":     round(sum(1 for r in rows if r.get("verified")) / len(rows), 4),
        "hallucination_rate": stats(col("hallucination_rate")),
        "inference_time_ms":  stats(col("inference_time_ms")),
        "numeric_accuracy_self": stats(col("numeric_accuracy_self")),
    }

    # Tier 1 metrics (ROUGE + vs-ref accuracy — only when reference exists)
    rouge1_vals = col("rouge1")
    if rouge1_vals:
        agg["rouge1"]  = stats(rouge1_vals)
        agg["rouge2"]  = stats(col("rouge2"))
        agg["rougeL"]  = stats(col("rougeL"))
        agg["numeric_accuracy_vs_ref"] = stats(col("numeric_accuracy_vs_ref"))
        agg["n_with_reference"] = len(rouge1_vals)
        # Roadmap thresholds
        agg["thresholds"] = {
            "p10_rouge1_gt_0.75":   agg["rouge1"]["p10"] > 0.75,
            "mean_rouge1_gt_0.85":  agg["rouge1"]["mean"] > 0.85,
            "mean_num_acc_gt_0.96": agg.get("numeric_accuracy_vs_ref", {}).get("mean", 0) > 0.96,
            "mean_hall_lt_0.04":    agg["hallucination_rate"]["mean"] < 0.04,
        }
    return agg


def discover_result_files() -> Dict[str, Dict[str, Path]]:
    """
    Returns structure: { tier_name: { model_name: path } }
    e.g. { "tier1": { "ml": Path(...), "dl_base": Path(...) }, "tier2": {...} }
    """
    found = {}
    for f in sorted(RESULTS_DIR.glob("*.jsonl")):
        # Naming: {model}_{tier}.jsonl  e.g. ml_tier1.jsonl, dl_base_tier2.jsonl
        name = f.stem
        parts = name.split("_", 1)   # split on first _
        if len(parts) == 2:
            model_part, tier_part = parts[0], parts[1]
            # handle dl_base and dl_ft (2-part model names)
            if name.startswith("dl_base_"):
                model_part = "dl_base"
                tier_part  = name[len("dl_base_"):]
            elif name.startswith("dl_ft_"):
                model_part = "dl_ft"
                tier_part  = name[len("dl_ft_"):]
            else:
                model_part = parts[0]
                tier_part  = parts[1]
            found.setdefault(tier_part, {})[model_part] = f
    return found


# ── Terminal report ────────────────────────────────────────────────────────────

MODEL_LABELS = {
    "ml":      "ML Engine (LightGBM + SlotFill)",
    "dl_base": "DL Base (Flan-T5 unfinetuned)",
    "dl_ft":   "DL Fine-tuned (Flan-T5 + LoRA adapter)",
}

def print_report(all_aggs: Dict):
    WIDTH = 68
    print("\n" + "=" * WIDTH)
    print("  CLINICALSAFE BENCHMARK REPORT")
    print("=" * WIDTH)

    for tier, models in all_aggs.items():
        print(f"\n{'─' * WIDTH}")
        tier_label = "Tier 1 — Gold Standard (with human reference)" \
                     if "tier1" in tier else "Tier 2 — Stress Test (table only)"
        print(f"  {tier_label}")
        print(f"{'─' * WIDTH}")

        for model_key, agg in models.items():
            label = MODEL_LABELS.get(model_key, model_key)
            print(f"\n  📊  {label}")
            print(f"      Total tables:    {agg.get('n_total', 0)}")
            print(f"      Success rate:    {agg.get('success_rate', 0)*100:.1f}%")
            print(f"      Verified rate:   {agg.get('verified_rate', 0)*100:.1f}%")

            hall = agg.get("hallucination_rate", {})
            if hall:
                print(f"      Hallucination:   mean={hall.get('mean',0)*100:.2f}%  "
                      f"p90={hall.get('p90',0)*100:.2f}%")

            speed = agg.get("inference_time_ms", {})
            if speed:
                print(f"      Speed (ms):      mean={speed.get('mean',0):.0f}  "
                      f"p90={speed.get('p90',0):.0f}")

            if "rouge1" in agg:
                r1 = agg["rouge1"]
                r2 = agg.get("rouge2", {})
                rl = agg.get("rougeL", {})
                na = agg.get("numeric_accuracy_vs_ref", {})
                print(f"      ROUGE-1:         mean={r1.get('mean',0):.4f}  "
                      f"p10={r1.get('p10',0):.4f}  p90={r1.get('p90',0):.4f}")
                print(f"      ROUGE-2:         mean={r2.get('mean',0):.4f}")
                print(f"      ROUGE-L:         mean={rl.get('mean',0):.4f}")
                print(f"      Num Acc (vs ref):mean={na.get('mean',0):.4f}")

                thresh = agg.get("thresholds", {})
                if thresh:
                    print(f"      Roadmap checks:")
                    checks = {
                        "p10 ROUGE-1 > 0.75":   thresh.get("p10_rouge1_gt_0.75"),
                        "mean ROUGE-1 > 0.85":  thresh.get("mean_rouge1_gt_0.85"),
                        "mean NumAcc > 0.96":   thresh.get("mean_num_acc_gt_0.96"),
                        "mean Hall < 4%":       thresh.get("mean_hall_lt_0.04"),
                    }
                    for k, v in checks.items():
                        mark = "✅" if v else "❌"
                        print(f"        {mark} {k}")

    print("\n" + "=" * WIDTH)


# ── PDF Report ─────────────────────────────────────────────────────────────────

def generate_pdf_report(all_aggs: Dict, sample_rows: Dict):
    """Generate a formatted PDF benchmark report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        print("  ⚠️  reportlab not installed — skipping PDF. pip install reportlab")
        return

    PURPLE = colors.HexColor("#7c3aed")
    GREEN  = colors.HexColor("#16a34a")
    RED    = colors.HexColor("#dc2626")
    BLUE   = colors.HexColor("#2563eb")
    GREY   = colors.HexColor("#f1f5f9")
    MID    = colors.HexColor("#cbd5e1")
    DARK   = colors.HexColor("#1e293b")
    WHITE  = colors.white

    base = getSampleStyleSheet()
    def S(n, **kw):
        return ParagraphStyle(n, parent=base["Normal"], **kw)

    h1   = S("h1", fontSize=18, leading=24, textColor=PURPLE, fontName="Helvetica-Bold")
    h2   = S("h2", fontSize=13, leading=18, textColor=BLUE,   fontName="Helvetica-Bold")
    body = S("body", fontSize=9, leading=13, textColor=DARK)
    mono = S("mono", fontSize=8, leading=12, textColor=DARK, fontName="Courier")
    cap  = S("cap",  fontSize=8, leading=11, textColor=colors.HexColor("#64748b"), alignment=TA_CENTER)
    W, H = A4

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    def tbl(rows, col_widths, bg_header=GREY):
        t = Table(rows, colWidths=col_widths)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), bg_header),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
            ("BOX",        (0, 0), (-1, -1), 0.8, MID),
            ("INNERGRID",  (0, 0), (-1, -1), 0.5, MID),
            ("LEFTPADDING",  (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ]
        t.setStyle(TableStyle(style))
        return t

    def P(text, style=body): return Paragraph(text, style)
    def SP(h=0.3): return Spacer(1, h * cm)

    story = []

    # Title
    story.append(P("ClinicalSafe — External Validation Benchmark Report", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE))
    story.append(SP(0.2))
    story.append(P("Models tested: ML Engine  |  DL Base (Flan-T5)  |  DL Fine-Tuned (Flan-T5 + LoRA)", body))
    story.append(P("Data source: Pfizer Clinical Study Report (CSR) synopses — publicly available PDFs", body))
    story.append(SP(0.4))

    def fmt(v, pct=False, ms=False):
        if v is None: return "—"
        if pct: return f"{v*100:.1f}%"
        if ms:  return f"{v:.0f} ms"
        return f"{v:.4f}"

    for tier, models in all_aggs.items():
        tier_label = "Tier 1 — Gold Standard (with human reference)" \
                     if "tier1" in tier else "Tier 2 — Robustness / Stress Test"
        story.append(P(tier_label, h2))
        story.append(HRFlowable(width="100%", thickness=1, color=MID))
        story.append(SP(0.15))

        # Summary table
        headers = [
            P("<b>Metric</b>", body),
            P("<b>ML Engine</b>", body),
            P("<b>DL Base</b>", body),
            P("<b>DL Fine-Tuned</b>", body),
        ]
        def row(label, fn):
            cells = [P(label, body)]
            for mk in ["ml", "dl_base", "dl_ft"]:
                agg = models.get(mk, {})
                cells.append(P(fn(agg), body))
            return cells

        rows = [headers]
        rows.append(row("Tables tested",        lambda a: str(a.get("n_total", "—"))))
        rows.append(row("Success rate",          lambda a: fmt(a.get("success_rate"), pct=True)))
        rows.append(row("Verified rate",         lambda a: fmt(a.get("verified_rate"), pct=True)))
        rows.append(row("Hall. rate (mean)",     lambda a: fmt(a.get("hallucination_rate", {}).get("mean"), pct=True)))
        rows.append(row("Hall. rate (P90)",      lambda a: fmt(a.get("hallucination_rate", {}).get("p90"), pct=True)))
        rows.append(row("Speed — mean",          lambda a: fmt(a.get("inference_time_ms", {}).get("mean"), ms=True)))
        rows.append(row("Speed — P90",           lambda a: fmt(a.get("inference_time_ms", {}).get("p90"), ms=True)))

        if any("rouge1" in v for v in models.values()):
            rows.append(row("ROUGE-1 (mean)",    lambda a: fmt(a.get("rouge1", {}).get("mean"))))
            rows.append(row("ROUGE-1 (P10)",     lambda a: fmt(a.get("rouge1", {}).get("p10"))))
            rows.append(row("ROUGE-2 (mean)",    lambda a: fmt(a.get("rouge2", {}).get("mean"))))
            rows.append(row("ROUGE-L (mean)",    lambda a: fmt(a.get("rougeL", {}).get("mean"))))
            rows.append(row("Num Acc vs ref",    lambda a: fmt(a.get("numeric_accuracy_vs_ref", {}).get("mean"))))

        cw = [(W - 4*cm) * f for f in [0.35, 0.21, 0.21, 0.23]]
        story.append(tbl(rows, cw))
        story.append(SP(0.3))

        # Roadmap thresholds
        for mk, agg in models.items():
            thresh = agg.get("thresholds")
            if not thresh:
                continue
            label = MODEL_LABELS.get(mk, mk)
            story.append(P(f"<b>Roadmap Targets — {label}</b>", body))
            for k, v in thresh.items():
                mark = "✅" if v else "❌"
                story.append(P(f"  {mark}  {k.replace('_', ' ')}", body))
            story.append(SP(0.15))

        story.append(SP(0.3))

    # Sample outputs
    story.append(PageBreak())
    story.append(P("Sample Outputs — Tier 1 Examples", h2))
    story.append(HRFlowable(width="100%", thickness=1, color=MID))
    story.append(SP(0.2))

    for i, ex in enumerate(sample_rows.get("tier1", [])[:5]):
        story.append(P(f"<b>Example {i+1}</b>  ({ex.get('source_pdf', '')})", body))
        story.append(P("<b>Input Table:</b>", body))
        ttext = ex.get("table_text", "")[:500].replace("\n", " ")
        story.append(P(ttext + "...", mono))
        story.append(P("<b>Reference (human written):</b>", body))
        story.append(P((ex.get("reference") or "—")[:400], mono))
        for mk, label in [("ml", "ML Engine"), ("dl_base", "DL Base"), ("dl_ft", "DL Fine-Tuned")]:
            key = f"output_{mk}"
            out = ex.get(key, "—")
            if out:
                story.append(P(f"<b>{label}:</b>", body))
                story.append(P(str(out)[:400], mono))
        story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
        story.append(SP(0.2))

    # Footer
    story.append(SP(0.5))
    story.append(P("ClinicalSafe Benchmark Report — Generated March 2026", cap))

    doc.build(story)
    print(f"\n  ✅ PDF report saved: {OUT_PDF}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  BENCHMARK REPORT GENERATOR — Phase 3")
    print("=" * 65)

    file_map  = discover_result_files()
    if not file_map:
        print("\n❌ No result files found in data/benchmark/results/")
        print("   Run benchmark_run.py first.")
        return

    all_aggs   = {}
    sample_map = {}   # for PDF report

    for tier, model_files in sorted(file_map.items()):
        all_aggs[tier] = {}
        sample_rows    = []

        for model_key, fpath in sorted(model_files.items()):
            rows = load_jsonl(fpath)
            if not rows:
                continue
            all_aggs[tier][model_key] = aggregate(rows)

        # Build sample rows (join model outputs for first 10 examples)
        # Use ML file as the base
        base_file = model_files.get("ml") or list(model_files.values())[0]
        base_rows = load_jsonl(base_file)[:10]
        other_outputs = {}
        for mk, fp in model_files.items():
            other_outputs[mk] = {r["pair_id"]: r for r in load_jsonl(fp)[:10]}

        for row in base_rows:
            pid = row["pair_id"]
            sample = {
                "pair_id":    pid,
                "source_pdf": row.get("source_pdf", ""),
                "table_text": row.get("table_text", ""),
                "reference":  row.get("reference"),
            }
            for mk in model_files:
                out_rec = other_outputs.get(mk, {}).get(pid, {})
                sample[f"output_{mk}"] = out_rec.get("summary", "—")
            sample_rows.append(sample)
        sample_map[tier] = sample_rows

    # Terminal report
    print_report(all_aggs)

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(all_aggs, f, indent=2)
    print(f"  ✅ JSON report saved: {OUT_JSON}")

    # PDF report
    generate_pdf_report(all_aggs, sample_map)


if __name__ == "__main__":
    main()
