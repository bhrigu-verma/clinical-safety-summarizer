"""
scripts/benchmark_extract.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 of the Benchmark Pipeline.

Reads all 147 Pfizer CSR PDFs already downloaded to pfizer_csrs/,
extracts every safety-related table and its accompanying narrative
(if present), deduplicates against the training set, and writes:

  data/benchmark/tier1_gold.jsonl   — table + human reference (for ROUGE)
  data/benchmark/tier2_stress.jsonl — table only (for robustness testing)

Usage:
    ./venv/bin/python3 scripts/benchmark_extract.py
"""

import re
import json
import hashlib
import sys
from pathlib import Path
from typing import Optional

import pdfplumber
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
PDF_DIR     = ROOT / "pfizer_csrs"
TRAIN_FILE  = ROOT / "data" / "processed" / "raw_pairs.json"
OUT_GOLD    = ROOT / "data" / "benchmark" / "tier1_gold.jsonl"
OUT_STRESS  = ROOT / "data" / "benchmark" / "tier2_stress.jsonl"

# ── Safety table detection patterns ───────────────────────────────────────────
SAFETY_HEADER_PATTERNS = [
    r"treatment[- ]emergent adverse",
    r"\bteae\b",
    r"overall summary.*adverse",
    r"adverse event.*summary",
    r"safety.*summary",
    r"incidence.*adverse",
    r"grade\s*3[-–]?4",
    r"\bsae\b",
    r"serious adverse",
    r"discontinuation.*teae",
    r"drug[- ]related.*adverse",
]
SAFETY_RE = re.compile("|".join(SAFETY_HEADER_PATTERNS), re.IGNORECASE)

NARRATIVE_PATTERNS = [
    r"overall[,\s]+teae.*were reported",
    r"treatment[- ]emergent adverse.*were reported",
    r"subjects.*experienced.*teae",
    r"incidence of.*adverse events",
    r"no (clinically significant|serious|grade 3)",
    r"\d+\s*\(\d+\.\d+%\).*subjects.*arm",
]
NARRATIVE_RE = re.compile("|".join(NARRATIVE_PATTERNS), re.IGNORECASE)


def table_fingerprint(table_text: str) -> str:
    nums = sorted(re.findall(r"\d+\.?\d*", table_text))
    return hashlib.md5(("".join(nums)).encode()).hexdigest()


def linearize_table(pdfplumber_table, title: str = "") -> Optional[str]:
    """Convert a pdfplumber table object to our linearized format."""
    if not pdfplumber_table or len(pdfplumber_table) < 2:
        return None

    rows = [r for r in pdfplumber_table if any(c for c in r if c)]
    if len(rows) < 2:
        return None

    # First row = headers
    headers = [str(c or "").strip() for c in rows[0]]
    num_cols = len(headers)

    # Check it has numbers (is a data table, not a text block)
    all_text = " ".join(str(c or "") for r in rows for c in r)
    if not re.search(r"\d+\.\d+%|\(\d+\.\d+\%\)|\d{2,}\s*\(", all_text):
        return None

    # Must look like a safety table
    if not SAFETY_RE.search(all_text):
        return None

    title_token = f"[TABLE_TITLE: {title}] " if title else ""
    header_str = " | ".join(headers)
    parts = [f"start_table {title_token}[HEADERS: {header_str}]"]

    for row in rows[1:]:
        cells = [str(c or "").strip() for c in row]
        if not any(c for c in cells):
            continue
        parts.append("[ROW] " + " | ".join(cells))

    parts.append("end_table")
    return "\n".join(parts)


def extract_narrative_near_table(page_text: str, table_bbox, page) -> Optional[str]:
    """
    Try to find a human-written narrative paragraph near a table.
    We look for text blocks above or below the table bounding box
    that match our NARRATIVE_RE pattern.
    """
    if not page_text:
        return None

    # Split into paragraphs
    paragraphs = re.split(r"\n{2,}", page_text)
    for para in paragraphs:
        para = para.strip()
        if len(para) < 80:
            continue
        if NARRATIVE_RE.search(para):
            # Must contain numbers to be a statistics narrative
            if re.search(r"\d+\.\d+%|\d+\s*\(\d+", para):
                return re.sub(r"\s+", " ", para).strip()
    return None


def process_pdf(pdf_path: Path, seen_fingerprints: set) -> dict:
    """
    Returns:
      {
        "tables_found": int,
        "pairs": [{"table_text":..., "writeup":... or None, "pair_id":...}],
        "errors": [str]
      }
    """
    result = {"tables_found": 0, "pairs": [], "errors": []}
    stem = pdf_path.stem

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    tables = page.extract_tables()
                    if not tables:
                        continue

                    # Try to find a table title from text above the table
                    for t_idx, raw_table in enumerate(tables):
                        # Attempt to find title in surrounding text
                        title_match = re.search(
                            r"(Table\s+\d+[\.\d]*[:\s][^\n]{5,60})", page_text
                        )
                        title = title_match.group(1).strip() if title_match else f"p{page_num+1}_t{t_idx+1}"

                        lin = linearize_table(raw_table, title)
                        if not lin:
                            continue

                        fp = table_fingerprint(lin)
                        if fp in seen_fingerprints:
                            continue
                        seen_fingerprints.add(fp)

                        result["tables_found"] += 1

                        # Try to find a narrative for this table
                        narrative = extract_narrative_near_table(page_text, None, page)

                        # Also check the next page
                        if narrative is None and page_num + 1 < len(pdf.pages):
                            next_text = pdf.pages[page_num + 1].extract_text() or ""
                            narrative = extract_narrative_near_table(next_text, None, None)

                        pair_id = f"{stem}_p{page_num+1}_t{t_idx+1}"
                        result["pairs"].append({
                            "pair_id":    pair_id,
                            "source_pdf": pdf_path.name,
                            "table_text": lin,
                            "writeup":    narrative,
                            "page":       page_num + 1,
                        })

                except Exception as e:
                    result["errors"].append(f"page {page_num+1}: {e}")

    except Exception as e:
        result["errors"].append(f"PDF open failed: {e}")

    return result


def load_train_fingerprints() -> set:
    """Load fingerprints of tables already in training set (must exclude from test)."""
    fps = set()
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE) as f:
            pairs = json.load(f)
        for p in pairs:
            fps.add(table_fingerprint(p.get("table_text", "")))
        print(f"  Loaded {len(fps)} training fingerprints to exclude")
    return fps


def main():
    print("=" * 65)
    print("  BENCHMARK EXTRACTION — Phase 1")
    print("=" * 65)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"\n📂 Found {len(pdf_files)} PDFs in pfizer_csrs/")

    seen_fps = load_train_fingerprints()

    gold_pairs   = []   # Tier 1: has writeup
    stress_pairs = []   # Tier 2: table only

    total_tables = 0
    total_errors = 0

    for pdf_path in tqdm(pdf_files, desc="Extracting tables", unit="pdf"):
        res = process_pdf(pdf_path, seen_fps)
        total_tables += res["tables_found"]
        total_errors += len(res["errors"])

        for pair in res["pairs"]:
            if pair["writeup"]:
                gold_pairs.append(pair)
            else:
                stress_pairs.append(pair)

    # Write Tier 1
    with open(OUT_GOLD, "w") as f:
        for p in gold_pairs:
            f.write(json.dumps(p) + "\n")

    # Write Tier 2
    with open(OUT_STRESS, "w") as f:
        for p in stress_pairs:
            f.write(json.dumps(p) + "\n")

    print("\n" + "=" * 65)
    print("  EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  PDFs processed:          {len(pdf_files)}")
    print(f"  Total tables extracted:  {total_tables}")
    print(f"  Extraction errors:       {total_errors}")
    print(f"\n  ─── Tier 1 (Gold — has human reference) ─────────────────")
    print(f"  Pairs:                   {len(gold_pairs)}")
    print(f"  Saved to:                {OUT_GOLD}")
    print(f"\n  ─── Tier 2 (Stress — table only) ────────────────────────")
    print(f"  Tables:                  {len(stress_pairs)}")
    print(f"  Saved to:                {OUT_STRESS}")
    print("=" * 65)

    return len(gold_pairs), len(stress_pairs)


if __name__ == "__main__":
    main()
