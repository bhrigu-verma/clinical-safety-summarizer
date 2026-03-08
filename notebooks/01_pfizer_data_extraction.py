#!/usr/bin/env python3
"""
01_pfizer_data_extraction.py — Pfizer CSR Safety Table Extractor
================================================================
Extracts (table_text, narrative) pairs from Pfizer CSR PDFs.
Target: 10,000 pairs from all available Pfizer Phase III/IV trials.

Run on any machine (no GPU needed). Output: data/pfizer_gold_pairs.jsonl

Usage:
  pip install requests tqdm beautifulsoup4 pdfplumber camelot-py
  python notebooks/01_pfizer_data_extraction.py
"""

import re
import json
import time
import os
import sys
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "pfizer_csrs"
PAIRS_OUTPUT = ROOT / "data" / "pfizer_gold_pairs.jsonl"
OUTPUT_DIR.mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Download ALL Pfizer CSR PDFs (Phase I–IV, all dates)
# ═══════════════════════════════════════════════════════════════════════

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

BASE_URL = "https://www.pfizer.com/views/ajax"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.pfizer.com/science/clinical-trials/data-and-results/trial-results",
    "X-Requested-With": "XMLHttpRequest",
}

BASE_PARAMS = {
    "_wrapper_format": "drupal_ajax",
    "field_propriety_name_value": "",
    "field_generic_name_value": "",
    "field_phase_value": "",
    "field_nct_id_value": "",
    "field_protocol_value": "",
    "field_start_date_value": "",       # ALL dates (no filter)
    "field_start_date_value_1": "",
    "field_completion_date_value": "",
    "field_completion_date_value_1": "",
    "items_per_page": "48",
    "view_name": "clinical_trial_results",
    "view_display_id": "clinical_trial_results",
    "view_args": "",
    "view_path": "/node/452636",
    "view_base_path": "admin/content/clinicat-trials/csv",
    "view_dom_id": "df8e31810b9d123de1888d795cc205c069a774cbf76237f67a3936a3f826ad3a",
    "pager_element": "0",
    "_drupal_ajax": "1",
    "ajax_page_state[theme]": "pfecpfizercomus",
    "ajax_page_state[theme_token]": "",
    "ajax_page_state[libraries]": "",
}

ALL_PHASES = ["PHASE I", "PHASE II", "PHASE III", "PHASE IV"]


def extract_html(data) -> str:
    """Extract HTML from Drupal AJAX response (handles dict/list/str)."""
    parts = []
    if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
        items = [data[k] for k in sorted(data.keys(), key=int)]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    for item in items:
        if isinstance(item, dict) and item.get("command") in ("insert", "replace") and item.get("data"):
            parts.append(item["data"])
    return "\n".join(parts)


def fetch_all_trials() -> List[Dict]:
    """Fetch ALL trial metadata across all phases."""
    all_trials = []
    session = requests.Session()

    for phase in ALL_PHASES:
        logger.info(f"Fetching {phase}...")
        for page in range(200):  # max 200 pages
            params = {**BASE_PARAMS, "field_phase_value": phase, "page": page}
            try:
                resp = session.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                ct = resp.headers.get("Content-Type", "")
                data = resp.json() if "json" in ct else resp.text
                html = extract_html(data)
                if not html.strip():
                    break

                soup = BeautifulSoup(html, "html.parser")
                rows = soup.select("tr, .views-row, article")
                found = 0
                for row in rows:
                    trial = {"phase": phase}
                    # NCT ID
                    for el in row.select("td, span, div"):
                        txt = el.get_text(strip=True)
                        m = re.search(r"NCT\d{6,}", txt)
                        if m:
                            trial["nct_id"] = m.group(0)
                            break
                    # PDF links
                    pdfs = []
                    for a in row.select("a[href$='.pdf'], a[href*='cdn.pfizer.com']"):
                        href = a.get("href", "")
                        if href:
                            url = href if href.startswith("http") else "https://www.pfizer.com" + href
                            pdfs.append(url)
                    trial["pdf_urls"] = list(set(pdfs))
                    if trial.get("nct_id") or pdfs:
                        all_trials.append(trial)
                        found += 1

                if found == 0:
                    break
                time.sleep(0.4)
            except Exception as e:
                logger.warning(f"  Page {page} failed: {e}")
                break

        logger.info(f"  {phase}: cumulative {len(all_trials)} trials")

    # Deduplicate by NCT ID
    seen = set()
    unique = []
    for t in all_trials:
        key = t.get("nct_id", hashlib.md5(str(t["pdf_urls"]).encode()).hexdigest())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    logger.info(f"Total unique trials: {len(unique)}")
    return unique


def download_pdfs(trials: List[Dict]) -> int:
    """Download all CSR PDFs."""
    session = requests.Session()
    all_urls = [(t.get("nct_id", "unknown"), url)
                for t in trials for url in t.get("pdf_urls", [])]
    logger.info(f"Downloading {len(all_urls)} PDFs...")

    downloaded = 0
    for nct_id, url in tqdm(all_urls, desc="PDFs"):
        fname = f"{nct_id}__{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
        dest = OUTPUT_DIR / fname
        if dest.exists() and dest.stat().st_size > 1024:
            downloaded += 1
            continue
        try:
            r = session.get(url, headers=HEADERS, timeout=60, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            downloaded += 1
            time.sleep(0.3)
        except Exception as e:
            logger.debug(f"  Failed {fname}: {e}")

    logger.info(f"Downloaded: {downloaded}/{len(all_urls)}")
    return downloaded


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Extract Safety Tables + Narrative from each PDF
# ═══════════════════════════════════════════════════════════════════════

TABLE_PATTERNS = [
    re.compile(r"(?:overall\s+)?summary\s+of\s+(?:treatment[- ]emergent\s+)?adverse\s+events",
               re.IGNORECASE),
    re.compile(r"overview\s+of\s+(?:treatment[- ]emergent\s+)?adverse\s+events", re.IGNORECASE),
    re.compile(r"table\s+\d+[\.:]\s*(?:treatment[- ]emergent|teae|adverse\s+event)",
               re.IGNORECASE),
    re.compile(r"summary\s+of\s+(?:teae|sae|serious\s+adverse)", re.IGNORECASE),
]

NARRATIVE_PATTERNS = [
    re.compile(r"(?:the\s+)?(?:overall\s+)?incidence\s+of\s+(?:treatment[- ]emergent|teae)",
               re.IGNORECASE),
    re.compile(r"treatment[- ]emergent\s+adverse\s+events?\s+(?:were|was)\s+reported",
               re.IGNORECASE),
    re.compile(r"(?:teae|adverse\s+event)s?\s+(?:were|was)\s+(?:observed|reported|experienced)",
               re.IGNORECASE),
]


def extract_pairs_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract (safety_table, narrative) pairs from a single CSR PDF."""
    pairs = []
    try:
        import pdfplumber
    except ImportError:
        logger.error("pip install pdfplumber")
        return pairs

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        logger.debug(f"Cannot open {pdf_path}: {e}")
        return pairs

    all_text = []
    all_tables = []

    for i, page in enumerate(pdf.pages):
        try:
            text = page.extract_text() or ""
            all_text.append((i, text))

            tables = page.extract_tables()
            for t_idx, table in enumerate(tables or []):
                if not table or len(table) < 3:
                    continue
                # Linearize the table
                linearized = linearize_table(table, i, t_idx)
                if linearized and len(linearized) > 100:
                    all_tables.append((i, linearized))
        except Exception:
            continue

    pdf.close()

    # Combine all page text
    full_text = "\n".join(t for _, t in all_text)

    # Find safety-related tables
    for page_num, table_text in all_tables:
        is_safety = any(pat.search(table_text) for pat in TABLE_PATTERNS)
        if not is_safety:
            continue

        # Look for narrative near this table (within 2 pages after)
        narrative = find_narrative_near(all_text, page_num)
        if narrative and len(narrative) > 80:
            pairs.append({
                "table_text": table_text.strip(),
                "writeup": narrative.strip(),
                "source": Path(pdf_path).name,
                "page": page_num,
            })

    return pairs


def linearize_table(table: List[List], page: int, idx: int) -> str:
    """Convert a pdfplumber table to linearized text format."""
    if not table or not table[0]:
        return ""

    lines = []
    headers = [str(c or "").strip() for c in table[0]]
    lines.append(f"start_table [TABLE_TITLE: Table from page {page+1}]")
    lines.append(f"[HEADERS: | {'| '.join(h for h in headers if h)}]")

    for row in table[1:]:
        cells = [str(c or "").strip() for c in row]
        if any(c for c in cells):
            lines.append(f"[ROW] {' | '.join(cells)}")

    lines.append("end_table")
    return " ".join(lines)


def find_narrative_near(all_text: List[Tuple[int, str]], table_page: int) -> Optional[str]:
    """Find the safety narrative text near a table's page."""
    search_range = range(max(0, table_page - 1), min(len(all_text), table_page + 5))

    for idx in search_range:
        page_num, text = all_text[idx]
        for pat in NARRATIVE_PATTERNS:
            match = pat.search(text)
            if match:
                # Extract paragraph starting from the match
                start = max(0, match.start() - 10)
                # Find end of paragraph (double newline or end of meaningful text)
                remaining = text[start:]
                # Take up to 2000 chars of narrative
                paragraphs = re.split(r"\n\s*\n", remaining)
                narrative = " ".join(paragraphs[:3]).strip()
                if len(narrative) > 80:
                    return narrative[:2000]
    return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Quality Audit (per research report recommendation)
# ═══════════════════════════════════════════════════════════════════════

def audit_pair_quality(pair: Dict) -> Dict:
    """Audit a single pair for OCR noise and alignment quality."""
    table = pair["table_text"]
    narrative = pair["writeup"]

    # Extract numbers from both
    table_nums = set(re.findall(r"\b\d+\.?\d*\b", table))
    narr_nums = set(re.findall(r"\b\d+\.?\d*\b", narrative))

    # Overlap ratio = how many narrative numbers appear in the table
    if narr_nums:
        overlap = len(narr_nums & table_nums) / len(narr_nums)
    else:
        overlap = 0.0

    # OCR noise detection (unusual characters, broken words)
    ocr_noise = len(re.findall(r"[^\x20-\x7E\n\t]", table + narrative))
    char_count = len(table + narrative)
    noise_ratio = ocr_noise / max(char_count, 1)

    pair["quality"] = {
        "numeric_overlap": round(overlap, 3),
        "ocr_noise_ratio": round(noise_ratio, 4),
        "table_len": len(table),
        "narrative_len": len(narrative),
        "passes": overlap >= 0.3 and noise_ratio < 0.02 and len(narrative) > 80,
    }
    return pair


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PFIZER CSR DATA EXTRACTION PIPELINE")
    print("  Target: 10,000 (table, narrative) pairs")
    print("=" * 70)

    # Step 1: Fetch trial index
    index_path = OUTPUT_DIR / "full_trial_index.json"
    if index_path.exists():
        with open(index_path) as f:
            trials = json.load(f)
        logger.info(f"Loaded existing index: {len(trials)} trials")
    else:
        trials = fetch_all_trials()
        index_path.write_text(json.dumps(trials, indent=2))

    # Step 2: Download PDFs
    download_pdfs(trials)

    # Step 3: Extract pairs from all PDFs
    pdf_files = sorted(OUTPUT_DIR.glob("*.pdf"))
    logger.info(f"Processing {len(pdf_files)} PDFs...")

    all_pairs = []
    for pdf_path in tqdm(pdf_files, desc="Extracting tables"):
        pairs = extract_pairs_from_pdf(str(pdf_path))
        all_pairs.extend(pairs)

    logger.info(f"Raw pairs extracted: {len(all_pairs)}")

    # Step 4: Quality audit
    audited = [audit_pair_quality(p) for p in all_pairs]
    passed = [p for p in audited if p["quality"]["passes"]]
    logger.info(f"Quality audit: {len(passed)}/{len(audited)} pairs passed")

    # Step 5: Deduplicate
    seen_hashes = set()
    unique_pairs = []
    for p in passed:
        h = hashlib.md5((p["table_text"][:200] + p["writeup"][:200]).encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_pairs.append(p)

    logger.info(f"Unique high-quality pairs: {len(unique_pairs)}")

    # Step 6: Save
    with open(PAIRS_OUTPUT, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps({
                "table_text": p["table_text"],
                "writeup": p["writeup"],
                "source": p.get("source", ""),
            }) + "\n")

    logger.info(f"✅ Saved {len(unique_pairs)} pairs → {PAIRS_OUTPUT}")
    print(f"\n{'=' * 70}")
    print(f"  EXTRACTION COMPLETE: {len(unique_pairs)} gold pairs")
    print(f"  Output: {PAIRS_OUTPUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
