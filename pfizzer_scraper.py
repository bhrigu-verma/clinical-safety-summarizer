"""
Pfizer CSR Downloader — uses the confirmed Drupal Views AJAX endpoint
=====================================================================
pip install requests tqdm beautifulsoup4
python pfizer_drupal_scraper.py
"""

import re
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

OUTPUT_DIR = Path("./pfizer_csrs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── The confirmed endpoint ────────────────────────────────────────────────
BASE_URL = "https://www.pfizer.com/views/ajax"

# Base params — clear all drug-specific filters, keep phase + date
BASE_PARAMS = {
    "_wrapper_format":           "drupal_ajax",
    "field_propriety_name_value": "",       # brand name — empty = all drugs
    "field_generic_name_value":   "",       # generic name — empty = all
    "field_phase_value":          "",       # set per-request (Phase III / IV)
    "field_nct_id_value":         "",
    "field_protocol_value":       "",
    "field_start_date_value":     "2015-01-01",   # last 10 years
    "field_start_date_value_1":   "",
    "field_completion_date_value": "",
    "field_completion_date_value_1": "",
    "items_per_page":             "48",     # max page size
    "view_name":                  "clinical_trial_results",
    "view_display_id":            "clinical_trial_results",
    "view_args":                  "",
    "view_path":                  "/node/452636",
    "view_base_path":             "admin/content/clinicat-trials/csv",
    "view_dom_id":                "df8e31810b9d123de1888d795cc205c069a774cbf76237f67a3936a3f826ad3a",
    "pager_element":              "0",
    "_drupal_ajax":               "1",
    "ajax_page_state[theme]":     "pfecpfizercomus",
    "ajax_page_state[theme_token]": "",
    "ajax_page_state[libraries]": "",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0",
    "Accept":     "application/json, text/javascript, */*; q=0.01",
    "Referer":    "https://www.pfizer.com/science/clinical-trials/data-and-results/trial-results",
    "X-Requested-With": "XMLHttpRequest",
}

PHASES = ["PHASE III", "PHASE IV"]


def extract_html_from_drupal_response(data) -> str:
    """Safely extract HTML from Drupal AJAX response regardless of format.

    Pfizer's Drupal AJAX endpoint returns a dict with numeric string keys:
      {"0": {"command": "settings", ...},
       "1": {"command": "add_css", ...},
       "4": {"command": "insert", "data": "<html...>"},
       ...}
    """
    html_parts = []

    # Normalise: convert {"0": ..., "1": ...} dict into a flat list of commands
    if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
        items = [data[k] for k in sorted(data.keys(), key=int)]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    for item in items:
        if isinstance(item, str):
            if item.strip().startswith("<"):
                html_parts.append(item)
            continue
        if isinstance(item, dict):
            cmd = item.get("command", "")
            if cmd in ("insert", "replace") and item.get("data"):
                html_parts.append(item["data"])

    return "\n".join(html_parts)




def fetch_page(phase: str, page: int) -> list[dict]:
    """Fetch one page of results for a given phase."""
    params = {**BASE_PARAMS, "field_phase_value": phase, "page": page}

    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Drupal AJAX may return JSON array of commands, or raw HTML
    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        try:
            data = resp.json()
        except Exception:
            # Malformed JSON — treat as raw HTML
            data = resp.text
    else:
        data = resp.text

    html_content = extract_html_from_drupal_response(data)

    if not html_content.strip():
        # Debug: print first 500 chars of raw response if we got nothing
        print(f"  ⚠️  No HTML extracted (page={page}). Raw start: {resp.text[:300]}")

    return parse_trials_from_html(html_content, phase)


def parse_trials_from_html(html: str, phase: str) -> list[dict]:
    """Extract trial records and PDF links from the HTML response."""
    soup = BeautifulSoup(html, "html.parser")
    trials = []

    # Each trial is typically a row or card — inspect the actual HTML and
    # adjust these selectors if needed
    rows = soup.select("tr, .views-row, .trial-result-row, article")

    for row in rows:
        trial = {"phase": phase}

        # Title / drug name
        title_el = row.select_one("td.views-field-title a, .field-title a, h3 a, h2 a")
        if title_el:
            trial["title"] = title_el.get_text(strip=True)
            trial["detail_url"] = "https://www.pfizer.com" + title_el.get("href", "")

        # NCT ID
        nct_el = row.select_one(".field-nct-id, td.views-field-field-nct-id")
        if nct_el:
            trial["nct_id"] = nct_el.get_text(strip=True)

        # Phase
        phase_el = row.select_one(".field-phase, td.views-field-field-phase-value")
        if phase_el:
            trial["phase"] = phase_el.get_text(strip=True)

        # Date
        date_el = row.select_one(".field-date, td.views-field-field-completion-date")
        if date_el:
            trial["date"] = date_el.get_text(strip=True)

        # PDF links — look for any .pdf href in the row
        pdf_links = []
        for a in row.select("a[href$='.pdf'], a[href*='cdn.pfizer.com']"):
            href = a.get("href", "")
            if href:
                pdf_links.append({
                    "url":   href if href.startswith("http") else "https://www.pfizer.com" + href,
                    "label": a.get_text(strip=True) or "PDF",
                })
        trial["pdf_links"] = pdf_links

        # Only add if we got something meaningful
        if trial.get("title") or trial.get("pdf_links"):
            trials.append(trial)

    return trials


def get_total_pages(phase: str) -> int:
    """Get the total number of pages by checking the pager on page 0."""
    params = {**BASE_PARAMS, "field_phase_value": phase, "page": 0}
    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        try:
            data = resp.json()
        except Exception:
            data = resp.text
    else:
        data = resp.text

    html = extract_html_from_drupal_response(data)
    soup = BeautifulSoup(html, "html.parser")

    # Drupal pager: last page link or total count text
    last_page = soup.select_one("li.pager__item--last a, .pager-last a")
    if last_page:
        href = last_page.get("href", "")
        match = re.search(r"[?&]page=(\d+)", href)
        if match:
            return int(match.group(1)) + 1

    # Fallback: count page links
    page_links = soup.select("li.pager__item a, .pager-item a")
    if page_links:
        nums = []
        for a in page_links:
            href = a.get("href", "")
            m = re.search(r"page=(\d+)", href)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            return max(nums) + 1

    return 99  # safe fallback — loop will stop when results are empty


def download_pdf(session: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    try:
        r = session.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ✗ {dest.name}: {e}")
        return False


def main():
    print("Pfizer CSR Downloader — Phase III/IV (2015–2026)")
    print("Using endpoint: /views/ajax (Drupal)")
    print("=" * 52)

    all_trials = []

    # ── Collect all trial metadata ────────────────────────
    for phase in PHASES:
        print(f"\nPhase: {phase}")
        total_pages = get_total_pages(phase)
        print(f"  Estimated pages: {total_pages}")

        for page in tqdm(range(total_pages), desc=f"  Fetching {phase}"):
            trials = fetch_page(phase, page)
            if not trials:
                print(f"  No results at page {page}, stopping.")
                break
            all_trials.extend(trials)
            time.sleep(0.5)  # polite rate limit

    print(f"\nTotal trials found: {len(all_trials)}")

    # ── Save full index ───────────────────────────────────
    index_path = OUTPUT_DIR / "trial_index.json"
    index_path.write_text(json.dumps(all_trials, indent=2))
    print(f"Index saved → {index_path}")

    # ── Download all PDFs ─────────────────────────────────
    all_pdfs = [
        (doc["url"], trial.get("title", "untitled"), trial.get("nct_id", ""))
        for trial in all_trials
        for doc in trial.get("pdf_links", [])
    ]

    print(f"\n{len(all_pdfs)} PDFs to download...")

    session = requests.Session()
    downloaded = 0

    for url, title, nct_id in tqdm(all_pdfs, desc="Downloading PDFs"):
        safe_name = re.sub(r"[^\w\-]", "_", f"{nct_id}__{title}")[:100]
        fname     = f"{safe_name}.pdf" if not safe_name.endswith(".pdf") else safe_name
        dest      = OUTPUT_DIR / fname

        if download_pdf(session, url, dest):
            downloaded += 1
        time.sleep(0.3)

    print(f"\n✓ Done. {downloaded}/{len(all_pdfs)} PDFs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()