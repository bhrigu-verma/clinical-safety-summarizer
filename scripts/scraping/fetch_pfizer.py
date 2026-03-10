"""
scripts/scraping/fetch_pfizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automated scraper for Pfizer Clinical Trial Results using Playwright.

Navigates the Drupal Views AJAX table, paginates through all available
trials, identifies CSR Synopses (PDF files), and downloads them into
pfizer_csrs/ while keeping track of what's already downloaded.
"""

import sys
import time
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, unquote

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "pfizer_csrs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = OUT_DIR / "trial_index.json"

URL = "https://www.pfizer.com/science/clinical-trials/data-and-results/trial-results"

def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_index(idx):
    with open(INDEX_FILE, "w") as f:
        json.dump(idx, f, indent=2)

def download_file(url, out_path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response, open(out_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        print(f"      [Error downloading {url}]: {e}")
        return False

def extract_nct_id(url):
    """Attempt to find NCTID from filename"""
    filename = unquote(url.split("/")[-1].split("?")[0])
    import re
    m = re.search(r'(NCT\d{8})', filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Try alternate forms like B5061004 (Pfizer protocol IDs)
    m2 = re.search(r'([A-Z]\d{6,})', filename, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return filename[:15]

def main():
    trials_data = load_index()
    downloaded_urls = {link["url"] for t in trials_data for link in t.get("pdf_links", [])}
    
    print("=" * 65)
    print("  PFIZER CSR SCRAPER (Playwright)")
    print("=" * 65)
    print(f"  Already indexed: {len(trials_data)} trials")
    print(f"  Already downloaded: {len(downloaded_urls)} PDFs")
    print("  Starting browser...")

    found_new = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        print(f"  Navigating to {URL}...")
        page.goto(URL, wait_until="domcontentloaded", timeout=90000)
        time.sleep(5)  # Let cookies settle
        
        # Click the "Show Results" button which is required to load the trials table
        try:
            print("  Clicking 'Show Results' button...")
            page.evaluate("document.querySelector('.agree-button, #onetrust-accept-btn-handler')?.click()")
            time.sleep(1)
            page.evaluate("document.querySelector('form.views-exposed-form input[type=submit], form.views-exposed-form button').click()")
            time.sleep(5)
        except Exception as e:
            print(f"  Warning: Could not click Show Results button: {e}")

        # Wait for table rows to load (avoid timing out on just views-table)
        try:
            page.wait_for_selector("table, .views-row", timeout=30000)
            print("  Table loaded.")
        except Exception:
            print("❌  Could not find main table rows.")
            # Let's save a screenshot to understand why
            page.screenshot(path="debug_pfizer_table.png")
            return

        page_num = 1
        
        while True:
            print(f"\n  [Page {page_num}] Scanning...")
            
            # Wait for network to settle so table isn't refreshing while we scan
            time.sleep(3)

            # Find all rows in the results table
            rows = page.locator("table.views-table tbody tr").all()
            if not rows:
                print("  No rows found, ending.")
                break
                
            print(f"  Found {len(rows)} trials on this page.")
            page_pdfs = 0
            
            for row in rows:
                try:
                    # Look for CSR Synopsis links inside the row
                    links = row.locator("a").element_handles()
                    for link in links:
                        href = getattr(link, 'get_attribute')("href") if hasattr(link, 'get_attribute') else link.get_attribute("href")
                        if not href: continue
                        
                        # We specifically want CSR Synopses which are PDFs
                        if ".pdf" in href.lower() or "csr" in href.lower() or "synopsis" in href.lower() or "study report" in href.lower():
                            if href.startswith("/"):
                                href = "https://www.pfizer.com" + href
                                
                            if href in downloaded_urls:
                                continue
                                
                            nct_id = extract_nct_id(href)
                            print(f"    → Found new PDF for: {nct_id}")
                            
                            filename = unquote(href.split("/")[-1].split("?")[0])
                            if not filename.endswith(".pdf"):
                                filename += ".pdf"
                                
                            safe_filename = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)
                            out_path = OUT_DIR / f"{nct_id}__{safe_filename}"
                            
                            print(f"      Downloading to {out_path.name}...")
                            if download_file(href, out_path):
                                trials_data.append({
                                    "nct_id": nct_id,
                                    "phase": "Unknown (Scraped)",
                                    "date": "2026",
                                    "pdf_links": [{"url": href, "label": "PDF"}]
                                })
                                downloaded_urls.add(href)
                                save_index(trials_data)
                                found_new += 1
                                page_pdfs += 1
                                
                                if len(trials_data) >= 2000:
                                    print("\n  🎯 Reached 2000 table goal!")
                                    browser.close()
                                    return
                except Exception as e:
                    # Ignore stale element exceptions
                    pass

            # Look for the 'Next' pagination button
            next_btn = page.locator("li.pager__item--next a")
            if next_btn.count() > 0:
                print("  Clicking 'Next Page'...")
                next_btn.first.click()
                page_num += 1
                # Wait for the table loader element to appear and disappear, or just sleep
                time.sleep(2)
            else:
                print("  No 'Next' button found. Pagination complete.")
                break

        browser.close()

    print("\n" + "=" * 65)
    print("  SCRAPING COMPLETE")
    print(f"  New PDFs downloaded: {found_new}")
    print(f"  Total Index size:    {len(trials_data)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
