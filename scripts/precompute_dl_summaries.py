"""
Precompute DL summaries for benchmark JSONL files.

This script is intended for GPU machines where `main_dl` is available,
so CPU-only evaluators can run metrics offline using pre-filled summary keys.

Usage:
  python scripts/precompute_dl_summaries.py \
      --input data/benchmark/tier1_gold.jsonl \
      --output data/benchmark/tier1_gold_with_summaries.jsonl \
      --dl-url http://localhost:8001 \
      --modes base_dl finetuned
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute DL summaries into benchmark JSONL records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input benchmark JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL with summary_* keys")
    parser.add_argument("--dl-url", default="http://localhost:8001", help="DL backend base URL")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["base_dl", "finetuned"],
        choices=["base_dl", "finetuned"],
        help="DL modes to precompute",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between requests in milliseconds")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty summary_* fields",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Optional max number of records to process",
    )
    return parser.parse_args()


def _summary_key_for_mode(mode: str) -> str:
    if mode == "base_dl":
        return "summary_dl_base"
    if mode == "finetuned":
        return "summary_finetuned"
    raise ValueError(f"Unsupported mode: {mode}")


def _call_summarize(dl_url: str, table_text: str, mode: str, timeout: int) -> str:
    endpoint = dl_url.rstrip("/") + "/summarize"
    payload = {"table_text": table_text, "mode": mode}
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("summary") or data.get("text") or ""


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {exc}") from exc


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = list(_iter_jsonl(input_path))
    if args.n_max is not None:
        records = records[: args.n_max]

    total = len(records)
    print(f"Loaded {total} records from {input_path}")

    processed = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out:
        for idx, record in enumerate(records, start=1):
            table_text = record.get("table_text", "")
            table_id = record.get("table_id", f"line_{idx}")

            if not table_text:
                print(f"[{idx}/{total}] SKIP {table_id}: missing table_text")
                failed += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            for mode in args.modes:
                key = _summary_key_for_mode(mode)
                existing = str(record.get(key, "")).strip()

                if existing and not args.overwrite:
                    continue

                try:
                    summary = _call_summarize(args.dl_url, table_text, mode, args.timeout)
                    record[key] = summary
                except Exception as exc:
                    # Keep processing other records/modes while preserving traceability.
                    err_key = f"error_{key}"
                    record[err_key] = str(exc)
                    failed += 1

                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            if idx % 10 == 0 or idx == total:
                print(f"[{idx}/{total}] processed")

    print("Done")
    print(f"Processed: {processed}")
    print(f"Failed calls: {failed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
