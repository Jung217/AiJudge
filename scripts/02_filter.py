"""Filter downloaded ZIPs for Keelung drug cases; output as JSONL.

Usage:
    python scripts/02_filter.py
    python scripts/02_filter.py --zip-dir data/raw --out data/filtered/keelung_drug.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filter import keelung_drug_cases  # noqa: E402
from records import iter_records_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/filtered/keelung_drug.jsonl"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.zip_dir.exists():
        print(f"error: zip dir {args.zip_dir} does not exist", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for r in keelung_drug_cases(iter_records_dir(args.zip_dir)):
            fh.write(json.dumps({
                "jid": r.jid,
                "jyear": r.jyear,
                "jcase": r.jcase,
                "jno": r.jno,
                "jdate": r.jdate,
                "jtitle": r.jtitle,
                "jfull": r.jfull,
                "jpdf": r.jpdf,
                "source_zip": r.source_zip,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count % 100 == 0:
                logging.info("filtered %d cases so far", count)

    print(f"Wrote {count} Keelung drug cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
