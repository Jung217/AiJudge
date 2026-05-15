"""Filter downloaded ZIPs/extracted-tree for northern-court drug cases.

Usage:
    python scripts/02_filter.py
    python scripts/02_filter.py --zip-dir data/extracted --out data/filtered/keelung_drug.jsonl
    python scripts/02_filter.py --courts northern --out data/filtered/north5_drug.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filter import (  # noqa: E402
    COURT_REGISTRY, KEELUNG_ONLY, NORTHERN_5_COURT_CODES, filter_drug_cases,
)
from records import iter_records_dir  # noqa: E402


_COURT_SETS = {
    "keelung": KEELUNG_ONLY,
    "northern": NORTHERN_5_COURT_CODES,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/filtered/keelung_drug.jsonl"))
    parser.add_argument("--courts", choices=sorted(_COURT_SETS),
                        default="keelung",
                        help="which court set to keep (default: keelung only)")
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

    courts = _COURT_SETS[args.courts]
    by_court: Counter[str] = Counter()
    count = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for r, code in filter_drug_cases(iter_records_dir(args.zip_dir), courts):
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
                "court": code,
                "court_name": COURT_REGISTRY[code][0],
            }, ensure_ascii=False) + "\n")
            count += 1
            by_court[code] += 1
            if count % 200 == 0:
                logging.info("filtered %d cases so far (%s)",
                              count, dict(by_court))

    print(f"Wrote {count} drug cases to {args.out}")
    for code in courts:
        n = by_court.get(code, 0)
        print(f"  {COURT_REGISTRY[code][0]:<4} ({code}): {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
