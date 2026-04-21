"""Incremental 7-day sync via JDoc REST API.

Pulls the last 7 days of Keelung drug-case JIDs from data.judicial.gov.tw
and saves passing records to data/filtered/keelung_drug.jsonl (append).

**Service window**: 00:00–06:00 Taiwan time.
**Credentials**: put JDOC_USER / JDOC_PASS in a `.env` file at repo root, or
export as env vars. `.env` is gitignored.

Usage:
    python scripts/04_jdoc_sync.py
    python scripts/04_jdoc_sync.py --dry-run       # list JIDs only
    python scripts/04_jdoc_sync.py --out custom.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

from filter import is_drug_case, is_first_instance_guilty, is_keelung, is_plea_bargain  # noqa: E402
from jdoc_client import JDocClient, is_keelung_jid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/filtered/keelung_drug.jsonl"))
    ap.add_argument("--dry-run", action="store_true",
                    help="list matched JIDs but don't fetch full documents")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = JDocClient()
    logger = logging.getLogger(__name__)
    try:
        client.authenticate()
    except Exception as e:
        msg = str(e)
        if "credentials missing" in msg:
            print("error: no credentials. Create a .env file at repo root:\n"
                  "  JDOC_USER=your_username\n"
                  "  JDOC_PASS=your_password",
                  file=sys.stderr)
        elif "服務時間" in msg or "服务时间" in msg:
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone(timedelta(hours=8)))
            print(f"error: JDoc service window is 00:00-06:00 Taiwan time.\n"
                  f"       current TW time: {now:%Y-%m-%d %H:%M}\n"
                  f"       retry after midnight.", file=sys.stderr)
        else:
            print(f"error: auth failed: {e}", file=sys.stderr)
        return 1
    logger.info("authenticated; fetching JList")

    jlist = client.list_recent_jids()
    total_jids = sum(len(d.get("list", [])) for d in jlist)
    logger.info("JList returned %d days, %d JIDs total", len(jlist), total_jids)

    candidates: list[str] = []
    for day in jlist:
        for jid in day.get("list", []):
            if is_keelung_jid(jid):
                candidates.append(jid)
    logger.info("keelung-prefix candidates: %d", len(candidates))

    # JList returns a rolling 7-day window, so consecutive runs overlap.
    # Dedupe by reading JIDs already saved before appending.
    seen_jids: set[str] = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen_jids.add(json.loads(line).get("jid", ""))
                except json.JSONDecodeError:
                    continue
    logger.info("existing output: %d JIDs (skipping overlap)", len(seen_jids))

    candidates = [j for j in candidates if j not in seen_jids]
    logger.info("new candidates after dedup: %d", len(candidates))

    if args.dry_run:
        for jid in candidates:
            print(jid)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for r in client.iter_documents(candidates, delay=args.delay):
            if not is_keelung(r):
                continue
            if not is_drug_case(r):
                continue
            if is_plea_bargain(r):
                continue
            if not is_first_instance_guilty(r):
                continue
            if r.jid in seen_jids:   # defense in depth
                skipped += 1
                continue
            seen_jids.add(r.jid)
            fh.write(json.dumps({
                "jid": r.jid, "jyear": r.jyear, "jcase": r.jcase, "jno": r.jno,
                "jdate": r.jdate, "jtitle": r.jtitle, "jfull": r.jfull,
                "jpdf": r.jpdf, "source_zip": r.source_zip,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"appended {written} new Keelung drug cases to {args.out} "
          f"(skipped {skipped} duplicates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
