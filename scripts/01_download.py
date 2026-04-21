"""Download monthly judicial opendata archives.

Usage:
    python scripts/01_download.py --start 63694 --end 64055
    python scripts/01_download.py --start 63694 --end 63700 --out data/raw
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetcher import download_range, load_cookies_json, make_session  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True, help="Starting dataset ID (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Ending dataset ID (inclusive)")
    parser.add_argument("--out", type=Path, default=Path("data/raw"),
                        help="Output directory (default: data/raw)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds (default: 0.5)")
    parser.add_argument("--cookies", type=Path, default=Path(".cookies.json"),
                        help="Cookie file from prime_cookies.py (default: .cookies.json)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cookies = load_cookies_json(args.cookies)
    if cookies:
        logging.info("loaded %d cookies from %s", len(cookies), args.cookies)
        session = make_session(cookie_jar=cookies)
    else:
        logging.warning("no cookies at %s; attempting unauthenticated download "
                        "(likely to hit bot-defense 500)", args.cookies)
        session = make_session()

    count = 0
    for _ in download_range(args.start, args.end, args.out,
                            delay=args.delay, session=session):
        count += 1
    print(f"Downloaded / verified {count} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
