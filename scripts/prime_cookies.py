"""Obtain a bot-defense cookie for opendata.judicial.gov.tw via Playwright.

The site is protected by F5 BIG-IP ASM + Cloudflare Turnstile. This script
launches Chromium, loads the homepage + a dataset page so the JS challenges
resolve, then saves cookies to `.cookies.json` for the requests-based fetcher.

Some sites detect headless — if --headless fails, retry without the flag to
see a visible browser window.

Usage:
    python scripts/prime_cookies.py                 # visible browser (default)
    python scripts/prime_cookies.py --headless
    python scripts/prime_cookies.py --verify 63694  # also test one download
    python scripts/prime_cookies.py --dataset 51427 # visit specific dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOME = "https://opendata.judicial.gov.tw/"
DATASET_TMPL = "https://opendata.judicial.gov.tw/dataset/{id}"
DL_TMPL = "https://opendata.judicial.gov.tw/api/FilesetLists/{id}/file"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("error: install playwright first — pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(".cookies.json"))
    ap.add_argument("--headless", action="store_true",
                    help="run browser headless (may fail Turnstile)")
    ap.add_argument("--dataset", type=int, default=51427,
                    help="datasetId to visit for challenge resolution (default: 51427 = 199601)")
    ap.add_argument("--verify", type=int, default=None,
                    help="after priming, try downloading this fileSetId and report status")
    ap.add_argument("--settle", type=int, default=10,
                    help="seconds to wait after each page load (default: 10)")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            user_agent=UA,
            locale="zh-TW",
            viewport={"width": 1280, "height": 800},
        )
        # Remove webdriver fingerprint
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        page = ctx.new_page()
        print(f"[1/3] loading {HOME}")
        page.goto(HOME, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(args.settle * 1000)

        dataset_url = DATASET_TMPL.format(id=args.dataset)
        print(f"[2/3] loading {dataset_url}")
        try:
            page.goto(dataset_url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(args.settle * 1000)
        except Exception as e:
            print(f"  warn: dataset page load: {e}")

        cookies = ctx.cookies()
        args.out.write_text(
            json.dumps(cookies, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[3/3] saved {len(cookies)} cookies to {args.out}")
        for c in cookies:
            print(f"  {c['name']} = {c['value'][:50]}...")

        if args.verify is not None:
            dl_url = DL_TMPL.format(id=args.verify)
            print(f"\nverifying download: {dl_url}")
            resp = ctx.request.get(
                dl_url,
                headers={"Referer": dataset_url, "User-Agent": UA},
                timeout=120000,
            )
            ct = resp.headers.get("content-type", "?")
            print(f"  status={resp.status}  content-type={ct}")
            if resp.status == 200 and "json" not in ct:
                body = resp.body()
                print(f"  OK got {len(body)} bytes (RAR archive)")
                out_rar = Path("data/raw") / f"{args.verify}.rar"
                out_rar.parent.mkdir(parents=True, exist_ok=True)
                out_rar.write_bytes(body)
                print(f"  saved to {out_rar}")
            else:
                txt = resp.text()[:400]
                print(f"  FAIL: {txt}")

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
