"""Helper for §57 batch extraction.

Loads cases from `data/filtered/keelung_drug_all.jsonl` for a given index range,
extracts the reason section for each, and dumps them to a JSONL file with fields
{idx, jid, jcase, reason} for downstream LLM processing.

Usage:
    python scripts/art57_batch_helper.py --start 960 --end 1279 --out /tmp/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make features importable when run from anywhere
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import _extract_section


MIN_LEN = 50


def get_reason(jfull: str) -> str:
    """Best-effort extraction of the reasoning section.

    Tries compound `事實及理由` header first (簡易判決 form), then falls back
    to a bare `理由` header (通常判決). If neither yields enough text, returns
    the full document — caller can still locate §57 reasoning text in 主文+事實
    that style cases sometimes contain.

    End markers are deliberately broad — `附表` was too aggressive (early cuts
    off at any 附表編號 occurrence inside reasoning prose); use `中華民國\d{3}`
    (the dating line in the back matter) or `書記官` (signature block) and
    fall through if neither hits.
    """
    reason = _extract_section(
        jfull,
        "事實及理由",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官"),
    )
    if len(reason) < MIN_LEN:
        reason = _extract_section(
            jfull,
            "理由",
            (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官", r"據\s*上\s*論\s*斷"),
        )
    if len(reason) < MIN_LEN:
        # Fall back to full text — some 簡易判決 stuff §57 reasoning into the
        # 犯罪事實 / 論罪科刑 sub-sections.
        return jfull
    return reason


def stream_cases(path: Path, start: int, end: int):
    """Yield (idx, record_dict) for indexes in [start, end] inclusive."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i > end:
                break
            yield i, json.loads(line)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/filtered/keelung_drug_all.jsonl")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    in_path = ROOT / args.input
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    short = 0
    with out_path.open("w", encoding="utf-8") as out:
        for idx, rec in stream_cases(in_path, args.start, args.end):
            reason = get_reason(rec["jfull"])
            row = {
                "idx": idx,
                "jid": rec["jid"],
                "jcase": rec["jcase"],
                "reason": reason,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if len(reason) < 200:
                short += 1
    print(f"wrote {n} cases to {out_path}; {short} cases with <200 char reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
