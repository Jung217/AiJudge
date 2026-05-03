"""Helper to dump reason sections for §57 extraction.

This script extracts the reasoning section from each case in a slice of the
filtered jsonl, and writes them to a JSONL with `jid`, `jcase`, `reason`,
ready for hand-classification of §57 factors.

Run with PYTHONIOENCODING=utf-8 to avoid Windows cp950 stdout issues.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import _extract_section


def extract_reason(jfull: str) -> str:
    """Extract the reasoning section, with multiple fallbacks.

    Order of preference:
      1. 事實及理由 (compound header — common in 簡易判決)
      2. 理由 (separate header — 通常判決)
      3. 論罪科刑 (used in 簡易判決 that split 犯罪事實 / 論罪科刑)
      4. Whole document (last resort)
    """
    reason = _extract_section(
        jfull, "事實及理由",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官"),
    )
    if len(reason) >= 50:
        return reason
    reason = _extract_section(
        jfull, "理由",
        (r"附.{0,3}表", r"中\s*華\s*民\s*國\s*\d{3}"),
    )
    if len(reason) >= 50:
        return reason
    reason = _extract_section(
        jfull, "論罪科刑",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官"),
    )
    if len(reason) >= 50:
        return reason
    return jfull


def main() -> None:
    src = Path(r"C:\Users\alex2\Desktop\vsCode\AiJudge\data\filtered\keelung_drug_all.jsonl")
    out = Path(r"/tmp/art57_batch_2_reasons.jsonl")
    start, end = 320, 640
    with src.open(encoding="utf-8") as fin:
        rows = [json.loads(l) for l in fin]
    sub = rows[start:end]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout:
        for i, d in enumerate(sub):
            jfull = d.get("jfull", "")
            reason = extract_reason(jfull)
            fout.write(json.dumps({
                "idx": start + i,
                "jid": d.get("jid", ""),
                "jcase": d.get("jcase", ""),
                "reason": reason,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(sub)} reasons to {out}")


if __name__ == "__main__":
    main()
