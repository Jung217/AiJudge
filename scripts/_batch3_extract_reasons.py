"""Helper: extract reason text for cases at index 640-959 inclusive
into a single jsonl. Each line: {"idx", "jid", "jcase", "jtitle", "reason"}.

Uses a more robust extraction strategy than features._extract_section:
the original implementation truncates at "附表" which appears mid-reason
in some 訴/重訴 cases. We instead look for sentencing-end markers.
"""
import json
import re
import sys
from pathlib import Path

INPUT = Path("data/filtered/keelung_drug_all.jsonl")
OUTPUT = Path("/tmp/art57_batch_3_reasons.jsonl")

START = 640
END = 960  # exclusive


def extract_section(jfull: str, start_marker: str, end_patterns: tuple[str, ...]) -> str:
    start_re = r"\s*".join(re.escape(c) for c in start_marker)
    m = re.search(start_re, jfull)
    if not m:
        return ""
    tail = jfull[m.end():]
    end_idx = len(tail)
    for pat in end_patterns:
        em = re.search(pat, tail)
        if em and em.start() < end_idx:
            end_idx = em.start()
    return tail[:end_idx]


def extract_reason(jfull: str) -> str:
    MIN_LEN = 50
    # End patterns: only trailing date and 書記官 (NOT 附表, which appears mid-reason)
    END_TIGHT = (r"中\s*華\s*民\s*國\s*\d{3}\s*年", r"書\s*記\s*官", r"以\s*上\s*正\s*本")
    # Try compound 事實及理由 first
    text = extract_section(jfull, "事實及理由", END_TIGHT)
    if len(text) < MIN_LEN:
        text = extract_section(jfull, "理由", END_TIGHT)
    if len(text) < MIN_LEN:
        # Last resort: from "事實" forward
        text = extract_section(jfull, "事實", END_TIGHT)
    return text


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(INPUT, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i < START:
                continue
            if i >= END:
                break
            row = json.loads(line)
            reason = extract_reason(row["jfull"])
            obj = {
                "idx": i,
                "jid": row["jid"],
                "jcase": row.get("jcase", ""),
                "jtitle": row.get("jtitle", ""),
                "reason": reason,
                "reason_len": len(reason),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
