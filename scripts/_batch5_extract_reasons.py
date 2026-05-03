"""Helper: extract reason text for cases at index 1280-1597 inclusive
into a single jsonl. Each line: {"idx", "jid", "jcase", "jtitle", "reason"}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import _extract_section  # noqa: E402

INPUT = Path("data/filtered/keelung_drug_all.jsonl")
OUTPUT = Path("/tmp/art57_batch_5_reasons.jsonl")

START = 1280
END = 1598  # exclusive


def extract_reason(jfull: str) -> str:
    MIN_LEN = 200
    # Try compound 事實及理由 first
    text = _extract_section(
        jfull, "事實及理由",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官"),
    )
    if len(text) >= MIN_LEN:
        return text
    # Try 理由 with safer end patterns (avoid stripping 附表 references inside)
    text = _extract_section(
        jfull, "理由",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官", r"附\s*錄"),
    )
    if len(text) >= MIN_LEN:
        return text
    # 簡易判決 補充式 — reasoning lives entirely under 犯罪事實
    text = _extract_section(
        jfull, "犯罪事實",
        (r"中\s*華\s*民\s*國\s*\d{3}", r"書\s*記\s*官", r"附\s*錄"),
    )
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
