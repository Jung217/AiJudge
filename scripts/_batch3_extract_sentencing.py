"""Helper: For each case in batch 3 (idx 640-959), extract the §57
sentencing-discussion portion (爰審酌 ... 一切情狀，量處...). Output
to a single jsonl file for batch reading.
"""
import json
import re
from pathlib import Path

INPUT = Path("/tmp/art57_batch_3_reasons.jsonl")
OUTPUT = Path("/tmp/art57_batch_3_sentencing.jsonl")

# Common opening markers for §57 discussion
START_PATS = [
    r"爰以行為人責任為基礎",
    r"爰審酌",
    r"爰斟酌",
    r"爰考量",
    r"本院審酌",
    r"審酌被告",
    r"茲審酌",
    r"並審酌",
]
# Common ending markers
END_PATS = [
    r"一切情狀[，,]?\s*量處",
    r"綜合考量[^。]*量處",
    r"量處如主文",
    r"以資懲儆",
    r"以示懲儆",
    r"以資警惕",
    r"以資警懲",
]


def find_sentencing(txt: str) -> tuple[int, int, str]:
    start_idx = -1
    for pat in START_PATS:
        m = re.search(pat, txt)
        if m and (start_idx < 0 or m.start() < start_idx):
            start_idx = m.start()
    if start_idx < 0:
        return -1, -1, ""

    # Search end pattern after start_idx
    end_idx = -1
    rest = txt[start_idx:]
    for pat in END_PATS:
        m = re.search(pat, rest)
        if m and (end_idx < 0 or m.end() < end_idx):
            end_idx = m.end()
    if end_idx < 0:
        # Fallback: take 1500 chars
        return start_idx, start_idx + 1500, txt[start_idx:start_idx + 1500]
    # Add 100 chars buffer
    end_abs = start_idx + end_idx + 100
    return start_idx, end_abs, txt[start_idx:end_abs]


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(INPUT, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            s, e, sent = find_sentencing(row["reason"])
            obj = {
                "idx": row["idx"],
                "jid": row["jid"],
                "jcase": row.get("jcase", ""),
                "jtitle": row.get("jtitle", ""),
                "sent_start": s,
                "sent_end": e,
                "sent_text": sent,
                "sent_len": len(sent),
                "reason_full": row["reason"],  # keep for fallback
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
