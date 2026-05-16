"""Diagnose §17Ⅱ / §59 false-positive triggers.

Logic: if a case has art17_2_applied=True but the actual sentence is *above*
the unreduced §4 statutory floor for that level, §17Ⅱ couldn't have been
genuinely applied (otherwise the sentence would have been halved). Same for
§59. For each such case, dump a 240-char window around the citation so we
can spot common false-positive patterns (引文 / 駁回 with weak reject /
共犯減刑誤抓 etc.).
"""
from __future__ import annotations
import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout to survive cp950 console on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Floor before reduction (months) for §4 charges by drug level
S4_FLOOR = {1: 15 * 12, 2: 10 * 12, 3: 7 * 12, 4: 5 * 12}
ART17_CITE = re.compile(r"毒\s*品\s*危\s*害\s*防\s*制\s*條\s*例\s*第\s*(?:十七|17)\s*條"
                          r"\s*第\s*(?:[一二1])\s*項")
ART59_CITE = re.compile(r"刑\s*法\s*第\s*(?:五十九|59)\s*條")


def main():
    outliers = list(csv.DictReader(open("data/processed/outliers.csv",
                                          encoding="utf-8-sig")))
    print(f"loaded {len(outliers)} outlier rows")

    # Build jid → jfull lookup from the source jsonl(s)
    jid_to_jfull: dict[str, str] = {}
    for src in ["data/filtered/north5_drug_all.jsonl",
                 "data/filtered/keelung_drug_all.jsonl"]:
        if not Path(src).exists():
            continue
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                jid_to_jfull[d["jid"]] = d.get("jfull", "")

    print(f"jfull dict size: {len(jid_to_jfull)}")

    # Filter outliers that have §17Ⅱ=True but y_true is suspiciously high
    fp_17 = []
    fp_59 = []
    for r in outliers:
        try:
            y = float(r["y_true"])
            lv = int(r["convicted_levels"].split(",")[0]) if r["convicted_levels"] else 0
            behs = r["convicted_behaviors"].split(",") if r["convicted_behaviors"] else []
        except (ValueError, IndexError):
            continue
        # Only consider §4 charges (販賣/運輸/製造) for §17 false positives
        is_s4 = any(b in ("販賣", "運輸", "製造") for b in behs)
        if not is_s4:
            continue
        floor = S4_FLOOR.get(lv)
        if floor is None:
            continue
        # If §17Ⅱ truly applied, sentence ≤ floor (because floor is halved to floor/2).
        # If sentence > floor, §17Ⅱ couldn't have applied (or only applied to non-primary count).
        if int(r["art17_2"]) and y > floor * 0.7:
            fp_17.append((r["jid"], y, floor, lv, behs))
        if int(r["art59"]) and y > floor * 0.7:
            fp_59.append((r["jid"], y, floor, lv, behs))

    print(f"\n§17Ⅱ likely-false-positive outliers: {len(fp_17)}")
    print(f"§59 likely-false-positive outliers: {len(fp_59)}")

    # Dump JFULL windows around the citation for the first ~20 cases.
    def dump(fps, cite_re, name, n=15):
        print(f"\n── {name} ── (first {min(n, len(fps))} cases, 240-char window) ──")
        for jid, y, floor, lv, behs in fps[:n]:
            jfull = jid_to_jfull.get(jid, "")
            if not jfull:
                print(f"\n[{jid}] (jfull not loaded)")
                continue
            print(f"\n[{jid}]  y={y:.0f}月  floor={floor}月  lv={lv}  behs={behs}")
            for m in cite_re.finditer(jfull):
                lo = max(0, m.start() - 60)
                hi = min(len(jfull), m.end() + 180)
                snippet = jfull[lo:hi].replace("\n", " ")
                print(f"  …{snippet}…")
                break  # Just first occurrence

    dump(fp_17, ART17_CITE, "§17Ⅱ false-positive")
    dump(fp_59, ART59_CITE, "§59 false-positive")

    # Aggregate common keywords near the citation
    def common_phrases(fps, cite_re, name):
        print(f"\n── {name} 常見模式 ──")
        cnt = Counter()
        for jid, *_ in fps:
            jfull = jid_to_jfull.get(jid, "")
            if not jfull:
                continue
            for m in cite_re.finditer(jfull):
                lo = max(0, m.start() - 80)
                hi = min(len(jfull), m.end() + 80)
                window = jfull[lo:hi]
                for phrase in ("尚不符合", "無從適用", "無適用", "並無",
                                "未自白", "並非", "難認", "不予",
                                "毋庸", "毋待", "難謂", "無餘地",
                                "起訴書", "被害人", "本院認", "減輕之",
                                "減輕其刑", "減輕之列", "至於", "另案",
                                "共犯", "共同正犯"):
                    if phrase in window:
                        cnt[phrase] += 1
                break
        for p, c in cnt.most_common():
            print(f"  {p:<12} {c}")

    common_phrases(fp_17, ART17_CITE, "§17Ⅱ false-positive")
    common_phrases(fp_59, ART59_CITE, "§59 false-positive")


if __name__ == "__main__":
    main()
