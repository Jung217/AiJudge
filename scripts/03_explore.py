"""Basic statistics on filtered Keelung drug cases.

Usage:
    python scripts/03_explore.py
    python scripts/03_explore.py --in data/filtered/keelung_drug.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import extract_features  # noqa: E402
from records import Record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path,
                        default=Path("data/filtered/keelung_drug.jsonl"))
    args = parser.parse_args()

    if not args.inp.exists():
        print(f"error: input {args.inp} does not exist", file=sys.stderr)
        return 1

    total = 0
    by_year: Counter[str] = Counter()
    by_case_type: Counter[str] = Counter()
    by_drug_level: Counter[int] = Counter()
    by_behavior: Counter[str] = Counter()
    art17_1 = 0
    art17_2 = 0
    recidivism = 0
    art59 = 0
    probation = 0
    sentences: list[int] = []

    with args.inp.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            r = Record.from_dict({
                "JID": d["jid"], "JYEAR": d["jyear"], "JCASE": d["jcase"],
                "JNO": d["jno"], "JDATE": d["jdate"], "JTITLE": d["jtitle"],
                "JFULL": d["jfull"], "JPDF": d.get("jpdf", ""),
            })
            feats = extract_features(r)
            total += 1
            by_year[r.jyear] += 1
            by_case_type[r.jcase] += 1
            for lvl in feats.drug_levels:
                by_drug_level[lvl] += 1
            for b in feats.behaviors:
                by_behavior[b] += 1
            if feats.art17_1_applied:
                art17_1 += 1
            if feats.art17_2_applied:
                art17_2 += 1
            if feats.recidivism:
                recidivism += 1
            if feats.art59_applied:
                art59 += 1
            if feats.probation_months:
                probation += 1
            if feats.sentence_months:
                sentences.append(feats.sentence_months)

    print(f"Total cases:              {total}")
    if total == 0:
        return 0

    def pct(n: int) -> str:
        return f"{n} ({n / total:.1%})"

    print(f"By year (民國):           {dict(sorted(by_year.items()))}")
    print(f"By 字別:                   {dict(by_case_type.most_common())}")
    print(f"By drug level:            {dict(sorted(by_drug_level.items()))}")
    print(f"By behavior:              {dict(by_behavior.most_common())}")
    print(f"§17 Ⅰ applied:            {pct(art17_1)}")
    print(f"§17 Ⅱ applied:            {pct(art17_2)}")
    print(f"累犯:                      {pct(recidivism)}")
    print(f"§59 酌減:                  {pct(art59)}")
    print(f"宣告緩刑:                  {pct(probation)}")

    if sentences:
        sentences.sort()
        n = len(sentences)
        print(f"\nSentence (months), n={n}:")
        print(f"  p25    = {sentences[n // 4]}")
        print(f"  median = {sentences[n // 2]}")
        print(f"  p75    = {sentences[3 * n // 4]}")
        print(f"  mean   = {sum(sentences) / n:.1f}")
        print(f"  max    = {sentences[-1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
