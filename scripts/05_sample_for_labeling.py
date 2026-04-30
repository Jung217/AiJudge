"""Sample N filtered cases and emit a CSV for human ground-truth labeling.

Workflow:
  1. python scripts/05_sample_for_labeling.py --n 50
     → writes data/labeling/sample.csv (UTF-8 BOM; opens in Excel)
  2. Human opens the CSV, fills gt_* columns by reading the snippet/full text.
  3. python scripts/06_evaluate_labels.py
     → prints per-field precision / recall / exact-match vs auto_*.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import extract_features  # noqa: E402
from records import Record  # noqa: E402


FIELDS = [
    "jid", "jdate", "jcase", "jtitle",
    "main_snippet", "facts_snippet",
    "auto_behaviors", "auto_drug_levels",
    "auto_sentence_months", "auto_detention_days", "auto_probation_months",
    "auto_art17_1", "auto_art17_2", "auto_art59", "auto_recidivism",
    "auto_can_convert_to_fine",
    "gt_behaviors", "gt_drug_levels",
    "gt_sentence_months", "gt_detention_days", "gt_probation_months",
    "gt_art17_1", "gt_art17_2", "gt_art59", "gt_recidivism",
    "gt_can_convert_to_fine",
    "notes",
]


def _snippet(jfull: str, anchor: str, before: int, after: int) -> str:
    """Find anchor, tolerating full-width spaces between characters
    (judgments often write "主　文", "犯　罪　事　實")."""
    import re
    pat = r"\s*".join(re.escape(c) for c in anchor)
    m = re.search(pat, jfull)
    if not m:
        return ""
    return jfull[max(0, m.start() - before): m.start() + after].replace("\n", " ").replace("\r", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/filtered/keelung_drug.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/labeling/sample.csv"))
    ap.add_argument("--n", type=int, default=50, help="sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefill", action="store_true",
                    help="copy auto_* into gt_* so you only correct wrong cells")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    random.Random(args.seed).shuffle(rows)
    sampled = rows[:args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in sampled:
            rec = Record.from_dict({
                "JID": r["jid"], "JYEAR": r["jyear"], "JCASE": r["jcase"],
                "JNO": r["jno"], "JDATE": r["jdate"], "JTITLE": r["jtitle"],
                "JFULL": r["jfull"], "JPDF": r.get("jpdf", ""),
            })
            f = extract_features(rec)
            auto = {
                "auto_behaviors": ",".join(f.behaviors),
                "auto_drug_levels": ",".join(map(str, f.drug_levels)),
                "auto_sentence_months": f.sentence_months or "",
                "auto_detention_days": f.detention_days or "",
                "auto_probation_months": f.probation_months or "",
                "auto_art17_1": int(f.art17_1_applied),
                "auto_art17_2": int(f.art17_2_applied),
                "auto_art59": int(f.art59_applied),
                "auto_recidivism": int(f.recidivism),
                "auto_can_convert_to_fine": int(f.can_convert_to_fine),
            }
            gt = ({k.replace("auto_", "gt_"): v for k, v in auto.items()}
                  if args.prefill
                  else {f"gt_{k}": "" for k in
                        ("behaviors", "drug_levels", "sentence_months",
                         "detention_days", "probation_months", "art17_1",
                         "art17_2", "art59", "recidivism",
                         "can_convert_to_fine")})
            w.writerow({
                "jid": r["jid"], "jdate": r["jdate"], "jcase": r["jcase"],
                "jtitle": r["jtitle"],
                "main_snippet": _snippet(r["jfull"], "主文", 0, 400)[:400],
                "facts_snippet": _snippet(r["jfull"], "犯罪事實", 0, 600)
                                  or _snippet(r["jfull"], "事實及理由", 0, 600)
                                  or _snippet(r["jfull"], "事實", 0, 600),
                **auto, **gt,
                "notes": "",
            })
    print(f"wrote {len(sampled)} rows to {args.out}")
    print(f"fill the gt_* columns in Excel / Sheets, then run 06_evaluate_labels.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
