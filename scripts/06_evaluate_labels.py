"""Evaluate auto-extraction against human ground-truth labels.

Reads data/labeling/sample.csv (as filled in Excel / Sheets) and reports
per-field agreement metrics:
  - binary flags (art17_1, art17_2, ...): precision / recall / F1
  - list fields (behaviors, drug_levels): Jaccard, precision, recall
  - numeric (sentence_months, detention_days, probation_months):
        exact-match %, MAE (months or days)

Rows with empty gt_* columns are skipped (un-labeled).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BIN_FIELDS = ("art17_1", "art17_2", "art59", "recidivism", "can_convert_to_fine")
SET_FIELDS = ("behaviors", "drug_levels")
NUM_FIELDS = ("sentence_months", "detention_days", "probation_months")


def _parse_bool(s: str) -> bool | None:
    s = (s or "").strip().lower()
    if s in ("1", "true", "t", "y", "yes", "v", "✓"):
        return True
    if s in ("0", "false", "f", "n", "no", "x"):
        return False
    return None


def _parse_int(s: str) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_set(s: str) -> set[str]:
    return {t.strip() for t in (s or "").split(",") if t.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/labeling/sample.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.inp.open(encoding="utf-8-sig")))
    print(f"read {len(rows)} rows from {args.inp}")

    labeled = [r for r in rows
               if any((r.get(f"gt_{f}") or "").strip() for f in
                      [*BIN_FIELDS, *SET_FIELDS, *NUM_FIELDS])]
    print(f"{len(labeled)} rows have at least one ground-truth cell filled\n")

    # Binary fields
    for field in BIN_FIELDS:
        tp = fp = fn = tn = 0
        covered = 0
        for r in labeled:
            gt = _parse_bool(r.get(f"gt_{field}", ""))
            if gt is None:
                continue
            auto = _parse_bool(r.get(f"auto_{field}", ""))
            if auto is None:
                auto = False
            covered += 1
            if gt and auto: tp += 1
            elif gt and not auto: fn += 1
            elif not gt and auto: fp += 1
            else: tn += 1
        if covered == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / covered
        print(f"  {field:25} n={covered:3d}  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}  acc={acc:.2f}  (tp={tp} fp={fp} fn={fn} tn={tn})")

    # Set (multi-value) fields
    print()
    for field in SET_FIELDS:
        total_jac = 0.0
        pt = pd = gt_total = 0
        covered = 0
        for r in labeled:
            gt_raw = r.get(f"gt_{field}", "")
            if not gt_raw.strip():
                continue
            gt = _parse_set(gt_raw)
            auto = _parse_set(r.get(f"auto_{field}", ""))
            if not gt and not auto:
                continue
            covered += 1
            inter = gt & auto
            union = gt | auto
            total_jac += len(inter) / len(union) if union else 1.0
            pt += len(inter)
            pd += len(auto)
            gt_total += len(gt)
        if covered == 0:
            continue
        jac = total_jac / covered
        prec = pt / pd if pd else 0.0
        rec = pt / gt_total if gt_total else 0.0
        print(f"  {field:25} n={covered:3d}  Jaccard={jac:.2f}  P={prec:.2f}  R={rec:.2f}")

    # Numeric fields
    print()
    for field in NUM_FIELDS:
        exact = 0
        errs: list[int] = []
        covered = 0
        for r in labeled:
            gt = _parse_int(r.get(f"gt_{field}", ""))
            if gt is None:
                continue
            auto = _parse_int(r.get(f"auto_{field}", "")) or 0
            covered += 1
            if gt == auto:
                exact += 1
            errs.append(abs(gt - auto))
        if covered == 0:
            continue
        em = exact / covered
        mae = sum(errs) / covered
        print(f"  {field:25} n={covered:3d}  exact-match={em:.0%}  MAE={mae:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
