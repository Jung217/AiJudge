"""Precompute bucket → median-sentence lookup for the GitHub Pages 互動 demo.

For each (court, behavior, drug_level, art17_1, art17_2, art59, self_surrender,
attempt, summary, recidivism) tuple we materialise the median (and n) of
``sentence_months`` from the training jsonl. The page JS reads the resulting
JSON, looks up the user's selected feature combo, and reports a "近年同類案件
中位數刑期" alongside the rule-derived statutory range.

Run:
    python scripts/08_build_page_buckets.py
        --in data/filtered/north5_drug_all.jsonl
        --out docs/assets/sentence_buckets.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import extract_features  # noqa: E402
from records import Record  # noqa: E402


BEHAVIORS = ["施用", "持有", "販賣", "運輸", "製造", "轉讓", "意圖販賣而持有"]
DRUG_LEVELS = [1, 2, 3, 4]
COURTS = ["KL", "TP", "SL", "PC", "TY"]


def _primary_behavior(behs: list[str]) -> str | None:
    """同 04_train_baseline 的 priority,讓 lookup 只看主行為。"""
    order = ["運輸", "製造", "販賣", "意圖販賣而持有", "轉讓", "持有", "施用"]
    for b in order:
        if b in behs:
            return b
    return None


def _key(court: str, behavior: str, drug_level: int, art17_1: bool,
         art17_2: bool, art59: bool, self_surrender: bool, attempt: bool,
         summary: bool, recidivism: bool) -> str:
    """Stable key string used both Python-side and JS-side."""
    flags = "".join(["1" if v else "0" for v in
                     (art17_1, art17_2, art59, self_surrender,
                      attempt, summary, recidivism)])
    return f"{court}|{behavior}|{drug_level}|{flags}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/filtered/north5_drug_all.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/assets/sentence_buckets.json"))
    ap.add_argument("--min-n", type=int, default=3,
                    help="drop buckets with fewer than N cases (default 3)")
    args = ap.parse_args()

    buckets: dict[str, list[int]] = defaultdict(list)
    n_total = 0
    n_used = 0
    with args.inp.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rec = Record.from_dict({
                "JID": d["jid"], "JYEAR": d["jyear"], "JCASE": d["jcase"],
                "JNO": d["jno"], "JDATE": d["jdate"], "JTITLE": d["jtitle"],
                "JFULL": d["jfull"], "JPDF": d.get("jpdf", ""),
            })
            f = extract_features(rec)
            n_total += 1
            # Skip 拘役 / 無刑期 / 多被告 / 數罪併罰 — these don't compare cleanly.
            if not f.sentence_months or f.n_defendants > 1:
                continue
            if f.is_aggregate_sentence or f.n_sentence_counts > 1:
                continue
            primary = _primary_behavior(f.convicted_behaviors)
            if primary is None or not f.convicted_drug_levels:
                continue
            level = min(f.convicted_drug_levels)
            court = d.get("court") or (rec.jid[:2] if rec.jid else "")
            if court not in COURTS:
                continue
            summary = "簡" in rec.jcase
            key = _key(court, primary, level,
                       f.art17_1_applied, f.art17_2_applied, f.art59_applied,
                       f.self_surrender, f.is_attempt, summary, f.recidivism)
            buckets[key].append(f.sentence_months)
            n_used += 1

    out_buckets: dict[str, dict] = {}
    dropped = 0
    for k, vals in buckets.items():
        if len(vals) < args.min_n:
            dropped += 1
            continue
        arr = np.array(vals, dtype=float)
        out_buckets[k] = {
            "n": int(arr.size),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "mean": round(float(arr.mean()), 1),
        }

    payload = {
        "schema": "bucket_v1",
        "key_format": "court|behavior|drug_level|art17_1+art17_2+art59+self_surrender+attempt+summary+recidivism (each 0/1)",
        "courts": COURTS,
        "behaviors": BEHAVIORS,
        "drug_levels": DRUG_LEVELS,
        "n_total_cases": n_used,
        "n_buckets": len(out_buckets),
        "n_buckets_dropped_under_min_n": dropped,
        "min_n": args.min_n,
        "buckets": out_buckets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"loaded {n_total} cases → {n_used} kept after filtering")
    print(f"wrote {len(out_buckets)} buckets ({dropped} dropped at min_n={args.min_n})"
          f" → {args.out}")
    print(f"top 5 buckets by size:")
    for k, v in sorted(out_buckets.items(), key=lambda kv: -kv[1]["n"])[:5]:
        print(f"  {k:<35} n={v['n']:>5}  p25/p50/p75 = "
              f"{v['p25']:.0f}/{v['p50']:.0f}/{v['p75']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
