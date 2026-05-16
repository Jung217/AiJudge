"""Dump top-K walk-forward outliers for manual JFULL inspection.

Helps answer "where does the residual 2.85 month MAE actually come from?":
each row gets pooled (y_true, y_pred, |residual|) and the worst N cases land
in a CSV with their JID + feature fingerprint so we can open the judgment in
the 司法院系統 and see what went wrong (extraction bug? label noise? missing
feature?).

Run:
    python scripts/09_inspect_outliers.py
    python scripts/09_inspect_outliers.py --top 200
    python scripts/09_inspect_outliers.py --in data/filtered/keelung_drug_all.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_train_module():
    # File name starts with a digit so import-by-name doesn't work directly.
    spec = importlib.util.spec_from_file_location(
        "train_baseline_module", "scripts/04_train_baseline.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["train_baseline_module"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/filtered/north5_drug_all.jsonl"))
    ap.add_argument("--art57", type=str,
                    default="data/processed/art57_factors.jsonl")
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/outliers.csv"))
    ap.add_argument("--top", type=int, default=100,
                    help="how many worst-residual rows to dump")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--per-defendant", action="store_true",
                    help="match training side: by default this diagnostic uses "
                         "single-row mode (= production model). Pass this flag "
                         "to use the experimental per-defendant V1 path.")
    args = ap.parse_args()

    tb = _load_train_module()
    art57_path = Path(args.art57) if args.art57 else None

    print(f"loading {args.inp} (per_defendant={args.per_defendant})…", flush=True)
    df = tb.build_dataframe(args.inp, art57_path, per_defendant=args.per_defendant)
    feat_cols = tb.feature_columns(df)
    print(f"  rows={len(df)}  features={len(feat_cols)}", flush=True)

    print(f"walk-forward CV ({args.folds} folds)…", flush=True)
    results = tb.walk_forward(
        df, feat_cols, folds=args.folds, rule_clip=True,
        seed=args.seed, rounds=args.rounds, calibrate=True,
    )

    pool: list[dict] = []
    for r in results:
        te = df.iloc[r["te_idx"]]
        for (_, row), yt, yp, p25v, p75v in zip(
                te.iterrows(), r["y_true"], r["y_pred"], r["p25"], r["p75"]):
            pool.append({
                "jid": row["jid"],
                "jdate": row["_jdate"],
                "court": row["_court"],
                "convicted_behaviors": row["_conv_beh"],
                "convicted_levels": row["_conv_lv"],
                "art17_1": int(row["art17_1"]),
                "art17_2": int(row["art17_2"]),
                "art59": int(row["art59"]),
                "self_surrender": int(row["self_surrender"]),
                "is_attempt": int(row["is_attempt"]),
                "recidivism": int(row["recidivism"]),
                "summary": int(row["_summary"]),
                "n_sentence_counts": int(row["n_sentence_counts"]),
                "is_aggregate_sentence": int(row["is_aggregate_sentence"]),
                "weight_g": float(row["max_drug_weight_g"]),
                "y_true": float(yt),
                "y_pred_p50": float(yp),
                "y_pred_p25": float(p25v),
                "y_pred_p75": float(p75v),
                "abs_residual": float(abs(yt - yp)),
                "signed_residual": float(yp - yt),
                # 越界檢查:y_true 是否落在 [p25, p75] 區間
                "in_band": int(p25v - 1e-6 <= yt <= p75v + 1e-6),
            })

    rdf = pd.DataFrame(pool).sort_values("abs_residual", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rdf.head(args.top).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nwrote top {args.top} outliers → {args.out}", flush=True)

    # ── 統計摘要 ────────────────────────────────────────────────────
    print(f"\n── residual distribution (all {len(rdf)} pooled rows) ──")
    abs_r = rdf["abs_residual"].values
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p:>2}  |err| = {np.percentile(abs_r, p):.1f} 月")
    print(f"\n  覆蓋率 [p25, p75]: {rdf['in_band'].mean():.1%}")

    # 前 100 名:按主要行為 / 法院 / 殘差方向(model 過估 vs 低估)分組
    top = rdf.head(args.top)
    print(f"\n── top {args.top} outlier 主要行為分布 ──")
    behs = top["convicted_behaviors"].str.split(",").str[0].value_counts()
    for k, v in behs.items():
        print(f"  {k or '(空)':<15} {v}")

    print(f"\n── top {args.top} 法院分布 ──")
    for k, v in top["court"].value_counts().items():
        print(f"  {k:<4} {v}")

    print(f"\n── top {args.top} 殘差方向 ──")
    n_over = int((top["signed_residual"] > 0).sum())
    n_under = int((top["signed_residual"] < 0).sum())
    mean_over = top.loc[top["signed_residual"] > 0, "signed_residual"].mean()
    mean_under = top.loc[top["signed_residual"] < 0, "signed_residual"].mean()
    print(f"  模型過估 (pred > true): {n_over} 件,平均 +{mean_over:.1f} 月")
    print(f"  模型低估 (pred < true): {n_under} 件,平均 {mean_under:.1f} 月")

    print(f"\n── top {args.top} 是否有減刑 flag ──")
    flag_cols = ["art17_1", "art17_2", "art59", "self_surrender",
                 "is_attempt", "recidivism", "summary",
                 "is_aggregate_sentence"]
    for f in flag_cols:
        n = int(top[f].sum())
        print(f"  {f:<25} {n}/{args.top}  ({n/args.top:.0%})")

    print(f"\nfirst 15 rows by |residual|:")
    cols = ["jid", "court", "convicted_behaviors", "convicted_levels",
            "y_true", "y_pred_p50", "signed_residual",
            "is_aggregate_sentence", "summary"]
    with pd.option_context("display.max_colwidth", 40,
                            "display.width", 200):
        print(top[cols].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
