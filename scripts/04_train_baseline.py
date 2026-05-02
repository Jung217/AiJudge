"""Baseline XGBoost regression for sentence_months.

Predicts 有期徒刑 month-count from features extracted by features.py.
Splits 80/20 train/test, reports MAE/RMSE/R^2 and feature importance.

Usage:
    python scripts/04_train_baseline.py
    python scripts/04_train_baseline.py --in data/filtered/keelung_drug_all.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import extract_features  # noqa: E402
from records import Record  # noqa: E402


BEHAVIORS = ["施用", "持有", "販賣", "運輸", "製造", "轉讓", "意圖販賣而持有"]
DRUG_LEVELS = [1, 2, 3, 4]


def build_dataframe(jsonl_path: Path) -> pd.DataFrame:
    """Extract features for every case with a non-null sentence_months target."""
    rows: list[dict[str, float | int]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rec = Record.from_dict({
                "JID": d["jid"], "JYEAR": d["jyear"], "JCASE": d["jcase"],
                "JNO": d["jno"], "JDATE": d["jdate"], "JTITLE": d["jtitle"],
                "JFULL": d["jfull"], "JPDF": d.get("jpdf", ""),
            })
            f = extract_features(rec)
            if not f.sentence_months:
                continue  # skip 拘役-only and label-less cases
            row = {
                "jid": rec.jid,
                "jyear": int(rec.jyear) if rec.jyear.isdigit() else 0,
                "art17_1": int(f.art17_1_applied),
                "art17_2": int(f.art17_2_applied),
                "art59": int(f.art59_applied),
                "recidivism": int(f.recidivism),
                "self_surrender": int(f.self_surrender),
                "can_convert_to_fine": int(f.can_convert_to_fine),
                "n_behaviors": len(f.behaviors),
                "n_drug_levels": len(f.drug_levels),
                "sentence_months": f.sentence_months,
            }
            for b in BEHAVIORS:
                row[f"b_{b}"] = int(b in f.behaviors)
            for lv in DRUG_LEVELS:
                row[f"lv_{lv}"] = int(lv in f.drug_levels)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/filtered/keelung_drug_all.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=500,
                    help="max boosting rounds (early-stopping enabled)")
    args = ap.parse_args()

    df = build_dataframe(args.inp)
    print(f"loaded {len(df)} cases with sentence_months")
    print(f"  median={df.sentence_months.median():.0f}  "
          f"mean={df.sentence_months.mean():.1f}  "
          f"max={df.sentence_months.max()}")

    feature_cols = [c for c in df.columns
                    if c not in ("jid", "sentence_months")]
    X = df[feature_cols].astype(float).values
    y = df["sentence_months"].astype(float).values
    print(f"features ({len(feature_cols)}): {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    print(f"train={len(X_train)} test={len(X_test)}")

    model = xgb.XGBRegressor(
        n_estimators=args.rounds,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=args.seed,
        early_stopping_rounds=30,
        eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print()
    print(f"Test set (n={len(X_test)}):")
    print(f"  MAE  = {mae:.2f} months")
    print(f"  RMSE = {rmse:.2f} months")
    print(f"  R^2  = {r2:.3f}")
    print(f"  best_iteration = {model.best_iteration}")

    # Naive baselines for comparison
    median_pred = np.full_like(y_test, np.median(y_train))
    print(f"\nMedian-baseline MAE = {mean_absolute_error(y_test, median_pred):.2f}")
    mean_pred = np.full_like(y_test, np.mean(y_train))
    print(f"Mean-baseline   MAE = {mean_absolute_error(y_test, mean_pred):.2f}")

    print("\nTop 15 features by gain importance:")
    importance = model.get_booster().get_score(importance_type="gain")
    feat_imp = sorted(importance.items(), key=lambda x: -x[1])[:15]
    name_map = {f"f{i}": name for i, name in enumerate(feature_cols)}
    for f, score in feat_imp:
        print(f"  {name_map.get(f, f):28} {score:>10.2f}")

    print("\nResidual analysis (|pred - y|):")
    abs_err = np.abs(pred - y_test)
    for pct in (50, 75, 90, 95, 99):
        print(f"  p{pct} = {np.percentile(abs_err, pct):.1f} months")
    print(f"  worst case: |{int(abs_err.max())}| months "
          f"(pred={int(pred[abs_err.argmax()])}, true={int(y_test[abs_err.argmax()])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
