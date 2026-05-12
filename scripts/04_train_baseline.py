"""Baseline XGBoost regression for sentence_months.

Predicts 有期徒刑 month-count from features extracted by features.py.
Splits 80/20 train/test, reports MAE/RMSE/R^2, feature importance, and the
法定刑越界率 (statutory-range violation rate) before/after layer-4 clipping.

NOTE: this is a quick baseline using a *random* split — plan.md §6.1 requires
walk-forward temporal CV for any reported headline number; treat the MAE here
as a sanity check, not a validated metric.

Usage:
    python scripts/04_train_baseline.py
    python scripts/04_train_baseline.py --in data/filtered/keelung_drug_all.jsonl
    python scripts/04_train_baseline.py --save data/processed/baseline_model.pkl
    python scripts/04_train_baseline.py --no-rule-clip   # ablation
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
from models import ModelBundle  # noqa: E402
from records import Record  # noqa: E402
from rules import (  # noqa: E402
    binding_constraint,
    clip_prediction,
    violation_rate,
)


BEHAVIORS = ["施用", "持有", "販賣", "運輸", "製造", "轉讓", "意圖販賣而持有"]
DRUG_LEVELS = [1, 2, 3, 4]
ART57_FACTORS = ["motive", "provocation", "means", "life_status", "character",
                 "intellect", "relation_victim", "duty_breach", "harm", "post_attitude"]


def _load_art57(path: Path) -> dict[str, dict[str, str]]:
    """Returns jid -> {factor_name: direction}. Empty dict if path missing."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["jid"]] = {f: v.get("direction", "absent")
                              for f, v in r.get("factors", {}).items()}
    return out


def build_dataframe(jsonl_path: Path, art57_path: Path | None) -> pd.DataFrame:
    """Extract features for every case with a non-null sentence_months target."""
    art57 = _load_art57(art57_path) if art57_path else {}
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
            wt = f.max_drug_weight_g
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
                "max_drug_weight_g": wt if wt is not None else -1.0,
                "has_drug_weight": int(wt is not None),
                "log_drug_weight": np.log1p(wt) if wt is not None else 0.0,
                "sentence_months": f.sentence_months,
                # Stored for rule-clip lookup (not fed as features).
                "_conv_beh": ",".join(f.convicted_behaviors),
                "_conv_lv": ",".join(map(str, f.convicted_drug_levels)),
            }
            # §57 factors as one-hot per direction (absent → all zeros for that factor)
            factors = art57.get(rec.jid, {})
            for fname in ART57_FACTORS:
                d_ = factors.get(fname, "absent")
                row[f"a57_{fname}_mit"] = int(d_ == "mitigating")
                row[f"a57_{fname}_agg"] = int(d_ == "aggravating")
                row[f"a57_{fname}_neu"] = int(d_ == "neutral")
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
    ap.add_argument("--art57", type=str,
                    default="data/processed/art57_factors.jsonl",
                    help="JSONL of §57 directions (set to '' to disable)")
    ap.add_argument("--no-art57", action="store_true",
                    help="disable §57 features for comparison")
    ap.add_argument("--no-rule-clip", action="store_true",
                    help="skip layer-4 statutory clipping (for ablation only — "
                         "production must keep it on)")
    ap.add_argument("--save", type=Path, default=None,
                    help="path to dump the trained ModelBundle (pickle)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=500,
                    help="max boosting rounds (early-stopping enabled)")
    args = ap.parse_args()

    art57_path = None if args.no_art57 or not args.art57 else Path(args.art57)
    df = build_dataframe(args.inp, art57_path)
    print(f"loaded {len(df)} cases with sentence_months")
    print(f"  median={df.sentence_months.median():.0f}  "
          f"mean={df.sentence_months.mean():.1f}  "
          f"max={df.sentence_months.max()}")

    feature_cols = [c for c in df.columns
                    if c not in ("jid", "sentence_months")
                    and not c.startswith("_")]
    X = df[feature_cols].astype(float).values
    y = df["sentence_months"].astype(float).values
    print(f"features ({len(feature_cols)}): {feature_cols}")

    indices = np.arange(len(X))
    i_train, i_test = train_test_split(indices, test_size=0.2, random_state=args.seed)
    X_train, X_test = X[i_train], X[i_test]
    y_train, y_test = y[i_train], y[i_test]
    X_test_idx = i_test
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

    pred_raw = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_test, pred_raw))
    r2 = r2_score(y_test, pred_raw)

    print()
    print(f"Test set (n={len(X_test)}):")
    print(f"  MAE  = {mae:.2f} months  (raw)")
    print(f"  RMSE = {rmse:.2f} months  (raw)")
    print(f"  R^2  = {r2:.3f}              (raw)")
    print(f"  best_iteration = {model.best_iteration}")

    # ── Layer 4: statutory clipping + 法定刑越界率 ────────────────────────
    #
    # The binding constraint is looked up from 主文-only convicted behaviors
    # (`_conv_beh` / `_conv_lv`), not the union of acts mentioned anywhere in
    # the judgment — the narrative often cites acts that weren't the actual
    # conviction (e.g. 起訴 販賣 but court convicted 施用), which would make the
    # statutory floor overshoot. With conviction-only lookup the clip is sound
    # and helps MAE slightly.
    #
    # 越界率 is reported three ways:
    #   - ground-truth labels: a real judge's sentence can't be illegal, so a
    #     non-zero value flags an upstream defect — either the penalty table
    #     (rules.py) or, more often, conviction extraction (convicted_behaviors
    #     / convicted_drug_levels picking up acts/levels that weren't the
    #     conviction, or sentence_months being a 數罪併罰 應執行刑). Use it as a
    #     data-quality canary, not a model metric.
    #   - raw model preds: how often the unconstrained model leaves the range.
    #   - rule-clipped preds: must be exactly 0.0% — the layer-4 guarantee.
    df_test = df.iloc[X_test_idx].reset_index(drop=True)
    constraints = []
    for _, row in df_test.iterrows():
        behaviors = {b for b in row["_conv_beh"].split(",") if b}
        levels = {int(lv) for lv in row["_conv_lv"].split(",") if lv}
        constraints.append(binding_constraint(
            behaviors, levels,
            art17_1=bool(row["art17_1"]),
            art17_2=bool(row["art17_2"]),
            art59=bool(row["art59"]),
            recidivism=bool(row["recidivism"]),
        ))
    n_no_constraint = sum(c is None for c in constraints)

    rate_gt, n_gt, n_chk = violation_rate(list(zip(y_test, constraints)))
    rate_raw, n_rawv, _ = violation_rate(list(zip(pred_raw, constraints)))
    print(f"\n法定刑越界率 (constraint resolved for {n_chk}/{len(X_test)} cases; "
          f"{n_no_constraint} unmapped):")
    print(f"  ground-truth labels : {rate_gt:6.2%}  ({n_gt} cases — "
          f"{'clean' if n_gt == 0 else 'upstream defect: penalty table or conviction extraction'})")
    print(f"  raw model preds     : {rate_raw:6.2%}  ({n_rawv} cases)")

    pred = pred_raw
    if not args.no_rule_clip:
        pred_clipped = pred_raw.copy()
        n_clipped = 0
        for i, c in enumerate(constraints):
            if c is None:
                continue
            new_p = clip_prediction(float(pred_raw[i]), c)
            if abs(new_p - pred_raw[i]) > 0.01:
                n_clipped += 1
            pred_clipped[i] = new_p
        rate_clip, n_clipv, _ = violation_rate(list(zip(pred_clipped, constraints)))
        mae_c = mean_absolute_error(y_test, pred_clipped)
        rmse_c = np.sqrt(mean_squared_error(y_test, pred_clipped))
        r2_c = r2_score(y_test, pred_clipped)
        print(f"  rule-clipped preds  : {rate_clip:6.2%}  ({n_clipv} cases — "
              f"{'OK' if n_clipv == 0 else 'LAYER-4 FAILURE'})")
        print(f"\nRule-clipped test metrics ({n_clipped}/{len(X_test)} clipped):")
        print(f"  MAE  = {mae_c:.2f} months")
        print(f"  RMSE = {rmse_c:.2f} months")
        print(f"  R^2  = {r2_c:.3f}")
        pred = pred_clipped

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

    if args.save:
        import pickle
        bundle = ModelBundle(
            sentence_regressor=model,
            feature_names=tuple(feature_cols),
            metadata={
                "trained_on": str(args.inp),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "mae_raw": float(mae),
                "rule_clip": not args.no_rule_clip,
                "art57": art57_path is not None,
            },
        )
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("wb") as fh:
            pickle.dump(bundle, fh)
        print(f"\nSaved ModelBundle → {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
