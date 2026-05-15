"""Layered baseline (plan.md §5.2) — sentence quantiles + probation classifier.

Predicts (p25, p50, p75) months of 有期徒刑 via three XGBoost quantile heads
(``reg:quantileerror``) sharing the same feature matrix, plus an XGBClassifier
head for 緩刑. Evaluated with **walk-forward temporal cross-validation** (plan
§6.1) — the test fold is always strictly later than its training data. For
each fold the regressor reports MAE / pinball / coverage and the 法定刑越界率
before/after layer-4 statutory clipping (must be 0% after clipping); the
classifier reports accuracy / precision / recall / PR-AUC vs. base rate.

Multi-defendant judgments (n_defendants > 1 in 主文) are dropped by default:
the per-judgment row mixes several people's sentences and 沒收/§59 flags become
ambiguous. Pass --keep-multi-defendant to include them.

Usage:
    python scripts/04_train_baseline.py
    python scripts/04_train_baseline.py --in data/filtered/keelung_drug_all.jsonl
    python scripts/04_train_baseline.py --folds 6 --save data/processed/baseline_model.pkl
    python scripts/04_train_baseline.py --holdout      # quick random 80/20 sanity check
    python scripts/04_train_baseline.py --no-rule-clip # layer-4 ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import extract_features  # noqa: E402
from models import ModelBundle  # noqa: E402
from records import Record  # noqa: E402
from rules import (  # noqa: E402
    aggregate_only_constraint,
    binding_constraint,
    clip_prediction,
    violation_rate,
)


BEHAVIORS = ["施用", "持有", "販賣", "運輸", "製造", "轉讓", "意圖販賣而持有"]
DRUG_LEVELS = [1, 2, 3, 4]
# Northern 5-court codes — see filter.COURT_REGISTRY. "??" is the fallback for
# jsonls produced before the multi-court filter (no "court" field).
COURTS = ["KL", "TP", "SL", "PC", "TY"]
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
    rows: list[dict] = []
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
            court = d.get("court") or (rec.jid[:2] if rec.jid else "")
            row = {
                "jid": rec.jid,
                "jyear": int(rec.jyear) if rec.jyear.isdigit() else 0,
                "_court": court,
                "art17_1": int(f.art17_1_applied),
                "art17_2": int(f.art17_2_applied),
                "art59": int(f.art59_applied),
                "recidivism": int(f.recidivism),
                "self_surrender": int(f.self_surrender),
                "can_convert_to_fine": int(f.can_convert_to_fine),
                "n_behaviors": len(f.behaviors),
                "n_drug_levels": len(f.drug_levels),
                "n_sentence_counts": f.n_sentence_counts,
                "is_aggregate_sentence": int(f.is_aggregate_sentence),
                "is_attempt": int(f.is_attempt),
                "max_drug_weight_g": wt if wt is not None else -1.0,
                "has_drug_weight": int(wt is not None),
                "log_drug_weight": np.log1p(wt) if wt is not None else 0.0,
                "sentence_months": f.sentence_months,
                # Bookkeeping columns (underscore prefix → excluded from features).
                "_jdate": rec.jdate,
                "_n_def": f.n_defendants,
                "_summary": int("簡" in rec.jcase),
                "_conv_beh": ",".join(f.convicted_behaviors),   # rule-clip lookup
                "_conv_lv": ",".join(map(str, f.convicted_drug_levels)),
                # Probation label (binary target for the §74 classifier head).
                # Underscore-prefixed → never enters X (leakage check).
                "_probation": int(f.probation_granted),
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
            for c in COURTS:
                row[f"court_{c}"] = int(court == c)
            rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in ("jid", "sentence_months") and not c.startswith("_")]


# ---------------------------------------------------------------------------
# Model + per-fold evaluation
# ---------------------------------------------------------------------------

def _make_model(seed: int, rounds: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=rounds,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        early_stopping_rounds=30,
        eval_metric="mae",
    )


# XGBoost 2.x quantile regression. ``quantile_alpha`` is the target quantile;
# ``reg:quantileerror`` minimises the pinball loss at that level. We keep the
# rest of the hyperparameters in sync with the squared-error median model so
# any quantile-vs-median gap is attributable to the loss, not capacity.
def _make_quantile_model(alpha: float, seed: int, rounds: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=rounds,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=seed,
        early_stopping_rounds=30,
        eval_metric="mae",
        tree_method="hist",
    )


def _make_classifier(seed: int, rounds: int, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=rounds,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=seed,
        early_stopping_rounds=30,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss at level ``alpha``."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def _fit(model, X_tr, y_tr, val_frac: float = 0.15):
    """Fit with early stopping on the *tail* of the training rows.

    In walk-forward mode X_tr is already time-ordered, so the tail is the most
    recent (still-in-the-past) slice — a legitimate validation set. For the
    holdout mode the order doesn't matter. Works for any XGBoost estimator
    (regressor or classifier) that accepts ``eval_set``.
    """
    n = len(X_tr)
    if n < 50:
        model.set_params(early_stopping_rounds=None)
        model.fit(X_tr, y_tr, verbose=False)
        return model
    cut = max(1, int(n * (1.0 - val_frac)))
    model.fit(X_tr[:cut], y_tr[:cut], eval_set=[(X_tr[cut:], y_tr[cut:])], verbose=False)
    return model


def _calibrate_quantile(model, X_tr: np.ndarray, y_tr: np.ndarray,
                        alpha: float, val_frac: float = 0.15) -> float:
    """Conformal marginal shift δ_α: empirical α-quantile of val-tail residuals.

    The quantile head trained with ``reg:quantileerror`` targets the conditional
    α-quantile but XGBoost regularisation pulls predictions toward the centre,
    so empirical coverage is consistently narrower than nominal. Shifting raw
    predictions by ``δ_α = quantile(y_val − ŷ_val, α)`` enforces exactly α
    coverage on the val tail; on exchangeable test data the shift transfers
    approximately. Uses the same tail slice as early stopping — small in-sample
    optimism, but bounded by the val_frac size.
    """
    n = len(X_tr)
    if n < 50:
        return 0.0
    cut = max(1, int(n * (1.0 - val_frac)))
    pred_val = np.asarray(model.predict(X_tr[cut:]), dtype=float)
    residuals = y_tr[cut:].astype(float) - pred_val
    return float(np.quantile(residuals, alpha))


def _constraints_for(df_rows: pd.DataFrame) -> list:
    out = []
    for _, row in df_rows.iterrows():
        # 數罪併罰: the label is an 應執行 aggregate (or 主文 has ≥2 有期徒刑
        # counts) — a single-count statutory range can't bound it.
        if row["is_aggregate_sentence"] or row["n_sentence_counts"] > 1:
            out.append(aggregate_only_constraint())
            continue
        behaviors = {b for b in row["_conv_beh"].split(",") if b}
        levels = {int(lv) for lv in row["_conv_lv"].split(",") if lv}
        out.append(binding_constraint(
            behaviors, levels,
            art17_1=bool(row["art17_1"]), art17_2=bool(row["art17_2"]),
            art59=bool(row["art59"]), attempt=bool(row["is_attempt"]),
            self_surrender=bool(row["self_surrender"]),
            recidivism=bool(row["recidivism"]), summary=bool(row["_summary"]),
        ))
    return out


def _hit_rate(y_true, y_pred, tol: float) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)) <= tol))


def _clip_vector(pred_raw: np.ndarray, constraints: list) -> tuple[np.ndarray, int]:
    """Apply layer-4 statutory clipping; returns (clipped, n_changed)."""
    out = pred_raw.copy()
    n = 0
    for i, c in enumerate(constraints):
        if c is None:
            continue
        cv = clip_prediction(float(pred_raw[i]), c)
        if abs(cv - float(pred_raw[i])) > 0.01:
            n += 1
        out[i] = cv
    return out, n


def evaluate_fold(
    df: pd.DataFrame, feat_cols: list[str],
    tr_idx: np.ndarray, te_idx: np.ndarray,
    *, rule_clip: bool, seed: int, rounds: int, calibrate: bool = True,
) -> dict:
    """Train on tr_idx, score on te_idx. Returns a metrics dict.

    Trains four heads on the same feature matrix:
      - sentence_regressor (squarederror, the existing median model)
      - p25 / p75 quantile regressors (``reg:quantileerror``)
      - probation classifier (XGBClassifier on ``_probation``)
    """
    X = df[feat_cols].astype(float).values
    y = df["sentence_months"].astype(float).values
    yp = df["_probation"].astype(int).values

    median_model = _fit(_make_model(seed, rounds), X[tr_idx], y[tr_idx])
    q25_model = _fit(_make_quantile_model(0.25, seed, rounds), X[tr_idx], y[tr_idx])
    q75_model = _fit(_make_quantile_model(0.75, seed, rounds), X[tr_idx], y[tr_idx])

    delta25 = _calibrate_quantile(q25_model, X[tr_idx], y[tr_idx], 0.25) if calibrate else 0.0
    delta75 = _calibrate_quantile(q75_model, X[tr_idx], y[tr_idx], 0.75) if calibrate else 0.0

    y_te = y[te_idx]
    X_te = X[te_idx]
    p50_raw = np.asarray(median_model.predict(X_te), dtype=float)
    p25_raw = np.asarray(q25_model.predict(X_te), dtype=float) + delta25
    p75_raw = np.asarray(q75_model.predict(X_te), dtype=float) + delta75

    constraints = _constraints_for(df.iloc[te_idx])
    rate_gt, n_gt, n_chk = violation_rate(list(zip(y_te, constraints)))
    rate_raw, n_rawv, _ = violation_rate(list(zip(p50_raw, constraints)))

    if rule_clip:
        p50, n_clipped = _clip_vector(p50_raw, constraints)
        p25, _ = _clip_vector(p25_raw, constraints)
        p75, _ = _clip_vector(p75_raw, constraints)
        # Re-impose monotonicity after independent clipping (mirror of
        # models.constrain_sentence's post-clip pass).
        p50 = np.maximum(p50, p25)
        p75 = np.maximum(p75, p50)
    else:
        p50, p25, p75 = p50_raw, p25_raw, p75_raw
        n_clipped = 0
    rate_clip, n_clipv, _ = violation_rate(list(zip(p50, constraints)))

    # Quantile-crossing diagnostics — count rows where raw p25 > p50 > p75.
    n_cross_raw = int(np.sum((p25_raw > p50_raw + 1e-6)
                              | (p50_raw > p75_raw + 1e-6)))

    coverage = float(np.mean((y_te >= p25) & (y_te <= p75)))

    # Probation classifier — same feature matrix, separate model. Drop the
    # leakage-suspect "_probation" column from X by construction (feat_cols
    # already excludes underscore-prefixed columns).
    #
    # scale_pos_weight is sqrt of the imbalance — full inverse-ratio weighting
    # pushes nearly all predictions above 0.5, sqrt is the usual compromise
    # that preserves probability ordering while still giving positives signal.
    # Threshold is then tuned on the val tail by F1-max (default 0.5 is wrong
    # for any weighted classifier on a 1% positive class).
    prob_metrics: dict = {"trained": False}
    yp_tr = yp[tr_idx]
    if int(yp_tr.sum()) >= 2 and int(yp_tr.sum()) < len(yp_tr):
        pos_w = float(np.sqrt(max(1.0, (len(yp_tr) - yp_tr.sum()) / max(1, yp_tr.sum()))))
        clf = _fit(_make_classifier(seed, rounds, scale_pos_weight=pos_w),
                   X[tr_idx], yp_tr)
        n_tr = len(tr_idx)
        cut = max(1, int(n_tr * (1.0 - 0.15)))
        prob_val = clf.predict_proba(X[tr_idx][cut:])[:, 1]
        y_val = yp_tr[cut:]
        threshold = 0.5
        if y_val.sum() >= 2:
            best_f1, best_t = -1.0, 0.5
            for t in np.linspace(0.05, 0.95, 19):
                pred_v = (prob_val >= t).astype(int)
                p = precision_score(y_val, pred_v, zero_division=0.0)
                r = recall_score(y_val, pred_v, zero_division=0.0)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            threshold = best_t
        yp_te = yp[te_idx]
        prob_pred = clf.predict_proba(X_te)[:, 1]
        yp_hat = (prob_pred >= threshold).astype(int)
        prob_metrics = {
            "trained": True,
            "model": clf,
            "y_true": yp_te,
            "y_prob": prob_pred,
            "threshold": threshold,
            "accuracy": float(np.mean(yp_hat == yp_te)),
            "precision": float(precision_score(yp_te, yp_hat, zero_division=0.0)),
            "recall": float(recall_score(yp_te, yp_hat, zero_division=0.0)),
            "pr_auc": float(average_precision_score(yp_te, prob_pred))
            if yp_te.sum() > 0 else float("nan"),
            "base_rate": float(yp_te.mean()),
        }

    return {
        "model": median_model,
        "q25_model": q25_model, "q75_model": q75_model,
        "delta25": delta25, "delta75": delta75,
        "feat_cols": feat_cols,
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "te_idx": np.asarray(te_idx),
        "y_true": y_te, "y_pred": p50, "y_pred_raw": p50_raw,
        "p25": p25, "p75": p75, "p25_raw": p25_raw, "p75_raw": p75_raw,
        "mae": mean_absolute_error(y_te, p50),
        "mae_raw": mean_absolute_error(y_te, p50_raw),
        "rmse": float(np.sqrt(mean_squared_error(y_te, p50))),
        "r2": r2_score(y_te, p50) if len(te_idx) > 1 else float("nan"),
        "hit3": _hit_rate(y_te, p50, 3.0),
        "hit6": _hit_rate(y_te, p50, 6.0),
        "pinball_p25": _pinball(y_te, p25, 0.25),
        "pinball_p50": _pinball(y_te, p50, 0.50),
        "pinball_p75": _pinball(y_te, p75, 0.75),
        "coverage_50": coverage,
        "n_cross_raw": n_cross_raw,
        "n_clipped": n_clipped, "n_constrained": n_chk,
        "viol_gt": rate_gt, "n_viol_gt": n_gt,
        "viol_raw": rate_raw, "n_viol_raw": n_rawv,
        "viol_clip": rate_clip, "n_viol_clip": n_clipv,
        "median_baseline_mae": mean_absolute_error(
            y_te, np.full_like(y_te, np.median(y[tr_idx]))),
        "probation": prob_metrics,
    }


def _print_fold(tag: str, m: dict, dates: tuple[str, str] | None = None) -> None:
    span = f"  test {dates[0]}–{dates[1]}" if dates else ""
    print(f"[{tag}] train={m['n_train']:>5}  test={m['n_test']:>4}{span}")
    print(f"        MAE={m['mae']:5.2f} (raw {m['mae_raw']:5.2f})  "
          f"RMSE={m['rmse']:6.2f}  R^2={m['r2']:5.3f}  "
          f"±3mo={m['hit3']:5.1%}  ±6mo={m['hit6']:5.1%}  "
          f"(median-base MAE={m['median_baseline_mae']:.2f})")
    print(f"        quantiles: pinball p25={m['pinball_p25']:.2f}  "
          f"p50={m['pinball_p50']:.2f}  p75={m['pinball_p75']:.2f}  "
          f"coverage[p25,p75]={m['coverage_50']:.1%}  "
          f"raw-crossings={m['n_cross_raw']}/{m['n_test']}  "
          f"δ25={m.get('delta25', 0.0):+.2f}  δ75={m.get('delta75', 0.0):+.2f}")
    print(f"        越界率: gt={m['viol_gt']:.2%}({m['n_viol_gt']}) "
          f"raw={m['viol_raw']:.2%}({m['n_viol_raw']}) "
          f"clipped={m['viol_clip']:.2%}({m['n_viol_clip']}) "
          f"[{m['n_clipped']} clipped / {m['n_constrained']} mapped]"
          f"{'  !! LAYER-4 FAILURE' if m['n_viol_clip'] else ''}")
    pm = m.get("probation", {})
    if pm.get("trained"):
        print(f"        緩刑分類: base={pm['base_rate']:.1%}  "
              f"acc={pm['accuracy']:.1%}  P={pm['precision']:.1%}  "
              f"R={pm['recall']:.1%}  PR-AUC={pm['pr_auc']:.3f}  "
              f"thr={pm.get('threshold', 0.5):.2f}")


def walk_forward(df: pd.DataFrame, feat_cols: list[str], *,
                 folds: int, rule_clip: bool, seed: int, rounds: int,
                 calibrate: bool = True) -> list[dict]:
    """Expanding-window temporal CV.

    Time-ordered rows are split into ``folds + 1`` contiguous bins. Bin 0 is
    the initial train-only seed; fold k (1..folds) trains on bins ``0..k-1``
    and tests on bin ``k``. So the test set is always strictly later than every
    training row.
    """
    order = df["_jdate"].astype(str).values.argsort(kind="stable")
    n = len(order)
    edges = np.linspace(0, n, folds + 2, dtype=int)
    results = []
    for k in range(folds):
        tr_idx = order[: edges[k + 1]]
        te_idx = order[edges[k + 1]: edges[k + 2]]
        if len(te_idx) == 0 or len(tr_idx) < 30:
            continue
        m = evaluate_fold(df, feat_cols, tr_idx, te_idx,
                          rule_clip=rule_clip, seed=seed, rounds=rounds,
                          calibrate=calibrate)
        d0 = str(df.iloc[te_idx]["_jdate"].min())
        d1 = str(df.iloc[te_idx]["_jdate"].max())
        _print_fold(f"fold {k + 1}/{folds}", m, (d0, d1))
        results.append(m)
    return results


def _print_pooled(results: list[dict]) -> None:
    if not results:
        print("\n(no folds evaluated)")
        return
    y_true = np.concatenate([r["y_true"] for r in results])
    y_pred = np.concatenate([r["y_pred"] for r in results])
    y_praw = np.concatenate([r["y_pred_raw"] for r in results])
    p25 = np.concatenate([r["p25"] for r in results])
    p75 = np.concatenate([r["p75"] for r in results])
    n_viol_gt = sum(r["n_viol_gt"] for r in results)
    n_viol_clip = sum(r["n_viol_clip"] for r in results)
    n_mapped = sum(r["n_constrained"] for r in results)
    n_cross_raw = sum(r["n_cross_raw"] for r in results)
    print("\n── Pooled across folds ──────────────────────────────────────")
    print(f"  n            = {len(y_true)}  ({len(results)} folds)")
    print(f"  MAE          = {mean_absolute_error(y_true, y_pred):.2f} months "
          f"(raw {mean_absolute_error(y_true, y_praw):.2f})")
    print(f"  RMSE         = {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} months")
    print(f"  R^2          = {r2_score(y_true, y_pred):.3f}")
    print(f"  ±3-month hit = {_hit_rate(y_true, y_pred, 3.0):.1%}")
    print(f"  ±6-month hit = {_hit_rate(y_true, y_pred, 6.0):.1%}")
    print(f"  pinball p25  = {_pinball(y_true, p25, 0.25):.2f} months")
    print(f"  pinball p50  = {_pinball(y_true, y_pred, 0.50):.2f} months")
    print(f"  pinball p75  = {_pinball(y_true, p75, 0.75):.2f} months")
    cov = float(np.mean((y_true >= p25) & (y_true <= p75)))
    print(f"  coverage [p25, p75] = {cov:.1%}  (target ~ 50%)")
    print(f"  raw quantile-crossings = {n_cross_raw}/{len(y_true)}  "
          f"({n_cross_raw / max(1, len(y_true)):.2%}; post-clip 0 by construction)")
    print(f"  越界率 (ground-truth labels) = {n_viol_gt}/{n_mapped} "
          f"= {n_viol_gt / max(1, n_mapped):.2%}  (data-quality canary)")
    print(f"  越界率 (rule-clipped preds)  = {n_viol_clip}/{n_mapped} "
          f"= {n_viol_clip / max(1, n_mapped):.2%}  "
          f"{'OK' if n_viol_clip == 0 else '!! LAYER-4 FAILURE'}")
    abs_err = np.abs(y_pred - y_true)
    print("  residual |pred-y| percentiles: " + "  ".join(
        f"p{p}={np.percentile(abs_err, p):.1f}" for p in (50, 75, 90, 95, 99)))

    # Probation classifier pooled metrics
    prob_results = [r for r in results if r.get("probation", {}).get("trained")]
    if prob_results:
        py_true = np.concatenate([r["probation"]["y_true"] for r in prob_results])
        py_prob = np.concatenate([r["probation"]["y_prob"] for r in prob_results])
        # Apply each fold's own tuned threshold to its own rows (pooling raw
        # probs across folds with a single global threshold is meaningless
        # when each fold trains a separately weighted model).
        py_hat = np.concatenate([
            (r["probation"]["y_prob"] >= r["probation"]["threshold"]).astype(int)
            for r in prob_results
        ])
        print(f"\n── Pooled probation classifier ──────────────────────────────")
        print(f"  n            = {len(py_true)}")
        print(f"  base rate    = {py_true.mean():.2%}  (positive label rate)")
        print(f"  accuracy     = {float(np.mean(py_hat == py_true)):.2%}")
        print(f"  precision    = {precision_score(py_true, py_hat, zero_division=0.0):.2%}  (at per-fold F1-max threshold)")
        print(f"  recall       = {recall_score(py_true, py_hat, zero_division=0.0):.2%}")
        print(f"  PR-AUC       = {average_precision_score(py_true, py_prob):.3f}  "
              f"(lift {average_precision_score(py_true, py_prob) / max(py_true.mean(), 1e-9):.1f}x base)")


_BEHAVIOR_PRIORITY = ["運輸", "製造", "販賣", "意圖販賣而持有", "轉讓", "持有", "施用"]


def _primary_behavior(row) -> str:
    for b in _BEHAVIOR_PRIORITY:
        if row.get(f"b_{b}", 0):
            return b
    return "(其他)"


def _print_by_court(df: pd.DataFrame, results: list[dict]) -> None:
    """Per-court MAE across pooled walk-forward test rows.

    Lets us see whether a 5-court model is still as accurate on its
    home-court rows (KL) as a KL-only model would be, vs. how it does on
    the other 4 courts it now has to span.
    """
    if not results or "_court" not in df.columns:
        return
    buckets: dict[str, list[tuple[float, float]]] = {}
    for r in results:
        te = df.iloc[r["te_idx"]]
        for (_, row), yt, yp in zip(te.iterrows(), r["y_true"], r["y_pred"]):
            buckets.setdefault(str(row["_court"]), []).append((float(yt), float(yp)))
    if len(buckets) <= 1:
        return
    print("\nPer-court MAE (pooled across folds):")
    print(f"  {'court':<8} {'n':>6}  {'MAE':>6}  {'median |err|':>12}")
    for k in sorted(buckets, key=lambda c: -len(buckets[c])):
        pairs = buckets[k]
        yt = np.array([p[0] for p in pairs])
        yp = np.array([p[1] for p in pairs])
        mae = mean_absolute_error(yt, yp)
        med = float(np.median(np.abs(yp - yt)))
        print(f"  {k:<8} {len(pairs):>6}  {mae:>6.2f}  {med:>12.2f}")


def _print_by_behavior(df: pd.DataFrame, results: list[dict]) -> None:
    """Per-primary-behavior MAE across pooled walk-forward test rows."""
    if not results:
        return
    buckets: dict[str, list[tuple[float, float]]] = {}
    for r in results:
        te = df.iloc[r["te_idx"]]
        for (_, row), yt, yp in zip(te.iterrows(), r["y_true"], r["y_pred"]):
            b = _primary_behavior(row)
            buckets.setdefault(b, []).append((float(yt), float(yp)))
    print("\nPer-primary-behavior MAE (pooled across folds):")
    print(f"  {'behavior':<20} {'n':>5}  {'MAE':>6}  {'median |err|':>12}")
    for b in _BEHAVIOR_PRIORITY + ["(其他)"]:
        if b not in buckets:
            continue
        pairs = buckets[b]
        n = len(pairs)
        yt = np.array([p[0] for p in pairs])
        yp = np.array([p[1] for p in pairs])
        mae = mean_absolute_error(yt, yp)
        med = float(np.median(np.abs(yp - yt)))
        print(f"  {b:<20} {n:>5}  {mae:>6.2f}  {med:>12.2f}")


def _print_importance(model, feat_cols: list[str], top: int = 15,
                       label: str = "last/full-data model") -> None:
    print(f"\nTop {top} features by gain importance ({label}):")
    importance = model.get_booster().get_score(importance_type="gain")
    name_map = {f"f{i}": name for i, name in enumerate(feat_cols)}
    for f, score in sorted(importance.items(), key=lambda x: -x[1])[:top]:
        print(f"  {name_map.get(f, f):28} {score:>10.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/filtered/north5_drug_all.jsonl"))
    ap.add_argument("--art57", type=str,
                    default="data/processed/art57_factors.jsonl",
                    help="JSONL of §57 directions (set to '' to disable)")
    ap.add_argument("--no-art57", action="store_true",
                    help="disable §57 features for comparison")
    ap.add_argument("--no-rule-clip", action="store_true",
                    help="skip layer-4 statutory clipping (ablation only — "
                         "production must keep it on)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip conformal δ-shift on p25/p75 (ablation; "
                         "with this flag, [p25, p75] coverage is the raw "
                         "quantile-head coverage, typically ~45%%)")
    ap.add_argument("--keep-multi-defendant", action="store_true",
                    help="include judgments whose 主文 names >1 defendant "
                         "(noisy per-defendant labels)")
    ap.add_argument("--folds", type=int, default=5,
                    help="walk-forward temporal CV folds (expanding window)")
    ap.add_argument("--holdout", action="store_true",
                    help="instead of walk-forward, do a single random 80/20 "
                         "split (quick sanity check — NOT a valid headline metric)")
    ap.add_argument("--save", type=Path, default=None,
                    help="after evaluation, retrain on ALL rows and dump a ModelBundle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=500,
                    help="max boosting rounds (early-stopping enabled)")
    args = ap.parse_args()

    art57_path = None if args.no_art57 or not args.art57 else Path(args.art57)
    df = build_dataframe(args.inp, art57_path)
    n_all = len(df)
    n_multi = int((df["_n_def"] > 1).sum())
    if not args.keep_multi_defendant:
        df = df[df["_n_def"] == 1].reset_index(drop=True)
    rule_clip = not args.no_rule_clip

    print(f"loaded {n_all} cases with sentence_months "
          f"({n_multi} multi-defendant {'kept' if args.keep_multi_defendant else 'dropped'}) "
          f"→ {len(df)} rows")
    print(f"  sentence_months: median={df.sentence_months.median():.0f}  "
          f"mean={df.sentence_months.mean():.1f}  max={df.sentence_months.max()}")
    print(f"  date span: {df['_jdate'].min()} – {df['_jdate'].max()}")
    feat_cols = feature_columns(df)
    print(f"  features ({len(feat_cols)}): {feat_cols}")
    print(f"  rule clipping: {'ON' if rule_clip else 'OFF'}")
    print()

    calibrate = not args.no_calibrate
    if args.holdout:
        print("=== Holdout (random 80/20 — sanity check only) ===")
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2, random_state=args.seed)
        m = evaluate_fold(df, feat_cols, tr, te,
                          rule_clip=rule_clip, seed=args.seed, rounds=args.rounds,
                          calibrate=calibrate)
        _print_fold("holdout", m)
        last_model = m["model"]
        results = [m]
    else:
        print(f"=== Walk-forward temporal CV ({args.folds} folds, expanding window) ===")
        print(f"  quantile calibration: {'ON (conformal δ-shift)' if calibrate else 'OFF'}")
        results = walk_forward(df, feat_cols, folds=args.folds,
                               rule_clip=rule_clip, seed=args.seed, rounds=args.rounds,
                               calibrate=calibrate)
        _print_pooled(results)
        last_model = results[-1]["model"] if results else None

    if not args.holdout:
        _print_by_court(df, results)
        _print_by_behavior(df, results)

    if last_model is not None:
        _print_importance(last_model, feat_cols, label="median regressor (last fold)")
    last_clf = (results[-1].get("probation", {}).get("model")
                if results else None)
    if last_clf is not None:
        _print_importance(last_clf, feat_cols, label="probation classifier (last fold)")

    if args.save:
        import pickle
        # Retrain every head on the full kept dataset for the production bundle.
        X = df[feat_cols].astype(float).values
        y = df["sentence_months"].astype(float).values
        yp = df["_probation"].astype(int).values
        full_p50 = _fit(_make_model(args.seed, args.rounds), X, y)
        full_p25 = _fit(_make_quantile_model(0.25, args.seed, args.rounds), X, y)
        full_p75 = _fit(_make_quantile_model(0.75, args.seed, args.rounds), X, y)
        full_d25 = _calibrate_quantile(full_p25, X, y, 0.25) if calibrate else 0.0
        full_d75 = _calibrate_quantile(full_p75, X, y, 0.75) if calibrate else 0.0

        full_clf = None
        full_threshold = 0.5
        if int(yp.sum()) >= 2 and int(yp.sum()) < len(yp):
            pos_w = float(np.sqrt(max(1.0, (len(yp) - yp.sum()) / max(1, yp.sum()))))
            full_clf = _fit(_make_classifier(args.seed, args.rounds,
                                              scale_pos_weight=pos_w), X, yp)
            n_full = len(X)
            cut_f = max(1, int(n_full * (1.0 - 0.15)))
            prob_v = full_clf.predict_proba(X[cut_f:])[:, 1]
            y_v = yp[cut_f:]
            if y_v.sum() >= 2:
                best_f1, best_t = -1.0, 0.5
                for t in np.linspace(0.05, 0.95, 19):
                    pv = (prob_v >= t).astype(int)
                    p = precision_score(y_v, pv, zero_division=0.0)
                    r = recall_score(y_v, pv, zero_division=0.0)
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1, best_t = f1, float(t)
                full_threshold = best_t

        pooled_mae = (mean_absolute_error(
            np.concatenate([r["y_true"] for r in results]),
            np.concatenate([r["y_pred"] for r in results])) if results else None)
        pooled_pr_auc = None
        prob_results = [r for r in results if r.get("probation", {}).get("trained")]
        if prob_results:
            py_true = np.concatenate([r["probation"]["y_true"] for r in prob_results])
            py_prob = np.concatenate([r["probation"]["y_prob"] for r in prob_results])
            pooled_pr_auc = float(average_precision_score(py_true, py_prob))

        bundle = ModelBundle(
            sentence_regressor=full_p50,
            sentence_quantile_p25=full_p25,
            sentence_quantile_p75=full_p75,
            probation_classifier=full_clf,
            feature_names=tuple(feat_cols),
            metadata={
                "trained_on": str(args.inp),
                "n_rows": int(len(df)),
                "probation_base_rate": float(yp.mean()),
                "multi_defendant_dropped": not args.keep_multi_defendant,
                "rule_clip": rule_clip,
                "art57": art57_path is not None,
                "eval": "holdout" if args.holdout else f"walkforward-{args.folds}",
                "eval_mae": float(pooled_mae) if pooled_mae is not None else None,
                "eval_probation_pr_auc": pooled_pr_auc,
                "delta25": full_d25,
                "delta75": full_d75,
                "calibrated": calibrate,
                "probation_threshold": full_threshold,
            },
        )
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("wb") as fh:
            pickle.dump(bundle, fh)
        print(f"\nSaved ModelBundle (retrained on all {len(df)} rows) → {args.save}")
        print(f"  heads: p50={full_p50 is not None}  p25={full_p25 is not None}  "
              f"p75={full_p75 is not None}  probation={full_clf is not None}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
