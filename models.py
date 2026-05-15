"""ML models — skeleton for phase 3.

See plan.md §5.2 for the layered-model design:
    - Behavior classifier (LightGBM multi-class)
    - Sentence regressor (XGBoost + Quantile Regression)
    - Probation classifier (LightGBM binary)
    - Aggregate-sentence discount regressor

This module only defines the interface. Actual training code arrives when
phase 2 (feature extraction) produces enough labelled samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rules import SentencingConstraint, clip_prediction, is_within


@dataclass
class ModelBundle:
    """A trained model bundle."""
    behavior_classifier: Any = None
    sentence_regressor: Any = None              # predicts median months
    sentence_quantile_p25: Any = None
    sentence_quantile_p75: Any = None
    probation_classifier: Any = None
    merge_discount_regressor: Any = None

    feature_names: Sequence[str] = ()
    metadata: dict = None


def train_behavior_classifier(X, y, **kwargs):
    """Train LightGBM multi-class classifier on behavior labels.

    TODO phase 3 — depends on §4.1/§4.2 outputs being stable.
    """
    raise NotImplementedError("Phase 3")


def train_sentence_regressor(X, y, sample_weight=None, **kwargs):
    """Train XGBoost quantile regressor — outputs (p25, p50, p75) months.

    TODO phase 3.
    """
    raise NotImplementedError("Phase 3")


def constrain_sentence(
    raw_months: dict[str, float],
    constraint: SentencingConstraint | None,
) -> dict:
    """Clip raw quantile predictions to the statutory range (layer 4).

    ``raw_months`` maps quantile keys (e.g. ``"p25"``, ``"p50"``, ``"p75"``)
    to raw model outputs in months. When ``constraint`` is ``None`` (no entry
    in the penalty table for this behavior/level) the raw values pass through
    unchanged and ``rule_applied`` is ``False``.

    The clipped output is guaranteed to satisfy :func:`rules.is_within` for
    every quantile — i.e. the per-prediction 法定刑越界率 is 0 by construction.

    Returns ``{"p25_months", "p50_months", "p75_months", ...,
    "raw": {...}, "rule_applied": bool, "clipped": bool, "constraint": {...}}``.
    """
    out: dict = {"raw": dict(raw_months)}
    if constraint is None:
        for k, v in raw_months.items():
            out[f"{k}_months"] = float(v)
        out["rule_applied"] = False
        out["clipped"] = False
        out["constraint"] = None
        return out

    clipped_any = False
    for k, v in raw_months.items():
        cv = clip_prediction(float(v), constraint)
        if abs(cv - float(v)) > 1e-6:
            clipped_any = True
        out[f"{k}_months"] = cv
    # Keep quantiles monotone after independent clipping.
    keys = [k for k in ("p25", "p50", "p75") if f"{k}_months" in out]
    for lo, hi in zip(keys, keys[1:]):
        if out[f"{hi}_months"] < out[f"{lo}_months"]:
            out[f"{hi}_months"] = out[f"{lo}_months"]
    assert all(is_within(out[f"{k}_months"], constraint) for k in raw_months)
    out["rule_applied"] = True
    out["clipped"] = clipped_any
    out["constraint"] = {
        "min_months": constraint.min_months,
        "max_months": constraint.max_months,
        "includes_life": constraint.includes_life,
        "includes_capital": constraint.includes_capital,
    }
    return out


def predict_with_constraints(
    bundle: ModelBundle,
    features: dict,
    constraint: SentencingConstraint | None,
) -> dict:
    """Predict sentence quantiles from a trained bundle and clip to statute.

    Thin wrapper: runs the regressors in ``bundle`` over ``features`` to get
    raw ``p25/p50/p75`` months, then defers to :func:`constrain_sentence`.
    Also attaches ``behavior`` and ``probation_prob`` when those heads exist.
    """
    if bundle.sentence_regressor is None:
        raise ValueError("bundle has no sentence_regressor")
    import numpy as np

    x = np.asarray([[float(features[name]) for name in bundle.feature_names]],
                   dtype=float)
    raw = {"p50": float(bundle.sentence_regressor.predict(x)[0])}
    meta = bundle.metadata or {}
    # If the bundle was saved with a joint quantile model, both
    # sentence_quantile_p25 and sentence_quantile_p75 point at the SAME
    # XGBRegressor that returns a (1, 2) prediction matrix. Otherwise each
    # field is its own single-output model (legacy bundles).
    if meta.get("joint_quantile") and bundle.sentence_quantile_p25 is not None:
        pred = bundle.sentence_quantile_p25.predict(x)
        alphas = meta.get("quantile_alphas") or [0.25, 0.75]
        idx_25 = alphas.index(0.25) if 0.25 in alphas else 0
        idx_75 = alphas.index(0.75) if 0.75 in alphas else 1
        raw["p25"] = float(pred[0, idx_25]) + float(meta.get("delta25", 0.0))
        raw["p75"] = float(pred[0, idx_75]) + float(meta.get("delta75", 0.0))
    else:
        if bundle.sentence_quantile_p25 is not None:
            raw["p25"] = (float(bundle.sentence_quantile_p25.predict(x)[0])
                          + float(meta.get("delta25", 0.0)))
        if bundle.sentence_quantile_p75 is not None:
            raw["p75"] = (float(bundle.sentence_quantile_p75.predict(x)[0])
                          + float(meta.get("delta75", 0.0)))
    result = constrain_sentence(raw, constraint)
    if bundle.probation_classifier is not None:
        prob = float(bundle.probation_classifier.predict_proba(x)[0, 1])
        result["probation_prob"] = prob
        threshold = float(meta.get("probation_threshold", 0.5))
        result["probation_predicted"] = prob >= threshold
        result["probation_threshold"] = threshold
    if bundle.behavior_classifier is not None:
        result["behavior"] = bundle.behavior_classifier.predict(x)[0]
    return result


def explain_prediction(bundle: ModelBundle, features: dict,
                        top: int = 10) -> dict:
    """SHAP-based per-feature contribution for the p50 (median) head.

    Returns the top-``top`` features by absolute SHAP value, each item a tuple
    ``(name, value, contribution_months)``:
      - ``value`` is the input feature value
      - ``contribution_months`` is the additive shift this feature contributes
        to the predicted month-count (positive → raised the sentence)
    Also returns ``base_value`` (model expected value) and ``predicted_raw``
    so callers can verify ``base_value + sum(contributions) ≈ predicted_raw``.

    Uses ``TreeExplainer(feature_perturbation="tree_path_dependent")`` so no
    background dataset is needed.
    """
    if bundle.sentence_regressor is None:
        raise ValueError("bundle has no sentence_regressor")
    import numpy as np
    import shap

    x = np.asarray([[float(features[name]) for name in bundle.feature_names]],
                   dtype=float)
    explainer = shap.TreeExplainer(
        bundle.sentence_regressor,
        feature_perturbation="tree_path_dependent",
    )
    sv = explainer.shap_values(x)
    contribs = sv[0] if sv.ndim == 2 else sv
    base = float(explainer.expected_value
                  if np.ndim(explainer.expected_value) == 0
                  else explainer.expected_value[0])
    predicted_raw = float(bundle.sentence_regressor.predict(x)[0])

    ranked = sorted(
        enumerate(contribs),
        key=lambda kv: -abs(kv[1]),
    )[:top]
    items = [
        {
            "name": bundle.feature_names[i],
            "value": float(x[0, i]),
            "contribution_months": float(c),
        }
        for i, c in ranked
    ]
    return {
        "base_value": base,
        "predicted_raw": predicted_raw,
        "top_contributions": items,
    }
