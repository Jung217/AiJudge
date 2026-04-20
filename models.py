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

from rules import SentencingConstraint, clip_prediction


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


def predict_with_constraints(
    bundle: ModelBundle,
    features: dict,
    constraint: SentencingConstraint,
) -> dict:
    """Predict sentence and clip to statutory range.

    Output format:
        {
          "p25_months": ...,
          "p50_months": ...,
          "p75_months": ...,
          "probation_prob": ...,
          "behavior": "販賣",
        }

    TODO phase 3.
    """
    raise NotImplementedError("Phase 3")


def explain_prediction(bundle: ModelBundle, features: dict) -> dict:
    """SHAP-based per-feature contribution. TODO phase 4."""
    raise NotImplementedError("Phase 4")
