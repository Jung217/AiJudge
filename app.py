"""FastAPI service serving the trained ModelBundle.

Endpoints:
  GET  /health   liveness + bundle-loaded flag
  GET  /version  bundle metadata (feature count, eval MAE, calibration δ etc.)
  POST /predict  case features → {p25/p50/p75, probation_prob, constraint}
                 with the statutory disclaimer attached to every response.

Run:
  uvicorn app:app --reload --port 8000

The model bundle path can be overridden with the AIJUDGE_BUNDLE env var
(default: data/processed/baseline_model.pkl).
"""
from __future__ import annotations

import math
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import ModelBundle, predict_with_constraints
from rules import aggregate_only_constraint, binding_constraint


BUNDLE_PATH = Path(os.environ.get("AIJUDGE_BUNDLE",
                                    "data/processed/baseline_model.pkl"))

DISCLAIMER = (
    "本模型僅為臺灣基隆地方法院毒品案件「量刑趨勢」的統計推估,「不可」"
    "作為法律建議、判決依據或律師代理之替代。重大刑案誤差仍大;請務必"
    "諮詢專業律師。本服務不支援法官使用以避免影響司法獨立。"
)

BEHAVIORS = ["施用", "持有", "販賣", "運輸", "製造", "轉讓", "意圖販賣而持有"]
DRUG_LEVELS = [1, 2, 3, 4]
ART57_FACTORS = ["motive", "provocation", "means", "life_status", "character",
                 "intellect", "relation_victim", "duty_breach", "harm",
                 "post_attitude"]


_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {BUNDLE_PATH}. Train one with "
            f"`python scripts/04_train_baseline.py --save {BUNDLE_PATH}`."
        )
    with BUNDLE_PATH.open("rb") as fh:
        _state["bundle"] = pickle.load(fh)
    yield
    _state.clear()


app = FastAPI(
    title="AiJudge sentencing predictor",
    description=DISCLAIMER,
    version="0.1.0",
    lifespan=lifespan,
)


class CaseInput(BaseModel):
    """單一被告、單一案件的結構化輸入。

    一份判決如有多被告或數罪併罰應執行刑,呼叫端應自行拆成多筆。
    """
    behaviors: list[str] = Field(
        ..., min_length=1,
        description="主文與事實中出現的行為態樣 (e.g. ['施用', '持有'])")
    drug_levels: list[int] = Field(
        ..., min_length=1,
        description="毒品級數 1-4")
    convicted_behaviors: Optional[list[str]] = Field(
        None,
        description="實際被定罪的行為(供 rule-clip 用);None 則回退到 behaviors")
    convicted_drug_levels: Optional[list[int]] = Field(
        None, description="實際被定罪的毒品級數;None 則回退到 drug_levels")
    art17_1: bool = Field(False, description="毒品條例 §17Ⅰ(供出來源因而查獲)")
    art17_2: bool = Field(False, description="毒品條例 §17Ⅱ(偵審均自白)")
    art59: bool = Field(False, description="刑法 §59 酌減")
    self_surrender: bool = Field(False, description="刑法 §62 自首")
    recidivism: bool = Field(False, description="刑法 §47 累犯")
    is_attempt: bool = Field(False, description="刑法 §25Ⅱ 未遂")
    can_convert_to_fine: bool = Field(False, description="得易科罰金")
    n_sentence_counts: int = Field(1, ge=1, description="主文宣告刑筆數")
    is_aggregate_sentence: bool = Field(
        False, description="標的刑期為應執行刑(數罪併罰)而非單一宣告刑")
    max_drug_weight_g: Optional[float] = Field(
        None, description="純質淨重最大值(克);未認定請填 null")
    jyear: int = Field(113, ge=80, le=200, description="民國年(JYEAR)")
    jcase: str = Field("訴", description="案件字別 (訴/簡/易/原訴/...)")


class ConstraintOut(BaseModel):
    min_months: float
    max_months: float
    includes_life: bool
    includes_capital: bool


class PredictionOut(BaseModel):
    p25_months: Optional[float] = None
    p50_months: float
    p75_months: Optional[float] = None
    raw: dict[str, float]
    rule_applied: bool
    clipped: bool
    probation_prob: Optional[float] = None
    probation_predicted: Optional[bool] = None
    probation_threshold: Optional[float] = None


class PredictResponse(BaseModel):
    prediction: PredictionOut
    constraint: Optional[ConstraintOut]
    disclaimer: str


def _build_features(c: CaseInput) -> dict[str, float]:
    """Map a CaseInput to the bundle's feature-name → value dict.

    Any name the trained bundle expects but we haven't been told about
    defaults to 0 (e.g. §57 factors when the LLM extractor hasn't run).
    """
    bundle: ModelBundle = _state["bundle"]
    wt = c.max_drug_weight_g
    feats: dict[str, float] = {
        "jyear": c.jyear,
        "art17_1": int(c.art17_1),
        "art17_2": int(c.art17_2),
        "art59": int(c.art59),
        "recidivism": int(c.recidivism),
        "self_surrender": int(c.self_surrender),
        "can_convert_to_fine": int(c.can_convert_to_fine),
        "n_behaviors": len(set(c.behaviors)),
        "n_drug_levels": len(set(c.drug_levels)),
        "n_sentence_counts": c.n_sentence_counts,
        "is_aggregate_sentence": int(c.is_aggregate_sentence),
        "is_attempt": int(c.is_attempt),
        "max_drug_weight_g": wt if wt is not None else -1.0,
        "has_drug_weight": int(wt is not None),
        "log_drug_weight": math.log1p(wt) if wt is not None else 0.0,
    }
    for f in ART57_FACTORS:
        feats[f"a57_{f}_mit"] = 0
        feats[f"a57_{f}_agg"] = 0
        feats[f"a57_{f}_neu"] = 0
    for b in BEHAVIORS:
        feats[f"b_{b}"] = int(b in c.behaviors)
    for lv in DRUG_LEVELS:
        feats[f"lv_{lv}"] = int(lv in c.drug_levels)
    for name in bundle.feature_names:
        feats.setdefault(name, 0)
    return feats


def _constraint_for(c: CaseInput):
    if c.is_aggregate_sentence or c.n_sentence_counts > 1:
        return aggregate_only_constraint()
    cb = c.convicted_behaviors or c.behaviors
    cl = c.convicted_drug_levels or c.drug_levels
    return binding_constraint(
        set(cb), set(cl),
        art17_1=c.art17_1, art17_2=c.art17_2, art59=c.art59,
        attempt=c.is_attempt, self_surrender=c.self_surrender,
        recidivism=c.recidivism, summary="簡" in c.jcase,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "bundle_loaded": "bundle" in _state}


@app.get("/version")
def version() -> dict:
    if "bundle" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    b: ModelBundle = _state["bundle"]
    return {
        "feature_count": len(b.feature_names),
        "metadata": b.metadata or {},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(c: CaseInput) -> PredictResponse:
    if "bundle" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    feats = _build_features(c)
    constraint = _constraint_for(c)
    result = predict_with_constraints(_state["bundle"], feats, constraint)
    return PredictResponse(
        prediction=PredictionOut(
            p25_months=result.get("p25_months"),
            p50_months=result["p50_months"],
            p75_months=result.get("p75_months"),
            raw=result["raw"],
            rule_applied=result["rule_applied"],
            clipped=result["clipped"],
            probation_prob=result.get("probation_prob"),
            probation_predicted=result.get("probation_predicted"),
            probation_threshold=result.get("probation_threshold"),
        ),
        constraint=ConstraintOut(
            min_months=constraint.min_months,
            max_months=constraint.max_months,
            includes_life=constraint.includes_life,
            includes_capital=constraint.includes_capital,
        ) if constraint else None,
        disclaimer=DISCLAIMER,
    )
