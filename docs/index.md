---
layout: default
title: AiJudge — 基隆地方法院毒品案件量刑預測
---

# AiJudge

> 用 38 個月、1,598 件臺灣基隆地方法院毒品判決，建立量刑預測模型。
> 來源：[司法院開放資料平臺](https://opendata.judicial.gov.tw/)
> 完整代碼：[github.com/Jung217/AiJudge](https://github.com/Jung217/AiJudge)

## 核心成果

### features.py 抽取品質（100 件人工驗證）

| 欄位 | n | Precision | Recall | F1 |
|---|---|---|---|---|
| §17 Ⅰ（供出查獲）| 100 | **1.00** | 1.00 | **1.00** |
| §17 Ⅱ（偵審自白）| 100 | 1.00 | 0.88 | 0.93 |
| §59 酌減 | 100 | **1.00** | **1.00** | **1.00** |
| 累犯（§47）| 100 | 0.97 | 1.00 | 0.99 |
| 易科罰金 | 100 | 1.00 | 0.98 | 0.99 |

| 欄位 | Jaccard | 備註 |
|---|---|---|
| 行為（販賣/施用/持有 等 7 類）| 0.95 | |
| 毒品級數（一/二/三/四級）| 0.97 | |

| 數值欄位 | n | exact-match | MAE |
|---|---|---|---|
| 量刑（月）| 94 | **100%** | **0.0 月** |
| 拘役（日）| 7 | 100% | 0.0 |

### XGBoost baseline 預測 sentence_months

> n=301 holdout（80/20 split, seed 42）

| 配置 | MAE | RMSE | R² |
|---|---|---|---|
| Median 基線 | 6.57 | — | — |
| Mean 基線 | 8.88 | — | — |
| **XGBoost** | **3.09 月** | 9.18 | **0.756** |
| **+ rule-clip（法定刑度約束）** | **3.09 月** | 9.54 | 0.736 |

殘差中位數 0.8 月。模型在輕罪（施用/持有）MAE 1–2 月，重罪（販賣/運輸）資料稀疏 MAE 17–27 月。

### Top 15 features（gain importance）

```
b_運輸                7236    art59           3750    can_convert_to_fine  1177
n_drug_levels         647    b_販賣            609    lv_3                  578
b_持有                  543    lv_4              409    b_施用                 406
art17_2               406    b_意圖販賣而持有       398    art17_1               350
recidivism            334    b_製造            334    n_behaviors           332
```

毒品行為（運輸/販賣/製造）+ 減刑因子（§17/§59）+ 易科罰金 三大量刑判準佔據前 15 名，與法律實務一致。

## 資料管線

```
司法院 RAR 月檔 ──bsdtar──> JSON / 案件
  (38 個月)                  ~2.17M files
                                |
                                | filter.py (KL prefix + 毒品案 + 一審有罪)
                                v
                         1,598 件
                         基隆毒品判決
                                |
                                | features.py
                                v
                         結構化特徵
                         - 行為(7類)
                         - 毒品級數
                         - §17/§59
                         - 純質淨重
                         - §57 因子 (LLM)
                         - 量刑(月/日)
                                |
                                | 04_train_baseline.py
                                v
                         XGBoost +
                         rules.py 約束
```

## 技術細節

### features.py 量刑因子偵測

§17 / §59 都採用 **citation-anchored + sentence-scoped + rejection-aware** 邏輯：

1. 找每個法條引文位置（如 `毒品危害防制條例第17條第2項`）
2. 跳過樣板引文（recital：`按⋯犯第4條⋯定有明文`）
3. 收斂窗口到所在**句子**（避免跨句吸到鄰近條文的「不適用」）
4. 要求應用動詞（減輕 / 遞減 / 酌減）出現
5. 拒絕詞（不符 / 雖無 / 改口否認 / 倘被告⋯）出現則否決
6. 跳過判例引用語境（`判決意旨` / `裁定意旨` / `判例` 在窗內）

這套邏輯把 §17 Ⅱ 從 P=0% (只用簡單字面 regex) 推到 F1=0.93，把 §59 從 P=0.28 推到 F1=1.00。

### §57 量刑因子 LLM 抽取

刑法第 57 條十款因子（動機/品行/犯後態度⋯）是**主觀質性判斷**，regex 抓不到。改用 5 個 Claude Code sub-agent 平行讀 1598 件判決理由段，每件輸出：

```json
{
  "post_attitude": {"direction": "mitigating",
                     "evidence": "犯後坦承犯行，態度尚佳"},
  ...
}
```

direction ∈ {mitigating, aggravating, neutral, absent}，evidence 須為**原文片段**驗證可回查。

訓練模型時 30 個 one-hot features（10 因子 × 3 方向）放進 XGBoost。`post_attitude_agg` / `_mit` 排到 importance 第 4–5 名，但整體 MAE 影響有限（~0.1 月）— 因子訊號和 §17Ⅱ 高度相關。

### 純質淨重抽取

```python
_NET_WEIGHT_RE = re.compile(
    r"(?:純\s*質\s*)?(?:驗\s*餘\s*)?淨\s*重\s*(?:約|共\s*計)?\s*"
    r"(?P<value>[\d點\.]+|[壹貳參肆伍...]+(?:點[壹貳參肆伍...]+)?)\s*"
    r"(?P<unit>公\s*斤|公\s*克|克)"
)
```

支援阿拉伯（`0.226`）和中文小數（`零點貳貳陸`、`拾陸點柒零`）。22.7% 案件抽得到淨重，最大 131 公斤（重訴 大型運輸案）。

### rules.py 法定刑度約束

毒品危害防制條例 §4–§11 + 刑法 §47/§59 法定刑度做成 lookup table，predict 後依 `binding_constraint(behaviors, drug_levels, reductions)` clip 到合法範圍。

關鍵：用**主文-only 已定罪行為**（非全文 union）做 lookup — 否則 union 包含起訴書/事實段提到但未定罪的高階行為，會把刑度上限算過高。改成主文-only 後，violations 從 18% 降到 4%。

## 誤差分析（按主要罪名）

| 主要罪名 | n | MAE | 中位刑期 |
|---|---|---|---|
| 施用 | 247 | **1.17 月** | 3 月 |
| 持有 | 23 | 1.70 月 | 4 月 |
| **販賣** | 22 | **16.80 月** | 30.5 月 |
| **運輸** | 5 | **26.88 月** | 48 月 |
| 意圖販賣而持有 | 2 | 39.00 月 | 87 月 |
| 製造 | 2 | 10.55 月 | 89 月 |

### 系統性誤差來源

1. **重罪訓練資料稀缺** — 販賣/運輸/製造合計 ~85 件，刑期變異 12–216 月
2. **重罪「過度減刑」失準** — §17Ⅱ + §59 雙重減刑時，模型估不準法官實際讓步幅度
3. **多被告 feature 對齊問題** — `behaviors` 是全文 union 但 target 是首被告刑期，~5–10% 案件受影響

## 未來方向

- 擴大重罪訓練資料（50+ 個月覆蓋）
- 多被告案件 per-defendant 拆解
- `convicted_behaviors`（主文-only）作為 ML feature 而非僅約束 lookup
- §57 因子降噪（由 Claude API 集成多 agent 投票替代單 agent）
- 信賴區間預測（quantile regression）取代 point estimate

## 模組

| 檔案 | 功能 |
|---|---|
| `records.py` | RAR/JSON 解析、案件結構 |
| `filter.py` | 基隆地院毒品案件篩選 |
| `features.py` | 結構化特徵抽取（行為、級數、§17/§59、淨重、量刑）|
| `rules.py` | 毒品條例 §4–§11 + 刑法 §47/§59 法定刑度 |
| `scripts/04_train_baseline.py` | XGBoost baseline + rule-clip |
| `scripts/05_sample_for_labeling.py` | 人工標註抽樣 + prefill |
| `scripts/06_evaluate_labels.py` | features.py 對 ground-truth 評估 |
| `data/processed/art57_factors.jsonl` | LLM 抽取的 §57 量刑因子（1,598 件）|

## 限制與聲明

- **僅供研究用途**。本模型不可作為法律建議或審判依據。
- 訓練資料限於基隆地方法院，**不適用於其他法院或不同罪名**。
- 純質淨重欄位 22.7% 覆蓋率有限，重大案件仍需專家輔助判斷。
- features.py 在多被告案件採首被告視角；複雜共犯結構未完整建模。

<small>Last updated: 2026-05-03 · Built with Python 3.9, XGBoost 2.1, Claude Code sub-agents</small>
