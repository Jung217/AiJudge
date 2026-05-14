# AiJudge

> **臺灣基隆地方法院毒品案件量刑預測模型**
> 
> 2018-01 ~ 2026-02、98 個月、5,809 件判決 · features.py 結構化抽取 + XGBoost + 法定刑度約束 + walk-forward 時序 CV

> 成果展示 → [jung217.github.io/AiJudge](https://jung217.github.io/AiJudge/)

## 核心成果

| 指標 | 結果 |
|---|---|
| 量刑（月）抽取準確率（n=94 標註樣本）| **100% exact match, MAE 0.0** |
| §17/§59 偵測 F1（n=100）| **0.93–1.00** |
| 行為 / 毒品級數抽取 Jaccard | **0.95 / 0.97** |
| **XGBoost p50 MAE**（walk-forward 5 折,pooled n=4,509）| **2.20 月** vs. median 基線 3.52 |
| **R²** / **±3 月命中率** / **±6 月命中率** | **0.589 / 91.0% / 95.3%** |
| **Quantile pinball loss** (p25 / p50 / p75) | **0.79 / 1.10 / 1.01** |
| **[p25, p75] 區間覆蓋率**(CQR δ-shift 校正後)| **51.4%**(目標 ~50%;校正前 44.9%)|
| **緩刑分類器**(基底 0.91%,sqrt(pos_w)+F1-max threshold)| **PR-AUC 0.305(33× 基底)**,P 37.5% / R 36.6% / acc 98.87% |
| **法定刑度越界率**(rule-clipped)| **0.00%**(raw 預測 3.5% → clip 後 0%)|

施用 MAE **1.00 月**、持有 **1.08 月**、轉讓 **4.50 月**;販賣 **7.09 月**、運輸 **28 月**、製造 **22 月**(重罪樣本仍稀,變異大)。

## 亮點

- **Citation-anchored §17/§59 偵測**：跳過 recital 樣板（按⋯定有明文）+ 句子收斂窗口 + 應用/拒絕詞偵測 → §17Ⅱ F1 0.93、§59 F1 1.00
- **多被告/多罪併罰應執行刑抽取**：偵測「。<被告名>犯」boundary 區分定執行刑歸屬，量刑 exact-match 100%
- **純質淨重 regex**：支援阿拉伯（`0.226`）+ 中文小數（`零點貳貳陸` / `拾陸點柒零`）
- **§57 量刑因子 LLM 抽取**：`scripts/07_llm_extract_factors.py` 已 wire 到 Anthropic Claude API(Haiku 4.5 + prompt cache + 並行 + resume),每件輸出 10 因子 × {mitigating, aggravating, neutral, absent};需設 `ANTHROPIC_API_KEY` 才實跑,全量 5,809 件估約 USD 17–45
- **多 head 輸出**(plan §5.2):p25 / p50 / p75 三個 quantile head(`reg:quantileerror`)+ 緩刑二元分類 head;`models.predict_with_constraints` 一次回 `{p25, p50, p75, probation_prob}`,post-clip 強制單調
- **Quantile 校正**(conformal δ-shift):每折拿 val tail 殘差算 `δ_α = quantile(y-ŷ, α)`,test 時 `pred + δ`,把 [p25, p75] 區間覆蓋率從 44.9% → **51.4%**(目標 50%);δ 存進 ModelBundle metadata,推論時自動套用
- **法定刑度約束**:用主文-only 已定罪行為 lookup + §17Ⅰ/§17Ⅱ/§59/§25Ⅱ 未遂/§62 自首 各自 ½ 減刑(§70 compound)+ 簡易判決 floor ½ + 數罪併罰 30 年上限 → 模型預測越界率 3.5% → clip 後 **0%**;ground-truth 標籤殘留越界率 **0.91%**(殘留主要是 §11 持有純質淨重加重型未建表 + §59 隱性減刑沒明引條號)
- **Walk-forward 時序 CV**:5 折擴展視窗,測試集嚴格晚於訓練;最後一折(2024-01–2026-02 測試) MAE 2.47 月、±6mo 93.6%

## 模組

| 檔案 | 功能 |
|---|---|
| `records.py` | RAR/JSON 串流解析 |
| `filter.py` | 基隆地院毒品一審有罪案件篩選 |
| `features.py` | 結構化特徵抽取（**驗證 F1≥0.93** on 100 件）|
| `rules.py` | 毒品條例 §4–§11 + 刑法 §47/§59/§51 法定刑度 |
| `scripts/02_filter.py` | filter pipeline |
| `scripts/04_train_baseline.py` | XGBoost p25/p50/p75 quantile + 緩刑分類 + rule-clip(walk-forward CV)|
| `scripts/05_sample_for_labeling.py` | 人工標註抽樣（--prefill）|
| `scripts/06_evaluate_labels.py` | features.py 對 ground-truth 評估 |
| `scripts/07_llm_extract_factors.py` | §57 量刑因子 Claude API 抽取(prompt cache + resume)|
| `data/processed/art57_factors.jsonl` | §57 因子（5 sub-agent 平行抽 1,598 件）|
| `app.py` | FastAPI 服務:`/health` / `/version` / `/predict` / `/explain`,回傳 p25-p50-p75-緩刑機率 + 法定刑度 + SHAP per-feature 拆解 + 免責聲明 |

## 安裝

```bash
git clone https://github.com/Jung217/AiJudge
cd AiJudge
pip install -r requirements.txt
# bsdtar (Windows tar.exe) 或 unrar 任選一個用於解壓 .rar
```

## 資料來源

- 司法院月度判決書 RAR：<https://opendata.judicial.gov.tw/api/FilesetLists/{id}/file>（id 範圍 63694–64055，1996-01 至 ~2026-03）
- 受機器人挑戰保護，需瀏覽器手動下載(file API 對所有 fileSetId 回 500)
- 主要欄位：`JID`(案號) / `JYEAR` / `JCASE`(字別) / `JNO` / `JDATE` / `JTITLE`(案由) / `JFULL`(全文) / `JPDF`

## 用法

```bash
# 1. 把月度 RAR 放進 data/raw/
# 2. 並行解壓(WORKERS 控制併發,預設 1;6 在 12 核機約 50 分鐘解 30 個月)
WORKERS=6 python scripts/01b_extract_rars.py
#    RAR_DIR / OUT_DIR / EXTRACT_LOG 可用 env var 覆寫(例如解到 D:\)

# 3. 過濾基隆毒品案件
python scripts/02_filter.py --zip-dir data/extracted \
    --out data/filtered/keelung_drug_all.jsonl

# 4. 統計探索
python scripts/03_explore.py --in data/filtered/keelung_drug_all.jsonl

# 5. 訓練 baseline(預設 walk-forward 時序 CV,法定刑度 clip 預設開)
python scripts/04_train_baseline.py
python scripts/04_train_baseline.py --no-rule-clip   # layer-4 ablation
python scripts/04_train_baseline.py --save data/processed/baseline_model.pkl

# 6. 抽 100 件人工驗證樣本（gt_* 預填 auto_*，只需修改錯的）
python scripts/05_sample_for_labeling.py --n 100 --prefill
# 填完 → python scripts/06_evaluate_labels.py

# 7. 起 API 服務(回傳 p25/p50/p75/緩刑機率/法定刑度/免責聲明)
uvicorn app:app --reload --port 8000
# 試打:
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"behaviors":["施用"],"drug_levels":[2],"can_convert_to_fine":true,"jcase":"簡"}'

# 8. SHAP 解釋(top=8 個對該預測月數影響最大的特徵)
curl -X POST "http://127.0.0.1:8000/explain?top=8" -H "Content-Type: application/json" \
  -d '{"behaviors":["販賣"],"drug_levels":[2],"art17_2":true,"art59":true}'
```

## 參考

- [司法院裁判書系統](https://judgment.judicial.gov.tw/FJUD/default.aspx)
- [司法院資料開放平臺](https://opendata.judicial.gov.tw/)
- 毒品危害防制條例 §4–§11
- 刑法 §47（累犯）/ §51（多罪併罰）/ §59（酌減）/ §66（減刑限度）
- 司法院釋字第 775 號（累犯加重必要性）

## 限制與聲明

- **僅供學術研究用途**。本模型不可作為法律建議或審判依據。
- 訓練資料限於基隆地方法院，不適用於其他法院或不同罪名。
- 純質淨重欄位 22.7% 覆蓋率，重大案件仍需專家輔助判斷。
- 多被告案件採首被告視角；複雜共犯結構未完整建模。
