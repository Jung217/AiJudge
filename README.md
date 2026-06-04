# AiJudge

> **臺灣北部 5 地方法院(基/北/士/新北/桃)毒品案件量刑預測模型**
> 
> 2018-01 ~ 2026-02、98 個月、**68,985 件判決**(其中基隆 5,809、新北 24,947、桃園 20,750、臺北 11,324、士林 6,155)· features.py 結構化抽取 + XGBoost + 法定刑度約束 + walk-forward 時序 CV + 多 head(p25/p50/p75 quantile + 緩刑分類器 + SHAP)

> 成果展示 → [jung217.github.io/AiJudge](https://jung217.github.io/AiJudge/)

## 核心成果

| 指標 | 結果 |
|---|---|
| 量刑（月）抽取準確率（n=94 標註樣本）| **100% exact match, MAE 0.0** |
| §17/§59 偵測 F1（n=100）| **0.93–1.00** |
| 行為 / 毒品級數抽取 Jaccard | **0.95 / 0.97** |
| **XGBoost p50 MAE**(walk-forward 5 折,pooled n=51,944,5 院混合)| **2.49 月** vs. median 基線 6.00 |
| **基隆 test 列 MAE**(n=4,484,同一個 5 院 model 評估)| **1.79 月**(比基隆-only model 的 2.20 還好 ~19%)|
| **R²** / **±3 月命中率** / **±6 月命中率** | **0.717 / 87.8% / 93.4%** |
| **Quantile pinball loss** (p25 / p50 / p75) | **0.98 / 1.25 / 1.17** |
| **[p25, p75] 區間覆蓋率**(CQR δ-shift 校正後)| **50.1%**(目標 ~50%,完美命中)|
| **緩刑分類器**(基底 2.87%,sqrt(pos_w)+F1-max threshold)| **PR-AUC 0.367(13× 基底)**,P 34.5% / R 71.7% / acc 95.27% |
| **法定刑度越界率**(rule-clipped)| **0.00%**(raw 預測 3.4% → clip 後 0%)|

**Per-court MAE**:基隆 **1.79**、新北 2.21、士林 2.20、臺北 2.44、桃園 3.10 月 — 桃園變異最大、基隆最穩。

**Per-primary-behavior MAE**(5 院 pooled):施用 **1.11 月**(n=25k)、持有 **1.06**(n=19k)、轉讓 **2.31**;販賣 **8.70**(n=5.9k)、運輸 **26.07**、製造 **15.10**、意圖販賣而持有 **11.36**。

**改善路徑**(五項實驗,MAE 2.85 → 2.49,基隆 2.06 → 1.79,販賣 10.36 → 8.70,raw quantile crossings 18.7% → 7.3%):
1. 加 `sum_individual_months` 特徵(數罪併罰個刑加總) → MAE 2.85→2.63
2. §17/§59 reject regex 增強(辯護人主張+駁回模式、未因而查獲、尚無...之適用) → 重罪 -1 月,raw crossings 20.8%→17.5%
3. p50 head 改用 `reg:absoluteerror`(L1 loss,對重罪 outliers 穩健) → MAE 2.62→2.52,raw crossings 17.5%→6.5%
4. learning rate 0.05→0.03、rounds 500→1000、early-stopping 30→50(L1 loss 喜歡 small-step / many-rounds)→ MAE 2.52→2.51
5. p50 head 改 3-seed ensemble(平均 seed 42/43/44 預測)→ MAE 2.51→**2.49**,販賣 -0.07 月、5 院全部微正向

## 亮點

- **Citation-anchored §17/§59 偵測**：跳過 recital 樣板（按⋯定有明文）+ 句子收斂窗口 + 應用/拒絕詞偵測 → §17Ⅱ F1 0.93、§59 F1 1.00
- **多被告/多罪併罰應執行刑抽取**：偵測「。<被告名>犯」boundary 區分定執行刑歸屬，量刑 exact-match 100%
- **純質淨重 regex**：支援阿拉伯（`0.226`）+ 中文小數（`零點貳貳陸` / `拾陸點柒零`）
- **§57 量刑因子啟發式抽取**：`scripts/art57_extract.py` 用高精度 regex/pattern 鎖定判決書「爰審酌⋯」量刑段,抽 10 因子 × {mitigating, aggravating, neutral, absent} + exact-substring evidence,**零外部 API**;產出 `data/processed/art57_factors.jsonl`(基隆 1,598 件),由 `04_train_baseline._load_art57` 編成 `a57_*` 特徵
- **多 head 輸出**(plan §5.2):p25 / p50 / p75 三個 quantile head(`reg:quantileerror`)+ 緩刑二元分類 head;`models.predict_with_constraints` 一次回 `{p25, p50, p75, probation_prob}`,post-clip 強制單調
- **Quantile 校正**(conformal δ-shift):每折拿 val tail 殘差算 `δ_α = quantile(y-ŷ, α)`,test 時 `pred + δ`,把 [p25, p75] 區間覆蓋率從 44.9% → **51.4%**(目標 50%);δ 存進 ModelBundle metadata,推論時自動套用
- **法定刑度約束**:用主文-only 已定罪行為 lookup + §17Ⅰ/§17Ⅱ/§59/§25Ⅱ 未遂/§62 自首 各自 ½ 減刑(§70 compound)+ 簡易判決 floor ½ + 數罪併罰 30 年上限 + **§11Ⅴ/Ⅵ 持有純質淨重加重型**(第一級 ≥10g / 第二級 ≥20g 自動切換到加重刑度範圍) → 模型預測越界率 ~4% → clip 後 **0%**
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
| `scripts/art57_extract.py` | §57 量刑因子啟發式抽取(regex/pattern,無外部 API)|
| `scripts/09_inspect_outliers.py` / `10_error_registry.py` | walk-forward 殘差離群 → 可累積錯誤登記(§6.3 偏誤登記)|
| `data/processed/art57_factors.jsonl` | §57 因子(啟發式抽 1,598 件;gitignored,`art57_extract.py` 可重生)|
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

# 3. 過濾北部 5 院毒品案件(基/北/士/新北/桃)
python scripts/02_filter.py --zip-dir data/extracted --courts northern \
    --out data/filtered/north5_drug_all.jsonl
#    若只想要基隆,把 --courts northern 拿掉(預設 keelung)

# 4. 統計探索
python scripts/03_explore.py --in data/filtered/north5_drug_all.jsonl

# 5. 訓練 baseline(預設 walk-forward 時序 CV,法定刑度 clip 預設開,
#    預設讀 north5_drug_all.jsonl)
python scripts/04_train_baseline.py
python scripts/04_train_baseline.py --no-rule-clip       # layer-4 ablation
python scripts/04_train_baseline.py --in data/filtered/keelung_drug_all.jsonl  # 只看基隆
python scripts/04_train_baseline.py --save data/processed/baseline_north5_model.pkl

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
- 訓練資料限於北部 5 地方法院(基/北/士/新北/桃)毒品案件,不適用於其他法院或不同罪名。模型用 `court_*` flag 區分各院量刑風格,基隆案件預測 MAE 2.06 月、其他院 2.5-3.5 月。
- 純質淨重欄位 22.7% 覆蓋率，重大案件仍需專家輔助判斷。
- 多被告案件採首被告視角；複雜共犯結構未完整建模。
