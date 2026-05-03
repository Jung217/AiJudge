# AiJudge

> **臺灣基隆地方法院毒品案件量刑預測模型**
> 38 個月、1,598 件判決 · features.py 結構化抽取 + XGBoost + 法定刑度約束

完整成果展示 → <https://jung217.github.io/AiJudge/>

## 核心成果

| 指標 | 結果 |
|---|---|
| 量刑（月）抽取準確率（n=94 標註樣本）| **100% exact match, MAE 0.0** |
| §17/§59 偵測 F1（n=100）| **0.93–1.00** |
| 行為 / 毒品級數抽取 Jaccard | **0.95 / 0.97** |
| **XGBoost MAE**（test n=301）| **3.09 月** vs. median 基線 6.57 |
| **R²** | **0.756** |

施用/持有 案件 MAE **1–2 月**；販賣/運輸 MAE **17–27 月**（資料稀疏）。

## 資料管線

```
司法院月度 RAR 檔  →  解壓 (bsdtar)  →  records.iter_records_dir
        |
filter.py  --->  KL prefix + 毒品案 + 一審有罪 → 1,598 件 JSONL
        |
features.py  --->  行為(7類) / 級數 / §17 §59 §47 / 純質淨重 / 量刑
        |                                       ^
        |                              §57 因子（Claude sub-agent 抽取）
        v
04_train_baseline.py  --->  XGBoost + rules.py 法定刑度 clip
```

## 安裝

```bash
git clone https://github.com/Jung217/AiJudge
cd AiJudge
pip install -r requirements.txt
# bsdtar (Windows tar.exe) 或 unrar 任選一個用於解壓 .rar
```

## 用法

```bash
# 1. 把月度 RAR 放進 data/raw/
# 2. 解壓
mkdir -p data/extracted
for r in data/raw/*.rar; do tar -xf "$r" -C data/extracted/; done

# 3. 過濾基隆毒品案件
python scripts/02_filter.py --zip-dir data/extracted \
    --out data/filtered/keelung_drug.jsonl

# 4. 統計探索
python scripts/03_explore.py --in data/filtered/keelung_drug.jsonl

# 5. 訓練 baseline
python scripts/04_train_baseline.py
python scripts/04_train_baseline.py --rule-clip   # 法定刑度約束

# 6. 抽 100 件人工驗證樣本（gt_* 預填 auto_*，只需修改錯的）
python scripts/05_sample_for_labeling.py --n 100 --prefill
# 填完 → python scripts/06_evaluate_labels.py
```

## 模組

| 檔案 | 功能 |
|---|---|
| `records.py` | RAR/JSON 串流解析 |
| `filter.py` | 基隆地院毒品一審有罪案件篩選 |
| `features.py` | 結構化特徵抽取（**驗證 F1≥0.93** on 100 件）|
| `rules.py` | 毒品條例 §4–§11 + 刑法 §47/§59/§51 法定刑度 |
| `scripts/02_filter.py` | filter pipeline |
| `scripts/04_train_baseline.py` | XGBoost regressor + rule-clip |
| `scripts/05_sample_for_labeling.py` | 人工標註抽樣（--prefill）|
| `scripts/06_evaluate_labels.py` | features.py 對 ground-truth 評估 |
| `scripts/07_llm_extract_factors.py` | §57 量刑因子 LLM 抽取 scaffold |
| `data/processed/art57_factors.jsonl` | §57 因子（5 sub-agent 平行抽 1,598 件）|

詳細演算法說明：<https://jung217.github.io/AiJudge/>

## 技術亮點

- **Citation-anchored §17/§59 偵測**：跳過 recital 樣板（按⋯定有明文）+ 句子收斂窗口 + 應用/拒絕詞偵測 → §17Ⅱ F1 0.93、§59 F1 1.00
- **多被告/多罪併罰應執行刑抽取**：偵測「。<被告名>犯」boundary 區分定執行刑歸屬，量刑 exact-match 100%
- **純質淨重 regex**：支援阿拉伯（`0.226`）+ 中文小數（`零點貳貳陸` / `拾陸點柒零`）
- **§57 量刑因子 LLM 抽取**：5 個 Claude Code sub-agent 平行處理，每件輸出 10 因子 × {mitigating, aggravating, neutral, absent}
- **法定刑度約束**：用主文-only 已定罪行為 lookup → 違約率 18% → 4%

## 資料來源

- 司法院月度判決書 RAR：<https://opendata.judicial.gov.tw/api/FilesetLists/{id}/file>（id 範圍 63694–64055，1996-01 至 ~2026-03）
- 受機器人挑戰保護，需 Playwright + cookie 注入或瀏覽器手動下載
- 主要欄位：`JID`(案號) / `JYEAR` / `JCASE`(字別) / `JNO` / `JDATE` / `JTITLE`(案由) / `JFULL`(全文) / `JPDF`

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

## License

MIT — see [LICENSE](LICENSE).
