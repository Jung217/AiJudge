---
layout: default
title: AiJudge — 臺灣北部 5 地方法院毒品案件量刑研究
---

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>

# AiJudge：臺灣北部 5 地方法院毒品案件量刑研究

<p class="byline">
以 2018-01 ~ 2026-02、98 個月、北部 5 地方法院（基隆／臺北／士林／新北／桃園）共
68,985 件毒品判決為樣本，建立可預測判決刑期的研究模型，並以 walk-forward 時序
交叉驗證評估其在實務上的泛化能力。
</p>

## 一、研究摘要

長期以來，社會對司法量刑的一個關注是：**性質相似的案件，是否得到合理一致的判決？**個案差異固然存在，但不同被告、相近事實、刑度卻顯著不同，往往讓當事人與家屬無所適從。本研究以**司法院公開資料**為基礎，將判決書這種文字密集的法律文件，系統性地轉換成電腦能讀懂的結構化資料，並以此訓練統計模型，預測在臺灣北部 5 地方法院審理的毒品案件可能會獲得怎樣的刑期。

研究發現：(1) 透過程式自動辨識判決中的關鍵法律要件（行為態樣、毒品級別、減刑事由等），準確率在驗證樣本上達到 **F1 0.93–1.00**；(2) 機器學習模型在 5 院聯合訓練後對中位刑期預測 **MAE 2.85 個月**、±3 個月命中率 86.0%；(3) 同一個 5 院模型在**基隆案件**測試列上 MAE 反而降到 **2.06 月**，比基隆-only 模型的 2.20 月還好 ~6%——其他 4 院的案件當作增強樣本、用「法院」one-hot flag 區分風格，反而讓基隆規律學得更穩；(4) 對重罪（販賣、運輸、製造）樣本擴大 10–100×，但平均誤差仍受桃園／新北的寬刑度分布牽動。

研究的價值不在取代法官，而在於提供**透明且可量化**的量刑趨勢資訊，幫助當事人、辯護律師理解自身案件可能落點，輔助學術研究觀察司法實務變化，並為日後可能的「量刑一致性監測」奠定方法論基礎。

## 二、研究背景

### 2.1 為何研究量刑

刑事審判最後階段——量刑——具有相當大的法官裁量空間。法律雖然規定了刑度範圍（例如販賣第二級毒品「無期徒刑或十年以上有期徒刑」），但同樣是販賣案件、相似的數量和情節，最終判決可能從十年到二十年不等。當事人在面對訴訟時，很難取得**客觀、量化、基於實際判決資料**的參考點。

過去這類研究多依賴法律學者的手工分析，受限於樣本大小和分析效率，難以在統計意義上得到結論。近年自然語言處理與機器學習的進步，讓我們有機會用程式系統性處理大量判決書，補上這個資料缺口。

### 2.2 為何選擇北部 5 院的毒品案件

選擇單一罪名類型、橫跨北部 5 院（基隆／臺北／士林／新北／桃園）做為起點，是基於四個考量：

1. **資料可得性**：司法院自 1996 年起公開判決書資料，可下載完整文字
2. **法條相對單純**：毒品危害防制條例 §4–§11 的構成要件清楚，便於程式辨識
3. **案件量充足**：5 院 8 年累積 68,985 件（單被告且含量刑數字者 62,328 件），足以做時序交叉驗證並涵蓋重罪
4. **法理一致、量刑風格不同**：5 院適用相同的中央法律，但歷年量刑統計存在系統性差異——這正好讓模型用「法院」one-hot 特徵自動吸收區域風格，並讓基隆等案件量較少的法院從其他院的訓練資料中受益

案件量分布（單被告、有量刑數字）：

| 法院 | JID prefix | 件數 |
|---|---|---|
| 新北 | PC | 24,947 |
| 桃園 | TY | 20,750 |
| 臺北 | TP | 11,324 |
| 士林 | SL | 6,155 |
| 基隆 | KL | 5,809 |

未來若將方法論擴及其他法院或罪名，需重新校準。

## 三、研究方法

### 3.1 從判決書到結構化資料

判決書是寫給人看的文字，要讓電腦處理，必須先把文字中的關鍵事實「結構化」。本研究將每份判決書轉換成以下幾類欄位：

- **行為態樣**：被告所犯的行為（販賣、施用、持有、運輸、製造、轉讓、意圖販賣而持有）
- **毒品級別**：第一級至第四級毒品
- **法定減免事由**：是否適用毒品危害防制條例第 17 條第 1 項（供出毒品來源）、第 2 項（偵審自白），以及刑法第 47 條（累犯）、第 59 條（酌減）
- **量刑事實**：純質淨重、是否得易科罰金、是否宣告緩刑
- **量刑因子**：刑法第 57 條十款主觀因素（犯罪動機、品行、犯後態度等）
- **判決結果**：實際宣告的刑期

### 3.2 為什麼這項工作不容易

判決書的書寫風格雖有套式但相當繁複。同樣一個減刑事由，法官可能寫成「依毒品危害防制條例第 17 條第 2 項規定，減輕其刑」、「爰依該項規定減刑」、或在解釋條文時順帶引用，但實際上**沒有適用**該條於該名被告。要正確辨識這些差異，需要兼顧：

- **法條引用語境**：是真正適用，還是只是引述法律（recital）或舉例
- **拒絕語**：「不符合」、「並無⋯之適用」、「改口否認」等否決語
- **多被告處理**：一份判決可能有多名被告，刑期、減刑各不相同

本研究透過反覆迭代（共驗證 100 件人工標註樣本）才將自動辨識準確率推到實用水準。

### 3.3 預測模型

在結構化資料上訓練 XGBoost，採「分層 bundle」設計：四個獨立的 head 共用同一份特徵矩陣。

- **中位數刑期迴歸（p50）**：squared error loss，預測「最可能的有期徒刑月數」
- **下／上分位數迴歸（p25 / p75）**：`reg:quantileerror`，給出「合理刑度的 50% 信賴區間」
- **緩刑分類器**：binary logistic，輸出「該案被宣告緩刑的機率」
- **法院 one-hot 特徵**：`court_KL/TP/SL/PC/TY` 進入特徵矩陣，讓模型自動區分各院量刑風格

預測完成後再經過「規則約束層」：依法定刑度（毒品條例 §4–§11，加上 §17Ⅰ／§17Ⅱ／§59／§62 自首／§25Ⅱ 未遂各自 ½ 減刑、簡易判決下限 ½、數罪併罰應執行刑 30 年上限）clip 到合法範圍，並重新單調化 p25 ≤ p50 ≤ p75。**模型在規則層之後 100% 不會輸出違反法律的刑期。**

### 3.4 評估方式：walk-forward 時序交叉驗證

判決資料隨時間累積，**鄰近時間的案件往往涉及相似事實、相似法官、相似量刑風氣**，若用隨機 80／20 切分會讓「時間上相鄰」的訓練／測試案件互相洩漏資訊，估出過於樂觀的誤差。

因此本研究改用 **expanding-window walk-forward CV**：把全部案件按 `JDATE` 排序後切成 6 個等量區段，第 1 段固定當作種子訓練資料，之後每一折以「累積前面所有區段」訓練、「下一段」測試，共得 5 折，**測試集永遠嚴格晚於每一筆訓練資料**。最後我們把 5 折的測試樣本（合計 51,940 件）pool 起來計算指標。

### 3.5 分位數校正：conformal δ-shift

XGBoost 的 quantile head 受正則化影響，原始預測的 `[p25, p75]` 經驗覆蓋率約 45%，比理想的 50% 略偏窄。本研究在每折訓練結束後，於 val 尾段算殘差的 α-分位數 `δ_α = quantile(y − ŷ, α)`，在 test 時把 raw quantile 預測加上 δ_α，這就是標準的 **conformalized quantile regression**（CQR）marginal shift。校正後 5 院 pooled 覆蓋率從 49.6% → **49.7%**（基隆-only model 校正前 44.9% → 51.4%），δ 與模型一起寫入 ModelBundle metadata，推論時自動套用。

## 四、研究發現

### 4.1 結構化抽取準確率

在 100 件人工標註的驗證樣本上，自動抽取的關鍵欄位達到下列水準：

<canvas id="chart-extract" style="max-height:340px"></canvas>

對最關鍵的**量刑數字**（有期徒刑月數），在 94 件有量刑的樣本中達到 **100% 完全相符**。這一數字得來不易，因為它涉及辨識主文中「應執行有期徒刑伍月」這類多罪併罰結果，並區分多被告場合下哪一段刑期屬於哪位被告。

### 4.2 模型預測誤差

在 walk-forward 5 折的合計 51,940 件測試樣本上，模型的整體表現如下：

<canvas id="chart-mae" style="max-height:300px"></canvas>

5 院聯訓 XGBoost 模型的中位數預測（p50）平均絕對誤差（MAE）為 **2.85 個月**，相較於以「永遠輸出中位數」的天真基線 6.00 個月，**誤差降低約 53%**；中位殘差 1.0 個月、±3 個月命中率 **86.0%**、±6 個月命中率 **92.4%**、R² **0.672**。最後一折（2024-05 ~ 2026-02 測試，訓練 51,940 件）MAE 3.55 月、R² 0.697，顯示模型對較近期的判決仍有穩定的泛化能力。

### 4.2.1 各法院 MAE 拆解

同一個 5 院模型在不同法院的 test 列上拆開算 MAE：

<canvas id="chart-by-court" style="max-height:300px"></canvas>

**基隆案件的 MAE 反而最低**（2.06 月），低於基隆-only 訓練時的 2.20 月——其他 4 院案件當作增強樣本、用「法院」one-hot flag 區分風格後，基隆規律學得更穩。這是「跨院聯訓 + 用 court flag 區分」的關鍵價值。整體 MAE 2.85 之所以較高，純粹是被桃園／新北的寬刑度分布拉高，並非每院個別變差。

### 4.2.2 區間預測（quantile）與緩刑分類

單一數字當預測對使用者其實不夠用——「比較可能落在哪個範圍」、「會不會給緩刑」往往是當事人更想知道的。本研究在同一份特徵上加上三個分位數迴歸頭（p25 / p50 / p75，採 XGBoost `reg:quantileerror`，並用 §3.5 的 CQR δ-shift 校正）與一個緩刑二元分類頭：

<canvas id="chart-quantile" style="max-height:300px"></canvas>

整體 `[p25, p75]` 區間覆蓋率經 CQR δ-shift 校正後達 **49.7%**（理想 50%）。三個 pinball loss（p25 / p50 / p75）= 1.03 / 1.43 / 1.31 月，量級對得起 MAE。

緩刑分類器在 5 院聯訓下基底為 **2.87%**（比單院基隆 0.91% 高 3 倍），達 **PR-AUC 0.367**（相當於基底的 13 倍），accuracy 95.27%、precision 34.5%、recall 71.7%（F1-max threshold tuning，threshold 隨各折在 0.20 ~ 0.55 間自動調）。使用者場景以「篩出可能符合緩刑條件的案件給律師複看」較合適，而非自動決策。

### 4.3 不同罪名的預測難度差異

把所有 walk-forward 測試樣本依主要罪名分組，預測誤差呈現明顯差異：

<canvas id="chart-by-crime" style="max-height:340px"></canvas>

施用案件（n = 25,367）誤差僅 1.25 個月、持有（n = 18,687）1.17 個月，這類案件刑期普遍 2–6 個月、影響因素相對單純，模型表現極佳。販賣（n = 5,936）平均誤差 10 個月，運輸（n = 879）和製造（n = 264）這類重罪則來到 18–29 個月。主因是這類案件**刑期變異極大**（例如販賣可能 6 個月、也可能十年以上有期徒刑甚至無期），而真正決定刑度的因子（共犯角色、犯後態度、和解、悔意等）目前還沒被結構化抽取出來。注意 5 院聯訓後重罪樣本數從基隆-only 的 47～571 件大幅擴大到 264～5,936 件，但平均 MAE 仍受桃園／新北寬分布牽動。

### 4.4 哪些因素最影響量刑

機器學習模型可以告訴我們，**哪些因素的存在最能解釋刑期的差異**。將模型內部的「特徵重要性」由高到低排序：

<canvas id="chart-importance" style="max-height:380px"></canvas>

排名前列的因素，與法律實務上的量刑邏輯**高度一致**：

- **行為類型「販賣」**居首：大宗加重型行為對刑度有最大解釋力
- **是否得易科罰金**反映案件嚴重程度（短期自由刑才能易科），排第 2
- **毒品級別第四／第一**緊隨其後
- **刑法第 59 條酌減**仍位列前 5
- **數罪併罰**（`is_aggregate_sentence`）、**未遂**（`is_attempt`）、**法院（court_TP 等）**也都進前 15，模型有自動區分各院量刑風格
- **§17Ⅱ 偵審自白**雖在重要性表外（5 院聯訓後判決訊號被稀釋），但在規則層仍套用 ½ 減刑

這個結果**驗證了模型確實學到了合理的判決邏輯**，而非依賴噪音特徵。

## 五、限制與誤差來源

研究在三個面向遇到瓶頸，這也是後續工作的方向：

**1. 重罪資料樣本仍偏少且分布寬**　5 院聯訓後重罪樣本擴大 10–100×（販賣 571→5,936、運輸 47→879、製造 14→264），但平均誤差仍 10–29 月。主因不是樣本量不足，而是這類案件本身刑度變異就極大（從 6 個月到無期），加上各院量刑風格差異被一併吸收進來。要進一步壓 MAE 需要結構化擷取「共犯角色、犯後態度、和解金額」等個案決定性因子。

**2. 法官裁量幅度估計困難**　當被告同時有「偵審自白」與「情堪憫恕」兩項減刑事由時，法定上限是降到原刑期的四分之一，但法官實際讓步多少，受個案無數細節影響——共犯關係、悔意程度、家庭因素等。目前模型只有「刑法第 57 條十款主觀因子」的 scaffold（透過 Claude AI 從判決書抽取），尚未跑完全資料集；對重罪案件預測幫助上限仍有限。

**3. 多被告案件對齊問題**　一份判決中可能有兩、三位被告，每人罪名不同、刑期不同。本研究目前**直接過濾掉多被告判決**（5 院 2,413 件 / 約佔 3.5%）以避免標籤雜訊，但同時也犧牲了販賣／運輸案件常見的共犯結構樣本。後續會改成「逐被告獨立建模」。

**4. 殘留法定刑度越界**　rule-clip 後模型預測 0% 違法；但 ground-truth 標籤仍有 **2.10%**（5 院聯訓）落在我們所抽取的法定刑度區間之外（基隆-only 0.91%），主要來自重罪 §59 未被偵測、簡易判決附件中的減刑事由不在 JFULL 中、§11 持有純質淨重加重型未建表、以及 5 院間細微的量刑慣例差異未對齊到單一規則表。

## 六、應用前景

> 研究的價值不在取代法律專業判斷，而在提供**事前可參考的量刑趨勢資訊**。

### 6.1 對當事人與家屬

毒品案件當事人面對訴訟時，最焦慮的就是「不知道會被判多少」。本研究的模型可以根據案情（行為類型、毒品級別、是否自白、是否累犯、起訴法院）等，給出**該類案件在北部 5 院近八年（2018–2026）的判決區間**，幫助當事人有合理預期，避免因資訊落差被誤導。

### 6.2 對辯護律師

律師在量刑辯論時，往往需要援引「類似案件」做為比較基礎。本研究累積的 68,985 件判決資料庫與抽取結果，可作為**量刑辯論的證據資料庫**，律師可快速檢索「同樣是販賣第二級毒品、有第 17 條第 2 項自白、無累犯」的歷年判決，做為向法院提出的參考。

### 6.3 對司法系統的潛在貢獻

更宏觀地說，這類研究最大的潛在價值在三方面：

**加速簡易案件審理**　一個成熟的量刑模型可作為**法官的參考工具**——不是替代法官判斷，而是提供「此類案件近期同庭判決中位數」這樣的數字。對於高度標準化的簡易判決（純施用、單純持有），可顯著降低法官查閱前案的時間成本。粗略估算，若每件簡易判決能省下 5–10 分鐘的查找時間，以基隆每月 50 件估算，**單一法院每年可釋出約 50–100 小時**的法官時間用於更需要審慎處理的疑難案件。

**量刑一致性監測**　司法行政機關可以用類似工具，定期檢視「相似案件刑度離散程度」是否在合理範圍。離群案件可作為司法研究或在職教育的樣本，**長期改善量刑一致性**。

**輔助當事人理解司法**　當判決是個黑箱時，民眾對司法的信任難以建立。本類研究的成果若以親民的方式呈現（例如「全國法院近五年此類案件刑期分布」），有助於提升司法透明度與民眾理解。

### 6.4 對學術與政策研究

這套方法論可推廣到：

- **修法前後量刑變化的影響評估**（如 2020 年毒品條例修正前後的判決趨勢比較）
- **不同法院量刑差異的實證研究**（同樣案情、北中南東法院的判決異同）
- **量刑與被告社經背景關係的探討**（在去識別化前提下）

## 七、未來展望

### 7.1 短期（3–6 個月）

- [done] 資料時長擴展至 98 個月（2018-01 ~ 2026-02）；[done] 擴及北部 5 個地方法院（基、北、士、新北、桃），完成 68,985 件聯合訓練
- [done] 加入信賴區間預測（quantile regression p25 / p50 / p75）並用 CQR δ-shift 校正到目標覆蓋率 50%
- [done] FastAPI 服務 + SHAP `/explain` per-feature 拆解端點
- 改善多被告案件的特徵對齊機制，逐被告獨立建模（目前直接過濾掉約 3.5% 多被告判決，但販賣／運輸重罪共犯結構樣本被一併排除）
- 接 LLM 完成 §57 量刑因子全資料集抽取，對重罪預測精度可望帶來最大幅度提升（已 wire 好 Claude API extractor，待 API 預算）
- 將 5 院 model 轉 ONNX、嵌進本頁面做 in-browser 互動 demo

### 7.2 中期（6 個月–1 年）

- 擴及其他罪名（詐欺、竊盜等案件量大、社會關注的類型）
- 與法律實務工作者合作建立公開查詢介面（網站 / API）
- 結合大型語言模型（如 Claude）做更精細的量刑因子（如和解金額、被害人態度）抽取

### 7.3 長期

- 與司法院 / 法院系統合作，將模型作為**法官資訊輔助工具**（明確定位為「參考」而非「決策」）
- 公開資料集與評估基準，讓其他研究團隊在共同基礎上比較不同方法
- 探討量刑透明度提升對民眾信任、被告認罪率、上訴率的長期影響

## 八、限制與聲明

本研究**僅供學術探討用途**，不能也不應作為下列用途：

- 個案的法律建議
- 任何法庭、行政程序的裁量依據
- 取代律師專業判斷或法官的審判

研究使用的資料限於臺灣北部 5 地方法院（基隆／臺北／士林／新北／桃園）毒品案件，**結果無法外推至其他法院或不同罪名**。模型對重罪預測誤差大，**重大刑案不可使用本工具的預測**。研究團隊不對使用本研究結果所造成的任何後果負責。

完整程式碼與資料處理流程開源於 [github.com/Jung217/AiJudge](https://github.com/Jung217/AiJudge)，歡迎研究者檢視、複現、改進。

## 九、互動量刑試算（教育用途）

下面這個工具讓你選擇案件特徵，**即時看到**該類案件依法的法定刑度區間，以及北部 5 院近 8 年同類案件的**實際判決中位數**。所有計算都在瀏覽器內完成，不送任何資料到後端。每個減刑事由旁邊**滑鼠移上去（hover）**可看到對應法條原文。

<style>
.aj-card {
  margin: 24px 0;
  font-size: 0.95em;
  color: #24292f;
}
.aj-section {
  margin: 20px 0;
}
.aj-section-label {
  font-size: 0.78em;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #57606a;
  margin-bottom: 8px;
}
.aj-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}
.aj-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  background: #f6f8fa;
  border-radius: 999px;
  cursor: pointer;
  transition: all 0.15s ease;
  font-size: 0.9em;
  user-select: none;
}
.aj-chip:hover {
  background: #ddf4ff;
  box-shadow: 0 0 0 1px #0969da inset;
}
.aj-chip input { margin: 0; cursor: pointer; }
.aj-chip input:checked ~ span { color: #0969da; font-weight: 600; }
.aj-chip:has(input:checked) {
  background: #ddf4ff;
  box-shadow: 0 0 0 1px #0969da inset;
}
.aj-chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
.aj-select {
  padding: 8px 12px;
  border-radius: 8px;
  border: none;
  background: #f6f8fa;
  font-size: 0.95em;
  font-family: inherit;
  cursor: pointer;
  min-width: 160px;
}
.aj-select:hover { background: #ddf4ff; }
.aj-select:focus { outline: 2px solid #0969da; background: #fff; }
.aj-input {
  padding: 8px 12px;
  border-radius: 8px;
  border: none;
  background: #f6f8fa;
  font-size: 0.95em;
  font-family: inherit;
  width: 8em;
}
.aj-input:focus { outline: 2px solid #0969da; background: #fff; }
.aj-output {
  margin-top: 28px;
  padding: 24px;
  background: linear-gradient(135deg, #ddf4ff 0%, #f6f8fa 100%);
  border-radius: 12px;
}
.aj-output-label {
  font-size: 0.78em;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #57606a;
  margin-bottom: 6px;
}
.aj-range {
  font-size: 1.6em;
  font-weight: 600;
  color: #0969da;
  letter-spacing: -0.01em;
}
.aj-sub {
  font-size: 1em;
  color: #24292f;
  margin-top: 4px;
}
.aj-row + .aj-row { margin-top: 18px; }
.aj-meta { font-size: 0.85em; color: #57606a; margin-top: 6px; }
.aj-bucket {
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px dashed #d0d7de;
}
.aj-quantiles {
  display: flex;
  gap: 24px;
  margin-top: 8px;
  flex-wrap: wrap;
}
.aj-q { display: flex; flex-direction: column; }
.aj-q-label { font-size: 0.75em; color: #57606a; }
.aj-q-val { font-size: 1.2em; font-weight: 600; color: #24292f; }
.aj-warn {
  font-size: 0.85em;
  color: #57606a;
  margin-top: 16px;
  line-height: 1.5;
}
.aj-warn b { color: #cf222e; }
.aj-tag {
  display: inline-block;
  margin-left: 6px;
  padding: 2px 8px;
  background: #ffebe9;
  color: #cf222e;
  border-radius: 4px;
  font-size: 0.75em;
  font-weight: 600;
}
[data-hint] { cursor: help; position: relative; }
.aj-tooltip {
  position: fixed;
  z-index: 9999;
  max-width: 340px;
  padding: 10px 14px;
  background: #1f2328;
  color: #f6f8fa;
  border-radius: 8px;
  font-size: 0.85em;
  line-height: 1.5;
  box-shadow: 0 4px 16px rgba(0,0,0,0.25);
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.12s ease;
}
.aj-tooltip.visible { opacity: 1; }
.aj-tooltip::after {
  content: "";
  position: absolute;
  top: -6px;
  left: 24px;
  border: 6px solid transparent;
  border-top: 0;
  border-bottom-color: #1f2328;
}
</style>

<div class="aj-card" id="aj-form">

<div class="aj-section">
  <div class="aj-section-label">案件背景</div>
  <div class="aj-chip-row">
    <select class="aj-select" id="aj-court" data-hint data-tip="案件起訴的地方法院(北部 5 院)">
      <option value="KL">基隆地方法院</option>
      <option value="TP">臺北地方法院</option>
      <option value="SL">士林地方法院</option>
      <option value="PC" selected>新北地方法院</option>
      <option value="TY">桃園地方法院</option>
    </select>
    <select class="aj-select" id="aj-level" data-hint data-tip="毒品危害防制條例 §2 將毒品分為一至四級。第一級最重(海洛因、嗎啡等)、第四級最輕。">
      <option value="1">第一級毒品(海洛因、嗎啡)</option>
      <option value="2" selected>第二級毒品(安非他命、大麻)</option>
      <option value="3">第三級毒品(FM2、K他命)</option>
      <option value="4">第四級毒品(管制原料)</option>
    </select>
    <input class="aj-input" id="aj-weight" type="number" min="0" step="0.1"
           placeholder="純質淨重 g (選填)"
           data-hint data-tip="持有第一級純質淨重 ≥10 公克 → §11Ⅴ 加重;持有第二級 ≥20 公克 → §11Ⅵ 加重。"
           style="width: 12em">
  </div>
</div>

<div class="aj-section">
  <div class="aj-section-label">行為態樣</div>
  <div class="aj-chip-row">
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §10:施用第一級 6 月-5 年、施用第二級 3 年以下。"><input type="radio" name="aj-beh" value="施用" checked><span>施用</span></label>
    <label class="aj-chip" data-hint data-tip="§11Ⅰ-Ⅳ:持有 1-3 年以下;§11Ⅴ/Ⅵ 純質淨重達 10g(第一級)/20g(第二級)→ 加重 1-7 年 / 6 月-5 年。"><input type="radio" name="aj-beh" value="持有"><span>持有</span></label>
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §8:轉讓第一級 1-7 年、第二級 6 月-5 年、第三級 3 年以下、第四級 1 年以下。"><input type="radio" name="aj-beh" value="轉讓"><span>轉讓</span></label>
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §4Ⅰ-Ⅳ:販賣第一級 死刑/無期/15+ 年、第二級 無期/10+ 年、第三級 7+ 年、第四級 5-12 年。"><input type="radio" name="aj-beh" value="販賣"><span>販賣</span></label>
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §5:意圖販賣而持有,比照販賣降一檔。"><input type="radio" name="aj-beh" value="意圖販賣而持有"><span>意圖販賣而持有</span></label>
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §4Ⅰ-Ⅳ:運輸罪刑度與販賣相同。"><input type="radio" name="aj-beh" value="運輸"><span>運輸</span></label>
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §4Ⅰ-Ⅳ:製造罪刑度與販賣相同。"><input type="radio" name="aj-beh" value="製造"><span>製造</span></label>
  </div>
</div>

<div class="aj-section">
  <div class="aj-section-label">減刑事由(滑鼠移上去看法條)</div>
  <div class="aj-chip-row">
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §17 第 1 項:犯第 4 條至第 8 條之罪,供出毒品來源因而查獲其他正犯或共犯者,減輕或免除其刑。"><input type="checkbox" id="aj-17_1"><span>§17Ⅰ 供出來源</span></label>
    <label class="aj-chip" data-hint data-tip="毒品危害防制條例 §17 第 2 項:犯第 4 條至第 8 條之罪,於偵查及審判中均自白者,減輕其刑。"><input type="checkbox" id="aj-17_2"><span>§17Ⅱ 偵審均自白</span></label>
    <label class="aj-chip" data-hint data-tip="刑法 §59:犯罪之情狀顯可憫恕,認科以最低度刑仍嫌過重者,得酌量減輕其刑。"><input type="checkbox" id="aj-59"><span>§59 酌減</span></label>
    <label class="aj-chip" data-hint data-tip="刑法 §62:對於未發覺之罪自首而受裁判者,得減輕其刑。"><input type="checkbox" id="aj-62"><span>§62 自首</span></label>
    <label class="aj-chip" data-hint data-tip="刑法 §25 第 2 項:未遂犯之處罰,得按既遂犯之刑減輕之。"><input type="checkbox" id="aj-attempt"><span>§25Ⅱ 未遂</span></label>
  </div>
</div>

<div class="aj-section">
  <div class="aj-section-label">加重 / 程序</div>
  <div class="aj-chip-row">
    <label class="aj-chip" data-hint data-tip="刑法 §47:受徒刑之執行完畢、或一部之執行而赦免後,五年以內故意再犯有期徒刑以上之罪者,為累犯,加重本刑至二分之一。"><input type="checkbox" id="aj-recid"><span>§47 累犯(加重)</span></label>
    <label class="aj-chip" data-hint data-tip="簡易判決最高僅得宣告 6 月以下得易科罰金之刑(刑事訴訟法 §449),若超過則隱含有減刑事由,本模型對下限自動 ½。"><input type="checkbox" id="aj-summary"><span>簡易判決</span></label>
  </div>
</div>

<div class="aj-output" id="aj-output">
  <div id="aj-result">調整上方選項…</div>
</div>

<div class="aj-warn">
本工具<b>僅供教育與一般參考</b>。法定刑度區間 100% 依現行法律計算;「同類案件中位數」來自 2018-01 ~ 2026-02 北部 5 院 4.9 萬件實際判決,但個案差異(共犯、和解、犯後態度、§57 主觀因子)無法在此呈現。<b>不可</b>作為法律建議、判決依據、或律師代理之替代。
</div>

</div>

<p class="footer-note">
本研究使用司法院開放資料平臺釋出之裁判書資料。
研究方法、資料處理流程、評估指標皆公開於 GitHub 倉庫，可供同儕審查。
</p>

<small>Last updated: 2026-05-17</small>

<script>
const baseColor = '#0969da';
const accentColor = '#d18616';
const mutedColor = '#57606a';
const lightBg = '#f6f8fa';

Chart.defaults.font.family = '"Helvetica Neue", "PingFang TC", "Microsoft JhengHei", sans-serif';
Chart.defaults.color = '#24292f';

new Chart(document.getElementById('chart-extract'), {
  type: 'bar',
  data: {
    labels: ['§17 Ⅰ\n供出查獲', '§17 Ⅱ\n偵審自白', '§59\n酌減', '累犯\n認定', '易科罰金',
              '行為類型\n(Jaccard)', '毒品級別\n(Jaccard)'],
    datasets: [{
      label: '抽取準確率（F1 / Jaccard）',
      data: [1.00, 0.93, 1.00, 0.99, 0.99, 0.95, 0.97],
      backgroundColor: baseColor,
      borderRadius: 4,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      title: { display: true, text: '圖一：判決書關鍵欄位自動辨識準確率（n=100）', font: { size: 14 } },
      legend: { display: false }
    },
    scales: {
      y: { min: 0.7, max: 1.02, ticks: { callback: v => v.toFixed(2) } }
    }
  }
});

new Chart(document.getElementById('chart-mae'), {
  type: 'bar',
  data: {
    labels: ['全用中位數預測', '全用平均數預測', 'XGBoost(raw 預測)', 'XGBoost + 法定刑度 clip'],
    datasets: [{
      label: '平均絕對誤差(月)',
      data: [6.00, 7.50, 3.00, 2.85],
      backgroundColor: [mutedColor, mutedColor, '#7e9bbf', baseColor],
      borderRadius: 4,
    }]
  },
  options: {
    responsive: true, indexAxis: 'y',
    plugins: {
      title: { display: true, text: '圖二:模型預測誤差與基線比較(walk-forward 5 折,pooled n=51,940,北部 5 院)', font: { size: 14 } },
      legend: { display: false }
    },
    scales: { x: { title: { display: true, text: '月(越小越準)' } } }
  }
});

new Chart(document.getElementById('chart-by-court'), {
  type: 'bar',
  data: {
    labels: ['基隆 KL\n(n=4,484)', '新北 PC\n(n=18,246)', '士林 SL\n(n=4,590)',
              '臺北 TP\n(n=7,847)', '桃園 TY\n(n=16,773)'],
    datasets: [{
      label: '平均絕對誤差(月)',
      data: [2.06, 2.52, 2.65, 2.74, 3.53],
      backgroundColor: [baseColor, '#56b870', '#56b870', '#56b870', '#d18616'],
      borderRadius: 4,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      title: { display: true, text: '圖二之一:同一個 5 院聯訓 model 在各院 test 列上的 MAE — 基隆反而最低', font: { size: 14 } },
      legend: { display: false }
    },
    scales: {
      y: { beginAtZero: true,
           title: { display: true, text: 'MAE(月)' } }
    }
  }
});

new Chart(document.getElementById('chart-quantile'), {
  type: 'bar',
  data: {
    labels: ['p25 pinball', 'p50 pinball (中位數)', 'p75 pinball', '[p25,p75] 覆蓋率 (%)'],
    datasets: [{
      label: '分位數迴歸頭表現',
      data: [1.03, 1.43, 1.31, 49.7],
      backgroundColor: ['#56b870', baseColor, '#56b870', accentColor],
      borderRadius: 4,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      title: { display: true, text: '圖二之三:分位數迴歸頭——pinball loss(月)與 CQR 校正後的區間覆蓋率', font: { size: 14 } },
      legend: { display: false }
    },
    scales: {
      y: { beginAtZero: true,
           title: { display: true, text: 'pinball loss(前三柱,月) / 覆蓋率(最後一柱,%)' } }
    }
  }
});

new Chart(document.getElementById('chart-by-crime'), {
  type: 'bar',
  data: {
    labels: ['施用 (n=25,367)', '持有 (n=18,687)', '轉讓 (n=643)', '販賣 (n=5,936)',
              '意圖販賣而持有 (n=119)', '製造 (n=264)', '運輸 (n=879)'],
    datasets: [{
      label: '平均絕對誤差(月)',
      data: [1.25, 1.17, 2.80, 10.29, 12.17, 17.76, 28.76],
      backgroundColor: ['#56b870', '#56b870', '#56b870', '#d18616',
                        '#d18616', '#cc4128', '#cc4128'],
      borderRadius: 4,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      title: { display: true, text: '圖三:各主要罪名類型的預測誤差(北部 5 院 pooled walk-forward)', font: { size: 14 } },
      legend: { display: false }
    },
    scales: {
      y: { title: { display: true, text: '預測誤差(月)' }, beginAtZero: true }
    }
  }
});

new Chart(document.getElementById('chart-importance'), {
  type: 'bar',
  data: {
    labels: ['販賣毒品(b_販賣)', '得易科罰金', '運輸毒品', '第四級毒品', '刑法 §59 酌減',
              '未遂(§25Ⅱ)', '數罪併罰應執行刑', '持有毒品', '第一級毒品', '第三級毒品',
              '§17Ⅱ 偵審自白', '§57 品行(aggravating)', '§57 損害(neutral)', '§62 自首', '施用毒品'],
    datasets: [{
      label: '特徵重要性(gain)',
      data: [118524, 27348, 21747, 20099, 18471, 14678, 12630, 9303, 8735, 6937, 5853, 5575, 4122, 3928, 3905],
      backgroundColor: baseColor,
      borderRadius: 4,
    }]
  },
  options: {
    responsive: true, indexAxis: 'y',
    plugins: {
      title: { display: true, text: '圖四:影響量刑預測的關鍵因素(前 15 名,5 院 model)', font: { size: 14 } },
      legend: { display: false }
    },
    scales: {
      x: { title: { display: true, text: 'gain importance (XGBoost 內部分數)' } }
    }
  }
});

// --- Interactive sentencing calculator (互動量刑試算) ----------------------
// Pure-JS port of rules.py: statutory penalty table + reduction compounding +
// §11Ⅴ/Ⅵ 持有加重 + 累犯 ceiling raise. Buckets JSON gives median sentence
// per (court, behavior, level, flags) so the page can show "actual judgments
// in similar cases" without running the full XGBoost in the browser.

const MAX_FIXED_TERM = 12 * 30;   // 30 年
const PENALTY_TABLE = {
  // 毒品條例 §4 Ⅰ-Ⅳ:販賣/運輸/製造
  "販賣|1":  {lo: 15*12, hi: MAX_FIXED_TERM, life: true, capital: true},
  "販賣|2":  {lo: 10*12, hi: MAX_FIXED_TERM, life: true},
  "販賣|3":  {lo:  7*12, hi: MAX_FIXED_TERM},
  "販賣|4":  {lo:  5*12, hi: 12*12},
  "運輸|1":  {lo: 15*12, hi: MAX_FIXED_TERM, life: true, capital: true},
  "運輸|2":  {lo: 10*12, hi: MAX_FIXED_TERM, life: true},
  "運輸|3":  {lo:  7*12, hi: MAX_FIXED_TERM},
  "運輸|4":  {lo:  5*12, hi: 12*12},
  "製造|1":  {lo: 15*12, hi: MAX_FIXED_TERM, life: true, capital: true},
  "製造|2":  {lo: 10*12, hi: MAX_FIXED_TERM, life: true},
  "製造|3":  {lo:  7*12, hi: MAX_FIXED_TERM},
  "製造|4":  {lo:  5*12, hi: 12*12},
  // §5 意圖販賣而持有
  "意圖販賣而持有|1": {lo: 10*12, hi: MAX_FIXED_TERM, life: true},
  "意圖販賣而持有|2": {lo:  5*12, hi: 12*12},
  "意圖販賣而持有|3": {lo:  3*12, hi: 10*12},
  "意圖販賣而持有|4": {lo:  1*12, hi:  7*12},
  // §8 轉讓
  "轉讓|1": {lo: 1*12, hi: 7*12},
  "轉讓|2": {lo: 6,    hi: 5*12},
  "轉讓|3": {lo: 0,    hi: 3*12},
  "轉讓|4": {lo: 0,    hi: 1*12},
  // §10 施用
  "施用|1": {lo: 6, hi: 5*12},
  "施用|2": {lo: 0, hi: 3*12},
  // §11Ⅰ-Ⅳ 持有(輕量)
  "持有|1": {lo: 0, hi: 3*12},
  "持有|2": {lo: 0, hi: 2*12},
  "持有|3": {lo: 0, hi: 1*12},
  "持有|4": {lo: 0, hi: 1*12},
};
// §11Ⅴ/Ⅵ 持有加重(純質淨重 ≥ threshold 自動套用)
const HOLD_ENH = {
  "1": {threshold_g: 10, lo: 1*12, hi: 7*12},   // §11Ⅴ
  "2": {threshold_g: 20, lo:    6, hi: 5*12},   // §11Ⅵ
};

function baseRange(behavior, level, weight_g) {
  if (behavior === "持有" && weight_g != null) {
    const enh = HOLD_ENH[String(level)];
    if (enh && weight_g >= enh.threshold_g) {
      return {lo: enh.lo, hi: enh.hi, life: false, capital: false};
    }
  }
  const base = PENALTY_TABLE[`${behavior}|${level}`];
  return base ? {...base} : null;
}

function applyReductions(base, flags) {
  let {lo, hi, life, capital} = base;
  let factor = 1.0;
  // §66/§70:每個半減 flag 把上下限同乘 ½ 並 compound
  for (const k of ["art17_1", "art17_2", "art59", "self_surrender", "attempt"]) {
    if (flags[k]) factor *= 0.5;
  }
  lo *= factor;
  hi *= factor;
  if (flags.recidivism) hi = Math.min(hi * 1.5, MAX_FIXED_TERM);
  // 任一減刑都剝除無期/死刑可能
  const anyReduction = ["art17_1","art17_2","art59","self_surrender","attempt"]
                         .some(k => flags[k]);
  if (anyReduction) { life = false; capital = false; }
  // 簡易判決對下限再 ½(§59 隱性減刑常見於簡易判決附件)
  if (flags.summary && lo > 0) lo *= 0.5;
  return {lo, hi, life, capital};
}

function fmtMonths(m) {
  if (m == null || isNaN(m)) return "—";
  if (m <= 0) return "0";
  if (m < 1) {
    // 0 < m < 1 月,通常是 §59 半減後 ½ 月之類的怪數字,以日表示
    return `${(m * 30).toFixed(0)} 日`;
  }
  const yrs = Math.floor(m / 12);
  const rem = m - yrs * 12;
  if (yrs === 0) return `${m.toFixed(1)} 月`.replace(".0", "");
  if (rem < 0.5) return `${yrs} 年`;
  return `${yrs} 年 ${rem.toFixed(0)} 月`;
}

function fmtRange(lo, hi) {
  // 下限 0 → 法律寫法是「X 以下」,不應顯示「0 – X」(誤導且 0 日無意義)
  // 法律上「X 以下」隱含可宣告拘役 / 罰金 / 緩刑,沒有最低刑期
  if (lo <= 0 && hi > 0) {
    return `<span class="aj-range">${fmtMonths(hi)}</span><span class="aj-meta" style="margin-left:8px">以下(得拘役 / 罰金)</span>`;
  }
  if (lo > 0 && hi > 0) {
    return `<span class="aj-range">${fmtMonths(lo)} 以上 · ${fmtMonths(hi)} 以下</span>`;
  }
  return `<span class="aj-range">—</span>`;
}

let BUCKETS = null;

function readForm() {
  const flags = {
    art17_1: document.getElementById("aj-17_1").checked,
    art17_2: document.getElementById("aj-17_2").checked,
    art59:   document.getElementById("aj-59").checked,
    self_surrender: document.getElementById("aj-62").checked,
    attempt: document.getElementById("aj-attempt").checked,
    recidivism: document.getElementById("aj-recid").checked,
    summary: document.getElementById("aj-summary").checked,
  };
  const wt = parseFloat(document.getElementById("aj-weight").value);
  return {
    court: document.getElementById("aj-court").value,
    level: parseInt(document.getElementById("aj-level").value, 10),
    behavior: document.querySelector('input[name="aj-beh"]:checked').value,
    weight_g: isNaN(wt) ? null : wt,
    flags,
  };
}

function bucketKey(s) {
  const f = s.flags;
  const flagStr = [f.art17_1, f.art17_2, f.art59, f.self_surrender,
                   f.attempt, f.summary, f.recidivism]
                   .map(b => b ? "1" : "0").join("");
  return `${s.court}|${s.behavior}|${s.level}|${flagStr}`;
}

function lookupBucket(s) {
  if (!BUCKETS) return null;
  const exact = BUCKETS[bucketKey(s)];
  if (exact) return {...exact, scope: "完全相同特徵組合"};
  // Fallback A:同案件特徵、忽略 summary + recidivism
  const f2 = {...s.flags, summary: false, recidivism: false};
  const k2 = bucketKey({...s, flags: f2});
  const e2 = BUCKETS[k2];
  if (e2) return {...e2, scope: "同類案件(忽略簡易判決/累犯)"};
  // Fallback B:行為 + 級數 + 法院,所有 reduction flag 都不限
  for (const k of Object.keys(BUCKETS)) {
    const [c, b, lv] = k.split("|");
    if (c === s.court && b === s.behavior && lv === String(s.level)) {
      // 累加 — 但不疊加,挑第一個就好(目的是 fallback,不求準確)
      return {...BUCKETS[k], scope: "粗略 fallback(同院同罪同級數,不問減刑)"};
    }
  }
  return null;
}

function render() {
  const s = readForm();
  const base = baseRange(s.behavior, s.level, s.weight_g);
  const out = document.getElementById("aj-result");
  if (!base) {
    out.innerHTML = `<i>此 (行為, 級數) 組合不在法定刑度表中</i>`;
    return;
  }
  const reduced = applyReductions(base, s.flags);
  const enhanced = (s.behavior === "持有" && s.weight_g != null &&
                    HOLD_ENH[String(s.level)] &&
                    s.weight_g >= HOLD_ENH[String(s.level)].threshold_g);
  const enhTag = enhanced
                  ? `<span class="aj-tag">§11Ⅴ/Ⅵ 加重</span>` : "";
  let lifeNote = "";
  if (reduced.capital) lifeNote = ` <span class="aj-tag">含死刑</span>`;
  else if (reduced.life) lifeNote = ` <span class="aj-tag">含無期徒刑</span>`;

  let bucketHtml = "";
  const b = lookupBucket(s);
  if (b) {
    bucketHtml = `
      <div class="aj-row">
        <div class="aj-output-label">預測刑期 — 北部 5 院近年同類判決 · n=${b.n} · ${b.scope}</div>
        <div class="aj-quantiles">
          <div class="aj-q"><span class="aj-q-label">較輕 25%</span><span class="aj-q-val">${fmtMonths(b.p25)}</span></div>
          <div class="aj-q"><span class="aj-q-label">中位數</span><span class="aj-q-val" style="color:#0969da;font-size:1.6em">${fmtMonths(b.p50)}</span></div>
          <div class="aj-q"><span class="aj-q-label">較重 25%</span><span class="aj-q-val">${fmtMonths(b.p75)}</span></div>
        </div>
      </div>`;
  } else {
    bucketHtml = `
      <div class="aj-row">
        <div class="aj-output-label">預測刑期</div>
        <div class="aj-meta">該特徵組合在資料集中 &lt; 3 件,樣本過稀,僅顯示法律邊界。</div>
      </div>`;
  }
  out.innerHTML = `
    ${bucketHtml}
    <div class="aj-bucket">
      <div class="aj-row">
        <div class="aj-output-label">法定刑度上下限 ${enhTag}</div>
        ${fmtRange(base.lo, base.hi)}
      </div>
      <div class="aj-row">
        <div class="aj-output-label">套用減刑/加重後合法範圍${lifeNote}</div>
        ${fmtRange(reduced.lo, reduced.hi)}
      </div>
    </div>
  `;
}

function initTooltips() {
  // Shared single tooltip node, repositioned on hover. Native title= has a
  // ~1s delay and ignores nested labels/spans inconsistently — this custom
  // version fires immediately and works on any element with [data-tip].
  const tip = document.createElement("div");
  tip.className = "aj-tooltip";
  document.body.appendChild(tip);

  function show(ev) {
    const t = ev.currentTarget;
    const text = t.getAttribute("data-tip");
    if (!text) return;
    tip.textContent = text;
    tip.classList.add("visible");
    const rect = t.getBoundingClientRect();
    const tipRect = tip.getBoundingClientRect();
    // Place below the element, clamped to viewport edges.
    let left = rect.left;
    if (left + tipRect.width > window.innerWidth - 12) {
      left = window.innerWidth - tipRect.width - 12;
    }
    if (left < 12) left = 12;
    let top = rect.bottom + 8;
    if (top + tipRect.height > window.innerHeight - 12) {
      top = rect.top - tipRect.height - 8;
    }
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function hide() { tip.classList.remove("visible"); }
  document.querySelectorAll("[data-tip]").forEach(el => {
    el.addEventListener("mouseenter", show);
    el.addEventListener("mouseleave", hide);
    el.addEventListener("focus", show);
    el.addEventListener("blur", hide);
  });
}

async function initCalculator() {
  try {
    const r = await fetch("./assets/sentence_buckets.json", {cache: "force-cache"});
    if (r.ok) {
      const payload = await r.json();
      BUCKETS = payload.buckets || {};
    }
  } catch (e) {
    console.warn("bucket lookup unavailable:", e);
  }
  document.getElementById("aj-form").addEventListener("change", render);
  document.getElementById("aj-form").addEventListener("input", render);
  initTooltips();
  render();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initCalculator);
} else {
  initCalculator();
}
</script>
