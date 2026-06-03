# -*- coding: utf-8 -*-
"""產生 AiJudge 簡報用圖表。輸出到 docs/assets/ppt/。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "ppt")
os.makedirs(OUT, exist_ok=True)

# 配色
C_MAIN = "#2c6e9b"
C_ACC = "#d9822b"
C_GREY = "#b8c4cc"
C_GOOD = "#3f8f5b"
C_BAD = "#c0504d"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.normpath(p))


# 1. 五院資料量
def fig_court_counts():
    courts = ["新北", "桃園", "臺北", "士林", "基隆"]
    counts = [24947, 20750, 11324, 6155, 5809]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(courts, counts, color=[C_MAIN]*4 + [C_ACC])
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, c+300, f"{c:,}", ha="center", fontsize=10)
    ax.set_ylabel("判決件數")
    ax.set_title("圖1  北部 5 地方法院毒品案件樣本分布（總計 68,985 件，2018-01~2026-02）")
    ax.set_ylim(0, 28000)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "01_court_counts.png")


# 2. 基隆單院 vs 5院聯訓
def fig_keelung_transfer():
    labels = ["基隆單院\n直接訓練", "5 院聯訓\n（基隆 test 列）"]
    mae = [2.20, 1.79]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(labels, mae, color=[C_GREY, C_GOOD], width=0.55)
    for b, v in zip(bars, mae):
        ax.text(b.get_x()+b.get_width()/2, v+0.04, f"{v:.2f} 月", ha="center", fontsize=12, fontweight="bold")
    ax.annotate("改善 ~19%", xy=(1, 1.79), xytext=(0.5, 2.45),
                ha="center", fontsize=12, color=C_GOOD,
                arrowprops=dict(arrowstyle="->", color=C_GOOD))
    ax.set_ylabel("基隆案件 MAE（月）")
    ax.set_title("圖2  跨院聯合訓練反超單院：小樣本法院的增益")
    ax.set_ylim(0, 2.8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "02_keelung_transfer.png")


# 3. 改善路徑 MAE 2.85 -> 2.49
def fig_improvement_path():
    steps = ["baseline", "+個刑加總\n特徵", "+§17/§59\nreject", "p50改L1\nloss", "調參\nlr/rounds", "3-seed\nensemble"]
    mae = [2.85, 2.63, 2.62, 2.52, 2.51, 2.49]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(range(len(steps)), mae, marker="o", color=C_MAIN, linewidth=2, markersize=8)
    for i, v in enumerate(mae):
        ax.text(i, v+0.015, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=9)
    ax.set_ylabel("Pooled MAE（月）")
    ax.set_title("圖3  五項實驗的累積改善：MAE 2.85 → 2.49 月")
    ax.set_ylim(2.4, 2.92)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "03_improvement_path.png")


# 4. 模型 vs 基線
def fig_model_vs_baseline():
    labels = ["median\n基線", "XGBoost\np50"]
    mae = [6.00, 2.49]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, mae, color=[C_GREY, C_MAIN], width=0.5)
    for b, v in zip(bars, mae):
        ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f} 月", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pooled MAE（月，n=51,944）")
    ax.set_title("圖4  模型較中位數基線降低 59% 誤差")
    ax.set_ylim(0, 6.8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "04_model_vs_baseline.png")


# 5. Per-court MAE
def fig_per_court_mae():
    courts = ["基隆", "士林", "新北", "臺北", "桃園"]
    mae = [1.79, 2.20, 2.21, 2.44, 3.10]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(courts[::-1], mae[::-1], color=C_MAIN)
    for b, v in zip(bars, mae[::-1]):
        ax.text(v+0.04, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=10)
    ax.set_xlabel("MAE（月）")
    ax.set_title("圖5  各法院預測誤差：基隆最穩、桃園變異最大")
    ax.set_xlim(0, 3.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "05_per_court_mae.png")


# 6. Per-behavior MAE (輕罪 vs 重罪)
def fig_per_behavior_mae():
    beh = ["持有", "施用", "轉讓", "販賣", "意圖販賣\n而持有", "製造", "運輸"]
    mae = [1.06, 1.11, 2.31, 8.70, 11.36, 15.10, 26.07]
    colors = [C_GOOD, C_GOOD, C_GOOD, C_BAD, C_BAD, C_BAD, C_BAD]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(beh, mae, color=colors)
    for b, v in zip(bars, mae):
        ax.text(b.get_x()+b.get_width()/2, v+0.4, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("MAE（月）")
    ax.set_title("圖6  行為別誤差：輕罪（大宗）極準，重罪（稀少）難")
    ax.set_ylim(0, 28)
    ax.spines[["top", "right"]].set_visible(False)
    # 圖例
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_GOOD, label="輕罪 < 2.5 月"),
                       Patch(color=C_BAD, label="重罪（樣本稀少）")], fontsize=9)
    save(fig, "06_per_behavior_mae.png")


# 7. 法定刑度約束：越界率歸零
def fig_constraint():
    labels = ["ground-truth\n標籤", "raw 預測", "rule-clipped\n預測"]
    rate = [2.10, 3.4, 0.0]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, rate, color=[C_GREY, C_BAD, C_GOOD], width=0.55)
    for b, v in zip(bars, rate):
        ax.text(b.get_x()+b.get_width()/2, v+0.07, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("法定刑度越界率")
    ax.set_title("圖7  約束層保證：raw 3.4% → clip 後 0.00%")
    ax.set_ylim(0, 4.0)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "07_constraint.png")


# 8. 命中率
def fig_hit_rate():
    labels = ["±3 月\n命中率", "±6 月\n命中率", "R²"]
    vals = [87.8, 93.4, 71.7]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(labels, vals, color=[C_MAIN, C_MAIN, C_ACC], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%" if v < 100 else f"{v}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("百分比")
    ax.set_title("圖8  整體預測表現（R² 0.717）")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "08_hit_rate.png")


# 9. 抽取準確率
def fig_extraction():
    labels = ["量刑月數\nexact", "§17/§59\nF1", "毒品級數\nJaccard", "行為\nJaccard"]
    vals = [1.00, 0.965, 0.97, 0.95]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=C_GOOD, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("準確率 / F1 / Jaccard")
    ax.set_title("圖9  特徵抽取品質（人工標註 n=94~100 驗證）")
    ax.set_ylim(0, 1.12)
    ax.axhline(0.9, color=C_BAD, linestyle="--", alpha=0.6)
    ax.text(3.3, 0.91, "0.90 門檻", color=C_BAD, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "09_extraction.png")


if __name__ == "__main__":
    fig_court_counts()
    fig_keelung_transfer()
    fig_improvement_path()
    fig_model_vs_baseline()
    fig_per_court_mae()
    fig_per_behavior_mae()
    fig_constraint()
    fig_hit_rate()
    fig_extraction()
    print("\nAll charts written to", os.path.normpath(OUT))
