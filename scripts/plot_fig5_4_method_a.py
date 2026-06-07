"""
生成图 5-4：空间过滤范围对 Recall@10 与 P95 时延的影响。

数据来源: reports/method_a_results.json
输出:    reports/fig5_4_method_a.png  +  reports/fig5_4_method_a.pdf

用法:
    python scripts/plot_fig5_4_method_a.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 直接注册 Noto CJK 字体文件，绕过缓存查找
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
_cjk_font = None
for _p in _CJK_CANDIDATES:
    if Path(_p).exists():
        fm.fontManager.addfont(_p)
        _cjk_font = fm.FontProperties(fname=_p).get_name()
        break

matplotlib.rcParams.update({
    "font.family":      [_cjk_font, "DejaVu Serif"] if _cjk_font else ["DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── 数据 ──────────────────────────────────────────────────────────
results_path = Path(__file__).resolve().parent.parent / "reports" / "method_a_results.json"
data = json.loads(results_path.read_text())

baseline_recall = data["baseline"]["Recall@10"]
baseline_p95    = data["baseline"]["latency_p95_ms"]

# bbox sweep 按 half_deg 排序（小 → 大）
sweep_order = ["half_0.5deg", "half_1.0deg", "half_2.0deg", "half_5.0deg", "half_10.0deg"]
half_degs    = [0.5,    1.0,    2.0,    5.0,    10.0]
coverages_pct= [0.82,   1.64,   3.93,   15.59,  43.17]   # avg_library_coverage × 100
recalls      = [data["bbox_sweep"][k]["Recall@10"]       for k in sweep_order]
p95s         = [data["bbox_sweep"][k]["latency_p95_ms"]  for k in sweep_order]

# 把无过滤基线作为最后一个点（coverage=100%）
all_cov     = coverages_pct + [100.0]
all_recalls = recalls        + [baseline_recall]
all_p95s    = p95s           + [baseline_p95]

# 用于 x 轴的 log10 坐标
log_x = np.log10(all_cov)

# ── 绘图 ──────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
ax2 = ax1.twinx()

COLOR_RECALL  = "#1f77b4"   # 蓝
COLOR_LATENCY = "#d62728"   # 红

# Recall@10
line1, = ax1.plot(log_x, all_recalls,
                  color=COLOR_RECALL, marker="o", markersize=5,
                  linewidth=1.5, label="Recall@10", zorder=3)
ax1.axhline(baseline_recall, color=COLOR_RECALL, linewidth=0.8,
            linestyle="--", alpha=0.6, label=f"基线 Recall@10 = {baseline_recall:.4f}")

# P95 延迟
line2, = ax2.plot(log_x, all_p95s,
                  color=COLOR_LATENCY, marker="s", markersize=5,
                  linewidth=1.5, linestyle="-.", label="P95 延迟 (ms)", zorder=3)
ax2.axhline(baseline_p95, color=COLOR_LATENCY, linewidth=0.8,
            linestyle=":", alpha=0.6, label=f"基线 P95 = {baseline_p95:.1f} ms")

# 标注"非单调回升"区域
ax1.annotate(
    "非单调\n回升",
    xy=(log_x[-2], all_recalls[-2]),      # half_10.0deg 点
    xytext=(log_x[-2] - 0.55, all_recalls[-2] + 0.012),
    fontsize=7.5, color=COLOR_RECALL,
    arrowprops=dict(arrowstyle="->", color=COLOR_RECALL, lw=0.8),
)

# ── x 轴刻度与标签 ────────────────────────────────────────────────
tick_covs   = [0.82, 1.64, 3.93, 15.59, 43.17, 100.0]
tick_labels = [
    "0.82%\n(±0.5°)",
    "1.64%\n(±1°)",
    "3.93%\n(±2°)",
    "15.6%\n(±5°)",
    "43.2%\n(±10°)",
    "100%\n(无过滤)",
]
ax1.set_xticks(np.log10(tick_covs))
ax1.set_xticklabels(tick_labels, fontsize=7)

# ── 轴范围与标签 ──────────────────────────────────────────────────
ax1.set_xlim(np.log10(0.5), np.log10(200))
ax1.set_ylim(0.73, 0.87)
ax2.set_ylim(0, 28)

ax1.set_xlabel("库覆盖率（%）/ bbox 半边长（度）", labelpad=4)
ax1.set_ylabel("Recall@10", color=COLOR_RECALL, labelpad=4)
ax2.set_ylabel("P95 检索延迟（ms）", color=COLOR_LATENCY, labelpad=4)

ax1.tick_params(axis="y", labelcolor=COLOR_RECALL)
ax2.tick_params(axis="y", labelcolor=COLOR_LATENCY)

# ── 图例 ──────────────────────────────────────────────────────────
handles = [line1, line2]
labels  = ["Recall@10", "P95 延迟 (ms)"]
ax1.legend(handles, labels, loc="lower left", framealpha=0.85, edgecolor="gray")

ax1.set_title("图 5-4  空间过滤范围对 Recall@10 与 P95 时延的影响\n"
              "（EuroSAT · 以查询坐标为中心的正方形 bbox · ef=128）",
              pad=6)

ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax1.set_axisbelow(True)

plt.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "reports"
fig.savefig(out_dir / "fig5_4_method_a.png", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "fig5_4_method_a.pdf", bbox_inches="tight")
print("saved: fig5_4_method_a.png / .pdf")
plt.show()
