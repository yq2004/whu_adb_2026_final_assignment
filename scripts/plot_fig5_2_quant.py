"""
生成图 5-2：向量量化策略对检索质量与存储成本的影响。

数据来源: reports/quant_results_int8AlsoWithRescoring.json
          (scripts/05_quant_compare.py 产出，INT8 与 Binary 均启用 Rescoring)
输出:    reports/fig5_2_quant.png  +  .pdf

双子图:
  左图: FP32 / INT8 / Binary 的 mAP@10 柱状图
        （y 轴放大至差异区间，凸显质量几乎无损）
  右图: 内存占用（柱，左轴）与 P95 时延（点线，右轴）对比
        （凸显内存压缩比的戏剧性收益）

用法:
    python scripts/plot_fig5_2_quant.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

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
    "font.family":      [_cjk_font, "DejaVu Sans"] if _cjk_font else ["DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── 数据 ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
data = json.loads(
    (root / "reports" / "quant_results_int8AlsoWithRescoring.json").read_text()
)

variants  = ["FP32", "INT8", "Binary"]
labels    = ["FP32\n(无量化)", "INT8\n(标量量化)", "Binary\n(1-bit)"]
colors    = ["#4878d0", "#ee854a", "#6acc65"]

map10  = [data[v]["mAP@10"]             for v in variants]
mem    = [data[v]["memory_mb_estimated"] for v in variants]
p95    = [data[v]["latency_p95_ms"]      for v in variants]

x = np.arange(len(variants))
bar_w = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
fig.subplots_adjust(wspace=0.38)

# ── 左图：mAP@10 柱状图 ───────────────────────────────────────────
bars1 = ax1.bar(x, map10, width=bar_w, color=colors,
                edgecolor="white", linewidth=0.8, zorder=3)

# 数值标注（柱顶）
for bar, val in zip(bars1, map10):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.0005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# FP32 水平参考线
ax1.axhline(map10[0], color="#4878d0", linewidth=0.8,
            linestyle="--", alpha=0.6, zorder=2)

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0.870, 0.910)
ax1.set_ylabel("mAP@10")
ax1.set_title("(a)  检索质量（mAP@10）")
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax1.set_axisbelow(True)

# y 轴放大到差异区间，加括号说明
ax1.annotate("← 纵轴起点 0.870\n    (非零刻度)", xy=(0.01, 0.02),
             xycoords="axes fraction", fontsize=7, color="#666666")

# ── 右图：内存（柱，左轴）+ P95 时延（点线，右轴）──────────────────
ax2r = ax2.twinx()

bars2 = ax2.bar(x, mem, width=bar_w, color=colors,
                edgecolor="white", linewidth=0.8, zorder=3, label="内存（MB）")

# 内存数值 + 压缩比
compress = [1.0, 4.0, 32.0]
for bar, val, ratio in zip(bars2, mem, compress):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + 1.0,
             f"{val:.1f} MB\n(1/{ratio:.0f}×)",
             ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# P95 时延折线
ax2r.plot(x, p95, color="#d62728", marker="D", markersize=6,
          linewidth=1.6, linestyle="-.", zorder=4, label="P95 延迟 (ms)")
for xi, yi in zip(x, p95):
    ax2r.text(xi, yi + 0.06, f"{yi:.2f}", ha="center", va="bottom",
              fontsize=7.5, color="#d62728")

ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylim(0, 105)
ax2.set_ylabel("估算内存（MB）", labelpad=4)
ax2r.set_ylim(3.0, 7.5)
ax2r.set_ylabel("P95 检索延迟（ms）", color="#d62728", labelpad=4)
ax2r.tick_params(axis="y", labelcolor="#d62728")

ax2.set_title("(b)  内存占用与 P95 时延")
ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax2.set_axisbelow(True)

# 合并图例
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2r.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper right",
           framealpha=0.85, edgecolor="gray", fontsize=7.5)

fig.suptitle("图 5-2  向量量化策略对检索质量与存储成本的影响（EuroSAT · ef=128 · Rescoring 开启）",
             fontsize=9.5, y=1.01)

out = root / "reports"
fig.savefig(out / "fig5_2_quant.png", dpi=200, bbox_inches="tight")
fig.savefig(out / "fig5_2_quant.pdf", bbox_inches="tight")
print("saved: fig5_2_quant.png / .pdf")
plt.show()
