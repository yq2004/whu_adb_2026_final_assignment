"""
生成图 5-5：范式 B 加权重排参数扫描结果。

数据来源: reports/method_b_results.json
输出:    reports/fig5_5_method_b.png  +  reports/fig5_5_method_b.pdf

双子图:
  左图: mAP@10 随 w_geo 的变化（三条 decay_km 曲线 + 基线）
  右图: avg_sem_score（实线）与 avg_prox_score（虚线）随 w_geo 的变化

用法:
    python scripts/plot_fig5_5_method_b.py
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
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── 数据加载 ───────────────────────────────────────────────────────
results_path = Path(__file__).resolve().parent.parent / "reports" / "method_b_results.json"
data = json.loads(results_path.read_text())

baseline = data["baseline"]
sweep    = data["rerank_sweep"]

w_geos    = [0.1, 0.2, 0.4, 0.6, 0.8]
decay_kms = [10, 50, 200]

def get_series(metric: str, decay: int) -> list[float]:
    out = []
    for wg in w_geos:
        ws = round(1.0 - wg, 6)
        tag = f"wsem{ws}_wgeo{wg}_decay{decay}km"
        out.append(sweep[tag][metric])
    return out

map10_d10  = get_series("mAP@10", 10)
map10_d50  = get_series("mAP@10", 50)
map10_d200 = get_series("mAP@10", 200)

sem_d10  = get_series("avg_sem_score", 10)
sem_d50  = get_series("avg_sem_score", 50)
sem_d200 = get_series("avg_sem_score", 200)

prox_d10  = get_series("avg_prox_score", 10)
prox_d50  = get_series("avg_prox_score", 50)
prox_d200 = get_series("avg_prox_score", 200)

# ── 调色板 ────────────────────────────────────────────────────────
C10  = "#1f77b4"   # 蓝  – decay=10km
C50  = "#ff7f0e"   # 橙  – decay=50km
C200 = "#2ca02c"   # 绿  – decay=200km
CGRAY = "#888888"

# ── 双子图布局 ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
fig.subplots_adjust(wspace=0.32)

x = np.array(w_geos)

# ── 左图：mAP@10 ──────────────────────────────────────────────────
for vals, color, label in [
    (map10_d10,  C10,  "decay = 10 km"),
    (map10_d50,  C50,  "decay = 50 km"),
    (map10_d200, C200, "decay = 200 km"),
]:
    ax1.plot(x, vals, marker="o", markersize=5, color=color,
             linewidth=1.6, label=label)

ax1.axhline(baseline["mAP@10"], color=CGRAY, linewidth=1.0,
            linestyle="--", label=f"基线 mAP@10 = {baseline['mAP@10']:.4f}")

# 标注最佳点
best_val = map10_d10[0]
ax1.annotate(f"最优 {best_val:.4f}",
             xy=(x[0], best_val),
             xytext=(x[0] + 0.06, best_val + 0.0025),
             fontsize=7.5, color=C10,
             arrowprops=dict(arrowstyle="->", color=C10, lw=0.8))

ax1.set_xlabel("地理权重 $w_{geo}$（$w_{sem} = 1 - w_{geo}$）")
ax1.set_ylabel("mAP@10")
ax1.set_xticks(w_geos)
ax1.set_ylim(0.870, 0.935)
ax1.legend(loc="upper right", framealpha=0.85, edgecolor="gray")
ax1.set_title("(a)  mAP@10 随 $w_{geo}$ 与衰减半径的变化")
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

# ── 右图：avg_sem_score（实线）与 avg_prox_score（虚线）────────────
ax2r = ax2.twinx()

for sem_vals, prox_vals, color, decay in [
    (sem_d10,  prox_d10,  C10,  10),
    (sem_d50,  prox_d50,  C50,  50),
    (sem_d200, prox_d200, C200, 200),
]:
    l1, = ax2.plot(x, sem_vals, marker="o", markersize=4, color=color,
                   linewidth=1.5, label=f"sem  d={decay}km")
    ax2r.plot(x, prox_vals, marker="s", markersize=4, color=color,
              linewidth=1.5, linestyle="--")

# 基线语义得分参考线
ax2.axhline(baseline["avg_sem_score"], color=CGRAY, linewidth=1.0,
            linestyle=":", label=f"基线 sem = {baseline['avg_sem_score']:.4f}")

# 右轴仅用虚线标注 proxy 概念，加一条隐形 legend 条目
prox_dummy, = ax2r.plot([], [], color=CGRAY, linewidth=1.5,
                        linestyle="--", label="邻近度得分 prox（右轴）")

ax2.set_xlabel("地理权重 $w_{geo}$")
ax2.set_ylabel("平均语义相似度 avg_sem", color="#333333")
ax2r.set_ylabel("平均地理邻近度 avg_prox", color="#555555")
ax2.set_xticks(w_geos)
ax2.set_ylim(0.9810, 0.9875)
ax2r.set_ylim(0.0, 0.85)

# 合并图例
h1, l1 = ax2.get_legend_handles_labels()
ax2.legend(h1 + [prox_dummy],
           l1 + ["邻近度 prox（右轴，虚线）"],
           loc="lower left", framealpha=0.85, edgecolor="gray", fontsize=7.5)

ax2.set_title("(b)  语义相似度与地理邻近度随 $w_{geo}$ 的变化")
ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

fig.suptitle("图 5-5  范式 B 加权重排参数扫描结果（EuroSAT · candidate_factor = 5 · ef = 128）",
             fontsize=9.5, y=1.01)

out_dir = Path(__file__).resolve().parent.parent / "reports"
fig.savefig(out_dir / "fig5_5_method_b.png", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "fig5_5_method_b.pdf", bbox_inches="tight")
print("saved: fig5_5_method_b.png / .pdf")
plt.show()
