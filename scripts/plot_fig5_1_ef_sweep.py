"""
生成图 5-1：HNSW ef 参数对检索质量与 P 时延的影响。

数据来源: reports/eval_results.json  (scripts/04_run_eval.py 产出)
输出:    reports/fig5_1_ef_sweep.png  +  .pdf

双子图:
  左图: Recall@10 / mAP@10 / NDCG@10 随 ef 的变化（展示质量平坦性）
  右图: P50 / P95 / P99 时延随 ef 的变化（展示延迟 U 形曲线）

用法:
    python scripts/plot_fig5_1_ef_sweep.py
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

# ── 数据 ──────────────────────────────────────────────────────────
root   = Path(__file__).resolve().parent.parent
data   = json.loads((root / "reports" / "eval_results.json").read_text())
sweep  = data["ef_sweep"]

ef_vals = [16, 32, 64, 96, 128, 192, 256]

def get(metric):
    return [sweep[str(ef)][metric] for ef in ef_vals]

recall = get("Recall@10")
map10  = get("mAP@10")
ndcg   = get("NDCG@10")
p50    = get("latency_p50_ms")
p95    = get("latency_p95_ms")
p99    = get("latency_p99_ms")

SWEET = 128   # 推荐 ef

# ── 颜色 ──────────────────────────────────────────────────────────
C_RECALL = "#1f77b4"
C_MAP    = "#ff7f0e"
C_NDCG   = "#2ca02c"
C_P50    = "#9467bd"
C_P95    = "#d62728"
C_P99    = "#8c564b"
C_SWEET  = "#888888"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
fig.subplots_adjust(wspace=0.32)

x = np.array(ef_vals)

# ── 左图：质量指标 ─────────────────────────────────────────────────
for vals, color, label in [
    (recall, C_RECALL, "Recall@10"),
    (map10,  C_MAP,    "mAP@10"),
    (ndcg,   C_NDCG,   "NDCG@10"),
]:
    ax1.plot(x, vals, marker="o", markersize=5, color=color,
             linewidth=1.6, label=label)

ax1.axvline(SWEET, color=C_SWEET, linewidth=1.0, linestyle="--",
            label=f"推荐 ef = {SWEET}")

ax1.set_xlabel("HNSW ef 参数")
ax1.set_ylabel("指标值")
ax1.set_xticks(ef_vals)
ax1.set_ylim(0.840, 0.940)
ax1.legend(loc="lower right", framealpha=0.85, edgecolor="gray")
ax1.set_title("(a)  检索质量随 ef 的变化（趋于饱和）")
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax1.set_axisbelow(True)

# ── 右图：延迟百分位 ───────────────────────────────────────────────
for vals, color, label, ls in [
    (p50, C_P50, "P50", "-"),
    (p95, C_P95, "P95", "-."),
    (p99, C_P99, "P99", ":"),
]:
    ax2.plot(x, vals, marker="s", markersize=4, color=color,
             linewidth=1.6, linestyle=ls, label=label)

ax2.axvline(SWEET, color=C_SWEET, linewidth=1.0, linestyle="--",
            label=f"推荐 ef = {SWEET}")

# 标注最低 P95 点
idx_min = int(np.argmin(p95))
ax2.annotate(f"P95 最低\n{p95[idx_min]:.2f} ms",
             xy=(ef_vals[idx_min], p95[idx_min]),
             xytext=(ef_vals[idx_min] + 22, p95[idx_min] + 0.5),
             fontsize=7.5, color=C_P95,
             arrowprops=dict(arrowstyle="->", color=C_P95, lw=0.8))

ax2.set_xlabel("HNSW ef 参数")
ax2.set_ylabel("检索延迟（ms）")
ax2.set_xticks(ef_vals)
ax2.set_ylim(3.5, 10.0)
ax2.legend(loc="upper left", framealpha=0.85, edgecolor="gray")
ax2.set_title("(b)  检索延迟随 ef 的变化（U 形曲线）")
ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax2.set_axisbelow(True)

fig.suptitle("图 5-1  HNSW ef 参数对检索质量与时延的影响（EuroSAT · INT8 + Rescoring）",
             fontsize=9.5, y=1.01)

out = root / "reports"
fig.savefig(out / "fig5_1_ef_sweep.png", dpi=200, bbox_inches="tight")
fig.savefig(out / "fig5_1_ef_sweep.pdf", bbox_inches="tight")
print("saved: fig5_1_ef_sweep.png / .pdf")
plt.show()
