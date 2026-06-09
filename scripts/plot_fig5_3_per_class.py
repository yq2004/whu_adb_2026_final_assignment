"""
生成图 5-3：分类别检索 mAP@20 分布。

数据来源: reports/per_class_results.json  (scripts/06_per_class_eval.py 产出)
输出:    reports/fig5_3_per_class.png  +  .pdf

水平柱状图（按 mAP@20 降序），含:
  - 误差棒（±std）
  - 宏平均参考竖线
  - 柱尾标注 mAP@20 数值与查询样本量 (n)

用法:
    python scripts/plot_fig5_3_per_class.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    "ytick.labelsize":  8.5,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── 数据 ──────────────────────────────────────────────────────────
root   = Path(__file__).resolve().parent.parent
data   = json.loads((root / "reports" / "per_class_results.json").read_text())

macro_avg = data["overall"]["mAP@20_macro"]
micro_avg = data["overall"]["mAP@20_micro"]

pc = data["per_class"]
# 按 mAP@20 降序排列
sorted_classes = sorted(pc.items(), key=lambda kv: kv[1]["mAP@20"], reverse=True)

classes  = [k for k, _ in sorted_classes]
map_vals = [v["mAP@20"]  for _, v in sorted_classes]
stds     = [v["std"]     for _, v in sorted_classes]
n_query  = [v["n_query"] for _, v in sorted_classes]

# 中英文类别名对照（便于论文阅读）
CLASS_CN = {
    "SeaLake":              "海湖 SeaLake",
    "Forest":               "森林 Forest",
    "Residential":          "居住区 Residential",
    "River":                "河流 River",
    "Industrial":           "工业区 Industrial",
    "PermanentCrop":        "多年生作物 PermanentCrop",
    "HerbaceousVegetation": "草本植被 HerbVeg.",
    "Pasture":              "牧场 Pasture",
    "AnnualCrop":           "一年生作物 AnnualCrop",
    "Highway":              "公路 Highway",
}
y_labels = [CLASS_CN.get(c, c) for c in classes]
y = np.arange(len(classes))

# ── 颜色（按 mAP 值映射，Highway 最差设为红色调）──────────────────
cmap   = plt.get_cmap("RdYlGn")
norm   = mcolors.Normalize(vmin=min(map_vals) - 0.05, vmax=1.0)
colors = [cmap(norm(v)) for v in map_vals]

# ── 绘图 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.2))

bars = ax.barh(y, map_vals, xerr=stds, height=0.6,
               color=colors, edgecolor="white", linewidth=0.6,
               error_kw=dict(elinewidth=0.9, capsize=3, ecolor="#555555"),
               zorder=3)

# 宏平均参考竖线
ax.axvline(macro_avg, color="#333333", linewidth=1.2, linestyle="--",
           label=f"宏平均 mAP@20 = {macro_avg:.4f}", zorder=4)

# 柱右侧标注数值 + 查询量
for i, (val, n) in enumerate(zip(map_vals, n_query)):
    ax.text(val + 0.005, i, f"{val:.4f}  (n={n})",
            va="center", ha="left", fontsize=7.8)

# 特别标注 Highway（最低分异常值）
highway_idx = classes.index("Highway")
ax.annotate("最低类别（线性目标难区分）",
            xy=(map_vals[highway_idx], highway_idx),
            xytext=(map_vals[highway_idx] + 0.04, highway_idx + 1.2),
            fontsize=7.5, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

ax.set_yticks(y)
ax.set_yticklabels(y_labels)
ax.set_xlim(0.45, 1.08)
ax.set_xlabel("mAP@20（误差棒 = ±std）")
ax.set_title("图 5-3  分类别检索性能（mAP@20）— EuroSAT 10 类，按均值降序排列",
             pad=8)
ax.legend(loc="lower right", framealpha=0.85, edgecolor="gray")
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.invert_yaxis()   # 最高分在上

# 色标（说明颜色含义）
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("mAP@20", fontsize=8)

plt.tight_layout()

out = root / "reports"
fig.savefig(out / "fig5_3_per_class.png", dpi=200, bbox_inches="tight")
fig.savefig(out / "fig5_3_per_class.pdf", bbox_inches="tight")
print("saved: fig5_3_per_class.png / .pdf")
plt.show()
