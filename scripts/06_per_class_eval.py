"""
Step 6:按类别统计 mAP@20。

对应论文 §5.3.2 类别级别的精度差异 + 图 5-3。

为每个查询: 取它的主标签(EuroSAT 单标签),按标签分组聚合 mAP@20,
最终输出 10 类各自的检索精度。

用法:
    python scripts/06_per_class_eval.py --config configs/config.yaml

输出:
    reports/per_class_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.qdrant_client import make_client, search
from src.evaluation.metrics import average_precision_at_k


def main(cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    reports_dir = Path(cfg["evaluation"]["output_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    query_embs = np.load(emb_dir / "queries_embeddings.npy")
    query_meta = pd.read_parquet(emb_dir / "queries_metadata.parquet")
    library_meta = pd.read_parquet(emb_dir / "library_metadata.parquet")

    query_labels = [list(x) for x in query_meta["labels"].tolist()]
    query_main_labels = query_meta["main_label"].tolist()
    library_labels_by_id = {row["patch_id"]: list(row["labels"])
                            for _, row in library_meta.iterrows()}

    client = make_client(cfg["qdrant"]["url"])
    collection = cfg["qdrant"]["collection"]
    ef = cfg["retrieval"]["ef_search"]
    K = 20

    # 按类别分组累加 AP@20
    ap_by_class: dict[str, list[float]] = defaultdict(list)

    for q_vec, q_labels, q_main in zip(query_embs, query_labels, query_main_labels):
        hits = search(client, collection, q_vec, top_k=K, ef_search=ef, filters=None)
        retrieved = [library_labels_by_id.get(h["patch_id"], []) for h in hits]
        ap = average_precision_at_k(q_labels, retrieved, K)
        ap_by_class[q_main].append(ap)

    # 聚合每个类别的平均 AP, 同时记录样本数(便于诊断)
    per_class: dict[str, dict] = {}
    for cls, aps in ap_by_class.items():
        per_class[cls] = {
            "mAP@20":  round(float(np.mean(aps)),  4),
            "std":     round(float(np.std(aps)),   4),
            "n_query": int(len(aps)),
        }

    # 全局平均
    overall = {
        "mAP@20_macro": round(float(np.mean([v["mAP@20"] for v in per_class.values()])), 4),
        "mAP@20_micro": round(float(np.mean(
            [ap for aps in ap_by_class.values() for ap in aps]
        )), 4),
        "n_classes": len(per_class),
        "n_queries": int(sum(v["n_query"] for v in per_class.values())),
    }

    result = {
        "overall": overall,
        "per_class": dict(sorted(per_class.items(), key=lambda x: -x[1]["mAP@20"])),
    }

    out = reports_dir / "per_class_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 终端打印一个易读的排名表
    logging.info("=" * 50)
    logging.info("Per-class mAP@20 (sorted)")
    logging.info("=" * 50)
    for cls, info in result["per_class"].items():
        logging.info(f"  {cls:<22} mAP@20={info['mAP@20']:.4f}  "
                     f"std={info['std']:.4f}  n={info['n_query']}")
    logging.info("-" * 50)
    logging.info(f"  Macro avg: {overall['mAP@20_macro']:.4f}   "
                 f"Micro avg: {overall['mAP@20_micro']:.4f}")
    logging.info(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
