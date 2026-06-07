"""
Step 4:评估实验。

对应论文 §5.2 检索性能评估、§5.3 检索精度评估。
覆盖:
- 默认配置下的总体精度(Recall@K / mAP@K / NDCG@K)
- HNSW ef 参数扫描 → 性能-精度权衡
- 检索时延统计(P50/P95)

用法:
    python scripts/04_run_eval.py --config configs/config.yaml
输出:
    {reports_dir}/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db.qdrant_client import make_client, search
from src.evaluation.metrics import aggregate_metrics


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def evaluate_one_ef(
    client, collection, query_embs, query_labels, library_labels_by_id,
    ef: int, ks: list[int],
) -> dict:
    """单次 ef 设置下的精度 + 时延评估。"""
    latencies_ms: list[float] = []
    retrieved_labels: list[list[list[str]]] = []
    top_k = max(ks)

    for q_vec, q_labels in zip(query_embs, query_labels):
        t0 = time.perf_counter()
        hits = search(client, collection, q_vec, top_k=top_k, ef_search=ef, filters=None)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        labels_per_hit = [
            library_labels_by_id.get(h["patch_id"], [])
            for h in hits
        ]
        retrieved_labels.append(labels_per_hit)

    metrics = aggregate_metrics(query_labels, retrieved_labels, ks=ks)
    metrics["latency_p50_ms"] = round(percentile(latencies_ms, 50), 2)
    metrics["latency_p95_ms"] = round(percentile(latencies_ms, 95), 2)
    metrics["latency_p99_ms"] = round(percentile(latencies_ms, 99), 2)
    return metrics


def main(cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    reports_dir = Path(cfg["evaluation"]["output_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 载入查询嵌入与库元数据(库集元数据用于把检索到的 patch_id 还原成 labels)
    query_embs = np.load(emb_dir / "queries_embeddings.npy")
    query_meta = pd.read_parquet(emb_dir / "queries_metadata.parquet")
    library_meta = pd.read_parquet(emb_dir / "library_metadata.parquet")

    query_labels = [list(x) for x in query_meta["labels"].tolist()]
    library_labels_by_id = {
        row["patch_id"]: list(row["labels"])
        for _, row in library_meta.iterrows()
    }

    client = make_client(cfg["qdrant"]["url"])
    collection = cfg["qdrant"]["collection"]
    ks = cfg["evaluation"]["ks"]

    results: dict[str, dict] = {}

    # 论文 §5.3.1 默认配置
    logging.info("== default config (ef=%d) ==", cfg["retrieval"]["ef_search"])
    results["default"] = evaluate_one_ef(
        client, collection, query_embs, query_labels,
        library_labels_by_id, ef=cfg["retrieval"]["ef_search"], ks=ks,
    )
    logging.info("default: %s", results["default"])

    # 论文 §5.2.1 ef 参数扫描
    logging.info("== ef sweep ==")
    results["ef_sweep"] = {}
    for ef in [16, 32, 64, 96, 128, 192, 256]:
        m = evaluate_one_ef(
            client, collection, query_embs, query_labels,
            library_labels_by_id, ef=ef, ks=ks,
        )
        results["ef_sweep"][str(ef)] = m
        logging.info("ef=%d: Recall@10=%.4f, P95=%.2fms",
                     ef, m["Recall@10"], m["latency_p95_ms"])

    out = reports_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
