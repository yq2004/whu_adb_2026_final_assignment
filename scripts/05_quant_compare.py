"""
Step 5:量化对比实验。

对应论文 §5.2.2 量化策略与精度—成本权衡。

测试三种存储模式 (FP32 / INT8 标量量化 / Binary 1-bit),
在相同 HNSW 参数下衡量:
  - Recall@10 / mAP@20 (检索精度)
  - P95 时延 (查询性能)
  - 估算内存占用

实现策略:
  - 不重新生成嵌入, 直接复用 step 2 产出的 embeddings.npy
  - 每种量化建一个独立 collection, 跑评估, 然后删除
  - 结果写入 reports/quant_results.json

用法:
    python scripts/05_quant_compare.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.http import models as qm

from src.db.qdrant_client import ingest, make_client, search
from src.db.schema import create_collection
from src.evaluation.metrics import aggregate_metrics


# 三种量化策略的 config patch
QUANT_VARIANTS = [
    {
        "name": "FP32",
        "label": "FP32 (no quantization)",
        "quantization": {"type": "none"},
        "bytes_per_vec": 4 * 1024,        # 1024 维 × 4 字节
    },
    {
        "name": "INT8",
        "label": "INT8 (scalar)",
        "quantization": {"type": "scalar", "quantile": 0.99, "always_ram": True},
        "bytes_per_vec": 1 * 1024,        # 1024 维 × 1 字节
    },
    {
        "name": "Binary",
        "label": "Binary (1-bit)",
        "quantization": {"type": "binary", "always_ram": True},
        "bytes_per_vec": 0.125 * 1024,    # 1024 维 × 1 比特 = 128 字节
    },
]


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def evaluate(client, collection, query_embs, query_labels, library_labels_by_id,
             ef: int, ks: list[int]) -> dict:
    """跑一次评估, 返回精度+时延。"""
    latencies = []
    retrieved_labels = []
    top_k = max(ks)

    for q_vec, q_labels in zip(query_embs, query_labels):
        t0 = time.perf_counter()
        hits = search(client, collection, q_vec, top_k=top_k, ef_search=ef, filters=None)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_labels.append([
            library_labels_by_id.get(h["patch_id"], []) for h in hits
        ])

    metrics = aggregate_metrics(query_labels, retrieved_labels, ks=ks)
    metrics["latency_p50_ms"] = round(percentile(latencies, 50), 2)
    metrics["latency_p95_ms"] = round(percentile(latencies, 95), 2)
    metrics["latency_p99_ms"] = round(percentile(latencies, 99), 2)
    return metrics


def main(cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_cfg = yaml.safe_load(open(cfg_path, "r"))

    emb_dir = Path(base_cfg["paths"]["embeddings_dir"])
    reports_dir = Path(base_cfg["evaluation"]["output_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 加载已生成的嵌入与元数据
    library_embs = np.load(emb_dir / "library_embeddings.npy")
    query_embs = np.load(emb_dir / "queries_embeddings.npy")
    library_meta = pd.read_parquet(emb_dir / "library_metadata.parquet")
    query_meta = pd.read_parquet(emb_dir / "queries_metadata.parquet")

    query_labels = [list(x) for x in query_meta["labels"].tolist()]
    library_labels_by_id = {row["patch_id"]: list(row["labels"])
                            for _, row in library_meta.iterrows()}

    n_vec = len(library_embs)
    vector_size = int(library_embs.shape[1])
    ef = base_cfg["retrieval"]["ef_search"]
    ks = base_cfg["evaluation"]["ks"]

    client = make_client(base_cfg["qdrant"]["url"])
    results: dict[str, dict] = {}

    for variant in QUANT_VARIANTS:
        logging.info("=" * 60)
        logging.info("== quantization variant: %s ==", variant["name"])

        # 派生一个 config 副本, 改 collection 名 + 量化设置
        cfg = copy.deepcopy(base_cfg)
        cfg["qdrant"]["collection"] = f"quant_test_{variant['name'].lower()}"
        cfg["qdrant"]["quantization"] = variant["quantization"]
        cfg["qdrant"]["vector_size"] = vector_size

        # 建表 + 入库
        create_collection(client, cfg)
        ingest(client, cfg["qdrant"]["collection"], library_embs, library_meta)

        # 等索引建好
        logging.info("waiting 10s for HNSW to build...")
        time.sleep(10)

        info = client.get_collection(cfg["qdrant"]["collection"])
        logging.info("indexed: %d / %d, segments: %d",
                     info.indexed_vectors_count, info.points_count, info.segments_count)

        # 评估
        metrics = evaluate(
            client, cfg["qdrant"]["collection"],
            query_embs, query_labels, library_labels_by_id,
            ef=ef, ks=ks,
        )
        metrics["memory_mb_estimated"] = round(
            n_vec * variant["bytes_per_vec"] / (1024 * 1024), 2
        )
        metrics["memory_relative_to_fp32"] = round(
            variant["bytes_per_vec"] / (4 * 1024), 4
        )
        metrics["label"] = variant["label"]

        results[variant["name"]] = metrics
        logging.info("result: %s", metrics)

        # 清理临时 collection (节省 Qdrant 存储)
        client.delete_collection(cfg["qdrant"]["collection"])
        logging.info("deleted collection %s", cfg["qdrant"]["collection"])

    out = reports_dir / "quant_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
