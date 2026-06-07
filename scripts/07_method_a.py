"""
Script 07: 范式 A（地理 bbox 元数据过滤）评估。

对应论文 §4.5.2 元数据过滤范式。

评估逻辑:
  对每条查询，以其自身经纬度为中心构造一个正方形 bbox，
  传给 Qdrant 做 payload 过滤 + 向量近邻检索（范式 A），
  与无过滤的纯向量基线对比精度和有效召回量。

  通过扫描不同的 half_deg（半边长，单位度）来观察：
    - 过滤越严格（half_deg 越小），library 覆盖率越低；
    - 精度（mAP / NDCG）是否因候选集语义更集中而提升；
    - 有效召回条数（avg_effective_k）是否下降。

不修改任何已有源码。

用法:
    python scripts/06_method_a.py --config configs/config.yaml

输出:
    {reports_dir}/method_a_results.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.qdrant_client import make_client, search
from src.evaluation.metrics import aggregate_metrics


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def evaluate_with_bbox(
    client,
    collection: str,
    query_embs: np.ndarray,
    query_meta: pd.DataFrame,
    library_labels_by_id: dict,
    library_meta: pd.DataFrame,
    ef: int,
    ks: list[int],
    half_deg: float,
) -> dict:
    """
    对所有查询施加以自身坐标为中心、边长 2*half_deg 度的 bbox 过滤，
    返回精度指标 + 过滤统计量。
    """
    top_k = max(ks)
    latencies_ms: list[float] = []
    retrieved_labels: list[list[list[str]]] = []
    effective_ks: list[int] = []          # 实际返回条数（bbox 内候选可能 < top_k）

    for i, (q_vec, (_, q_row)) in enumerate(
        zip(query_embs, query_meta.iterrows())
    ):
        lat = float(q_row["lat"])
        lon = float(q_row["lon"])
        bbox_filter = {
            "bbox": [
                lon - half_deg,   # min_lon
                lat - half_deg,   # min_lat
                lon + half_deg,   # max_lon
                lat + half_deg,   # max_lat
            ]
        }

        t0 = time.perf_counter()
        hits = search(
            client, collection, q_vec,
            top_k=top_k, ef_search=ef, filters=bbox_filter,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        effective_ks.append(len(hits))
        retrieved_labels.append([
            library_labels_by_id.get(h["patch_id"], []) for h in hits
        ])

    # 精度指标（对返回条数不足 top_k 的查询，指标自然偏保守）
    query_labels = [list(x) for x in query_meta["labels"].tolist()]
    metrics = aggregate_metrics(query_labels, retrieved_labels, ks=ks)

    metrics["latency_p50_ms"] = round(percentile(latencies_ms, 50), 2)
    metrics["latency_p95_ms"] = round(percentile(latencies_ms, 95), 2)
    metrics["latency_p99_ms"] = round(percentile(latencies_ms, 99), 2)
    metrics["avg_effective_k"] = round(float(np.mean(effective_ks)), 1)
    metrics["min_effective_k"] = int(np.min(effective_ks))

    # 估算平均覆盖率：library 中落在"平均 bbox"内的比例
    # 用随机抽 500 条查询的 bbox 采样估计，避免全量 O(n²) 扫描
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(query_meta), size=min(500, len(query_meta)), replace=False)
    lib_lats = library_meta["lat"].values.astype(float)
    lib_lons = library_meta["lon"].values.astype(float)
    coverage_ratios: list[float] = []
    for idx in sample_idx:
        row = query_meta.iloc[idx]
        lat, lon = float(row["lat"]), float(row["lon"])
        mask = (
            (lib_lats >= lat - half_deg) & (lib_lats <= lat + half_deg) &
            (lib_lons >= lon - half_deg) & (lib_lons <= lon + half_deg)
        )
        coverage_ratios.append(mask.mean())
    metrics["avg_library_coverage"] = round(float(np.mean(coverage_ratios)), 4)

    return metrics


def evaluate_baseline(
    client,
    collection: str,
    query_embs: np.ndarray,
    query_meta: pd.DataFrame,
    library_labels_by_id: dict,
    ef: int,
    ks: list[int],
) -> dict:
    """无过滤的纯向量基线（复现 04_run_eval.py 的 default 条件）。"""
    top_k = max(ks)
    latencies_ms: list[float] = []
    retrieved_labels: list[list[list[str]]] = []
    query_labels = [list(x) for x in query_meta["labels"].tolist()]

    for q_vec in query_embs:
        t0 = time.perf_counter()
        hits = search(client, collection, q_vec, top_k=top_k, ef_search=ef, filters=None)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        retrieved_labels.append([
            library_labels_by_id.get(h["patch_id"], []) for h in hits
        ])

    metrics = aggregate_metrics(query_labels, retrieved_labels, ks=ks)
    metrics["latency_p50_ms"] = round(percentile(latencies_ms, 50), 2)
    metrics["latency_p95_ms"] = round(percentile(latencies_ms, 95), 2)
    metrics["latency_p99_ms"] = round(percentile(latencies_ms, 99), 2)
    metrics["avg_effective_k"] = float(top_k)
    metrics["avg_library_coverage"] = 1.0
    return metrics


def main(cfg_path: str, half_degs: list[float]) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    reports_dir = Path(cfg["evaluation"]["output_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    query_embs   = np.load(emb_dir / "queries_embeddings.npy")
    query_meta   = pd.read_parquet(emb_dir / "queries_metadata.parquet")
    library_meta = pd.read_parquet(emb_dir / "library_metadata.parquet")

    library_labels_by_id = {
        row["patch_id"]: list(row["labels"])
        for _, row in library_meta.iterrows()
    }

    client     = make_client(cfg["qdrant"]["url"])
    collection = cfg["qdrant"]["collection"]
    ef         = cfg["retrieval"]["ef_search"]
    ks         = cfg["evaluation"]["ks"]

    results: dict[str, dict] = {}

    # 基线
    logging.info("== baseline (no filter) ==")
    results["baseline"] = evaluate_baseline(
        client, collection, query_embs, query_meta,
        library_labels_by_id, ef=ef, ks=ks,
    )
    logging.info("baseline: Recall@10=%.4f  mAP@10=%.4f  NDCG@10=%.4f",
                 results["baseline"]["Recall@10"],
                 results["baseline"]["mAP@10"],
                 results["baseline"]["NDCG@10"])

    # 范式 A：扫描不同 bbox 半径
    results["bbox_sweep"] = {}
    for hd in half_degs:
        tag = f"half_{hd}deg"
        logging.info("== bbox half=%.2f deg ==", hd)
        m = evaluate_with_bbox(
            client, collection, query_embs, query_meta,
            library_labels_by_id, library_meta,
            ef=ef, ks=ks, half_deg=hd,
        )
        results["bbox_sweep"][tag] = m
        logging.info(
            "  Recall@10=%.4f  mAP@10=%.4f  NDCG@10=%.4f  "
            "avg_effective_k=%.1f  avg_coverage=%.2f%%  p95=%.2fms",
            m["Recall@10"], m["mAP@10"], m["NDCG@10"],
            m["avg_effective_k"], m["avg_library_coverage"] * 100,
            m["latency_p95_ms"],
        )

    out = reports_dir / "method_a_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="范式 A（bbox 元数据过滤）评估")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument(
        "--half-degs", nargs="+", type=float,
        default=[0.5, 1.0, 2.0, 5.0, 10.0],
        help="bbox 半边长（度），可多值。默认 0.5 1.0 2.0 5.0 10.0",
    )
    args = ap.parse_args()
    main(args.config, args.half_degs)
