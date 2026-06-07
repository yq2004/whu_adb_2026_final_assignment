"""
Script 08: 范式 B（语义 + 地理加权重排）评估。

对应论文 §4.5.3 加权重排序范式:
    S(c) = w_sem * sim_sem(q, c) + w_geo * proximity_geo(q, c)
    proximity_geo = exp(- distance_km / decay_km)

评估逻辑:
  对每条查询, 先用 HNSW 召回 top_k × candidate_factor 个候选,
  再按上述融合分数重排, 取最终 top_k。

  扫描两个维度的参数空间:
    - w_geo ∈ {0.1, 0.2, 0.4, 0.6, 0.8}   (w_sem = 1 - w_geo)
    - decay_km ∈ {10, 50, 200}

  对每种配置计算 Recall / mAP / NDCG, 并额外记录:
    - avg_sem_score:  返回结果的平均语义相似度  → 量化重排对语义质量的损耗
    - avg_prox_score: 返回结果的平均地理邻近度  → 量化地理偏置的强度
    - avg_fused_score: 平均融合分数

不修改任何已有源码。

用法:
    python scripts/08_method_b.py --config configs/config.yaml

    # 自定义参数空间
    python scripts/08_method_b.py --config configs/config.yaml \\
        --w-geos 0.2 0.4 0.6 \\
        --decay-kms 10 50 200 \\
        --candidate-factor 5

输出:
    {reports_dir}/method_b_results.json
"""
from __future__ import annotations

import argparse
import itertools
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
from src.retrieval.search import search_with_rerank
from src.evaluation.metrics import aggregate_metrics


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def evaluate_baseline(
    client,
    collection: str,
    query_embs: np.ndarray,
    query_meta: pd.DataFrame,
    library_labels_by_id: dict,
    ef: int,
    ks: list[int],
) -> dict:
    """无过滤、无重排的纯向量基线（复现 04_run_eval.py default 条件）。"""
    top_k = max(ks)
    latencies_ms: list[float] = []
    retrieved_labels: list[list[list[str]]] = []
    sem_scores: list[float] = []
    query_labels = [list(x) for x in query_meta["labels"].tolist()]

    for q_vec in query_embs:
        t0 = time.perf_counter()
        hits = search(client, collection, q_vec, top_k=top_k, ef_search=ef, filters=None)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        retrieved_labels.append([
            library_labels_by_id.get(h["patch_id"], []) for h in hits
        ])
        sem_scores.extend(h["score"] for h in hits)

    metrics = aggregate_metrics(query_labels, retrieved_labels, ks=ks)
    metrics["latency_p50_ms"]  = round(percentile(latencies_ms, 50), 2)
    metrics["latency_p95_ms"]  = round(percentile(latencies_ms, 95), 2)
    metrics["latency_p99_ms"]  = round(percentile(latencies_ms, 99), 2)
    metrics["avg_sem_score"]   = round(float(np.mean(sem_scores)), 4)
    metrics["avg_prox_score"]  = None   # 基线无地理重排，不适用
    metrics["avg_fused_score"] = None
    return metrics


def evaluate_rerank(
    client,
    collection: str,
    query_embs: np.ndarray,
    query_meta: pd.DataFrame,
    library_labels_by_id: dict,
    ef: int,
    ks: list[int],
    w_geo: float,
    decay_km: float,
    candidate_factor: int,
) -> dict:
    """
    范式 B 评估：对所有查询跑加权重排，聚合精度指标与分数统计。
    """
    w_sem = round(1.0 - w_geo, 6)
    top_k = max(ks)
    query_labels = [list(x) for x in query_meta["labels"].tolist()]

    latencies_ms: list[float]   = []
    retrieved_labels: list[list[list[str]]] = []
    sem_scores: list[float]     = []
    prox_scores: list[float]    = []
    fused_scores: list[float]   = []

    rows = list(query_meta.iterrows())
    for q_vec, (_, q_row) in zip(query_embs, rows):
        q_latlon = (float(q_row["lat"]), float(q_row["lon"]))

        t0 = time.perf_counter()
        results = search_with_rerank(
            client, collection, q_vec,
            query_latlon=q_latlon,
            top_k=top_k,
            ef_search=ef,
            filters=None,
            w_sem=w_sem,
            w_geo=w_geo,
            candidate_factor=candidate_factor,
            decay_km=decay_km,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        retrieved_labels.append([
            # payload 里有 labels；也可用 library_labels_by_id 做 fallback
            results[i].payload.get("labels") or
            library_labels_by_id.get(results[i].patch_id, [])
            for i in range(len(results))
        ])

        for r in results:
            sem_scores.append(r.semantic_score)
            prox_scores.append(r.proximity_score or 0.0)
            fused_scores.append(r.score)

    metrics = aggregate_metrics(query_labels, retrieved_labels, ks=ks)
    metrics["latency_p50_ms"]  = round(percentile(latencies_ms, 50), 2)
    metrics["latency_p95_ms"]  = round(percentile(latencies_ms, 95), 2)
    metrics["latency_p99_ms"]  = round(percentile(latencies_ms, 99), 2)
    metrics["avg_sem_score"]   = round(float(np.mean(sem_scores)),   4)
    metrics["avg_prox_score"]  = round(float(np.mean(prox_scores)),  4)
    metrics["avg_fused_score"] = round(float(np.mean(fused_scores)), 4)
    return metrics


def main(
    cfg_path: str,
    w_geos: list[float],
    decay_kms: list[float],
    candidate_factor: int,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    emb_dir     = Path(cfg["paths"]["embeddings_dir"])
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

    # ── 基线 ──────────────────────────────────────────────────────
    logging.info("== baseline (no rerank) ==")
    results["baseline"] = evaluate_baseline(
        client, collection, query_embs, query_meta,
        library_labels_by_id, ef=ef, ks=ks,
    )
    logging.info(
        "baseline: Recall@10=%.4f  mAP@10=%.4f  NDCG@10=%.4f  p95=%.2fms",
        results["baseline"]["Recall@10"],
        results["baseline"]["mAP@10"],
        results["baseline"]["NDCG@10"],
        results["baseline"]["latency_p95_ms"],
    )

    # ── 范式 B 参数扫描 ───────────────────────────────────────────
    results["rerank_sweep"] = {}
    grid = list(itertools.product(w_geos, decay_kms))
    logging.info("sweeping %d configs (w_geo × decay_km) ...", len(grid))

    for w_geo, decay_km in grid:
        w_sem = round(1.0 - w_geo, 6)
        tag   = f"wsem{w_sem}_wgeo{w_geo}_decay{int(decay_km)}km"
        logging.info(
            "== w_sem=%.2f  w_geo=%.2f  decay_km=%.0f  candidate_factor=%d ==",
            w_sem, w_geo, decay_km, candidate_factor,
        )

        m = evaluate_rerank(
            client, collection, query_embs, query_meta,
            library_labels_by_id,
            ef=ef, ks=ks,
            w_geo=w_geo, decay_km=decay_km,
            candidate_factor=candidate_factor,
        )
        m["w_sem"]            = w_sem
        m["w_geo"]            = w_geo
        m["decay_km"]         = decay_km
        m["candidate_factor"] = candidate_factor
        results["rerank_sweep"][tag] = m

        logging.info(
            "  Recall@10=%.4f  mAP@10=%.4f  NDCG@10=%.4f"
            "  sem=%.3f  prox=%.3f  fused=%.3f  p95=%.2fms",
            m["Recall@10"], m["mAP@10"], m["NDCG@10"],
            m["avg_sem_score"], m["avg_prox_score"], m["avg_fused_score"],
            m["latency_p95_ms"],
        )

    out = reports_dir / "method_b_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="范式 B（语义+地理加权重排）评估")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument(
        "--w-geos", nargs="+", type=float,
        default=[0.1, 0.2, 0.4, 0.6, 0.8],
        help="地理权重 w_geo（w_sem = 1 - w_geo），可多值",
    )
    ap.add_argument(
        "--decay-kms", nargs="+", type=float,
        default=[10.0, 50.0, 200.0],
        help="指数衰减半径（km），可多值",
    )
    ap.add_argument(
        "--candidate-factor", type=int, default=None,
        help="初次召回倍数，默认取 config.yaml retrieval.rerank_candidate_factor",
    )
    args = ap.parse_args()

    # candidate_factor 优先用 CLI 参数，其次读 config
    if args.candidate_factor is None:
        _cfg = yaml.safe_load(open(args.config, "r"))
        candidate_factor = _cfg["retrieval"]["rerank_candidate_factor"]
    else:
        candidate_factor = args.candidate_factor

    main(args.config, args.w_geos, args.decay_kms, candidate_factor)
