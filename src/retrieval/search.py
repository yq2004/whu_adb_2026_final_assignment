"""
空间—语义混合检索的高层接口。

对应论文 §4.5 空间—语义混合检索的实现:
- 4.5.2 元数据过滤范式 (此模块的 search_filtered)
- 4.5.3 加权重排序范式 (此模块的 search_with_rerank)
- 4.5.4 完整检索流程
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from shapely.geometry import Point, shape

from ..db.qdrant_client import search
from .rerank import compute_proximity_score

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    patch_id: str
    score: float                       # 最终排序分(语义 or 融合)
    semantic_score: float              # 原始语义相似度
    proximity_score: Optional[float]   # 地理邻近度(若启用)
    payload: dict[str, Any]


def search_filtered(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    top_k: int = 10,
    ef_search: int = 128,
    filters: Optional[dict[str, Any]] = None,
    polygon_geojson: Optional[dict] = None,   # 任意多边形精确过滤(应用层)
) -> list[SearchResult]:
    """
    范式 A:元数据过滤(论文 §4.5.2)。

    可选的 polygon_geojson 用于"超出 Qdrant 原生支持的复杂空间条件":
    Qdrant 用 bbox 做粗过滤, 应用层用 shapely 做精确多边形判定。
    """
    # 如果给了多边形, 自动派生其 bbox 作为粗过滤
    if polygon_geojson is not None:
        poly = shape(polygon_geojson)
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        filters = dict(filters or {})
        filters.setdefault("bbox", [min_lon, min_lat, max_lon, max_lat])
    else:
        poly = None

    # 多边形过滤会丢弃部分候选, 召回时多取一些做缓冲
    fetch_k = top_k * 3 if poly is not None else top_k

    hits = search(client, collection, query_vector,
                  top_k=fetch_k, ef_search=ef_search, filters=filters)

    results: list[SearchResult] = []
    for h in hits:
        if poly is not None:
            p = Point(h["payload"]["lon"], h["payload"]["lat"])
            if not poly.covers(p):
                continue
        results.append(SearchResult(
            patch_id=h["patch_id"],
            score=h["score"],
            semantic_score=h["score"],
            proximity_score=None,
            payload=h["payload"],
        ))
        if len(results) >= top_k:
            break
    return results


def search_with_rerank(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    query_latlon: tuple[float, float],
    top_k: int = 10,
    ef_search: int = 128,
    filters: Optional[dict[str, Any]] = None,
    w_sem: float = 0.6,
    w_geo: float = 0.4,
    candidate_factor: int = 5,
    decay_km: float = 50.0,
) -> list[SearchResult]:
    """
    范式 B:加权重排序(论文 §4.5.3)。

    在初次召回 (top_k * candidate_factor) 后, 按
        S(c) = w_sem * sim_sem(q, c) + w_geo * proximity_geo(q, c)
    重新排序。

    proximity_geo 用对距离的指数衰减归一化, 落在 [0, 1] 区间, 见 rerank.py。
    """
    assert abs((w_sem + w_geo) - 1.0) < 1e-6, "weights should sum to 1"

    fetch_k = top_k * candidate_factor
    hits = search(client, collection, query_vector,
                  top_k=fetch_k, ef_search=ef_search, filters=filters)

    scored: list[SearchResult] = []
    for h in hits:
        c_lat = float(h["payload"]["lat"])
        c_lon = float(h["payload"]["lon"])
        prox = compute_proximity_score(query_latlon, (c_lat, c_lon), decay_km)
        fused = w_sem * h["score"] + w_geo * prox
        scored.append(SearchResult(
            patch_id=h["patch_id"],
            score=fused,
            semantic_score=h["score"],
            proximity_score=prox,
            payload=h["payload"],
        ))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
