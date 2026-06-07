"""
Qdrant 客户端封装:批量入库与基础检索接口。

对应论文 §4.4.3 入库与构建、§4.5.2 元数据过滤范式的实现。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

logger = logging.getLogger(__name__)


def patch_id_to_uuid(patch_id: str) -> str:
    """
    Qdrant 的 point ID 要求为 unsigned int 或 UUID。
    BigEarthNet patch_id 是字符串, 用 md5 派生稳定 UUID。
    """
    h = hashlib.md5(patch_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def make_client(url: str, timeout: int = 60) -> QdrantClient:
    return QdrantClient(url=url, timeout=timeout)


def upsert_batch(
    client: QdrantClient,
    collection: str,
    patch_ids: list[str],
    vectors: np.ndarray,                 # [B, D] float32
    payloads: list[dict[str, Any]],
) -> None:
    """单批次 upsert。"""
    points = [
        qm.PointStruct(
            id=patch_id_to_uuid(pid),
            vector=vec.tolist(),
            payload={**payload, "patch_id": pid},   # 保留原 patch_id 便于回查
        )
        for pid, vec, payload in zip(patch_ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection, points=points, wait=False)


def ingest(
    client: QdrantClient,
    collection: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    batch_size: int = 512,
) -> None:
    """
    将一批(向量, 元数据)整体写入 collection。

    实现要点(论文 §4.4.3):
    - 按 batch_size 分批 upsert, 避免单次请求过大;
    - wait=False, 利用 Qdrant 异步写入与 WAL 提升吞吐;
    - 最后调用 update_collection 触发 indexing optimizer。
    """
    assert len(embeddings) == len(metadata), "embeddings and metadata must align"
    n = len(embeddings)
    for start in tqdm(range(0, n, batch_size), desc=f"upsert→{collection}"):
        end = min(start + batch_size, n)
        sub_meta = metadata.iloc[start:end]
        payloads = []
        for _, row in sub_meta.iterrows():
            payloads.append({
                "tile_id": row["tile_id"],
                "country": row.get("country") or "Unknown",
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "geohash": row["geohash"],
                "acquired_at": row.get("acquired_at") or "1970-01-01T00:00:00Z",
                "labels": list(row["labels"]) if row["labels"] is not None else [],
                "main_label": row.get("main_label", ""),
            })
        upsert_batch(
            client, collection,
            patch_ids=sub_meta["patch_id"].tolist(),
            vectors=embeddings[start:end].astype(np.float32),
            payloads=payloads,
        )

    # 触发优化器, 让新写入的段落转为带 HNSW 索引的段落
    client.update_collection(
        collection_name=collection,
        optimizer_config=qm.OptimizersConfigDiff(indexing_threshold=100),
    )
    logger.info("ingested %d points into %s", n, collection)


# ── 检索基础原语(供 retrieval/search.py 使用)─────────────────────────────

def build_filter(filters: dict[str, Any] | None) -> qm.Filter | None:
    """
    把用户请求中的 filters 字典翻译成 Qdrant Filter。

    支持的字段(对应论文 §4.4.1 payload schema):
        country:     str
        labels:      list[str]  ── 候选必须包含其中至少一个
        tile_id:     str
        bbox:        [min_lon, min_lat, max_lon, max_lat]
        time_range:  [iso_start, iso_end]
        geohash_prefix: str
    """
    if not filters:
        return None
    must: list[qm.Condition] = []

    if v := filters.get("country"):
        must.append(qm.FieldCondition(key="country", match=qm.MatchValue(value=v)))

    if v := filters.get("tile_id"):
        must.append(qm.FieldCondition(key="tile_id", match=qm.MatchValue(value=v)))

    if labels := filters.get("labels"):
        must.append(qm.FieldCondition(key="labels", match=qm.MatchAny(any=list(labels))))

    if bbox := filters.get("bbox"):
        min_lon, min_lat, max_lon, max_lat = bbox
        must.append(qm.FieldCondition(key="lon", range=qm.Range(gte=min_lon, lte=max_lon)))
        must.append(qm.FieldCondition(key="lat", range=qm.Range(gte=min_lat, lte=max_lat)))

    if tr := filters.get("time_range"):
        must.append(qm.FieldCondition(
            key="acquired_at",
            range=qm.DatetimeRange(gte=tr[0], lte=tr[1]),
        ))

    if prefix := filters.get("geohash_prefix"):
        must.append(qm.FieldCondition(key="geohash", match=qm.MatchText(text=prefix)))

    return qm.Filter(must=must) if must else None


# def search(
#     client: QdrantClient,
#     collection: str,
#     query_vector: np.ndarray,             # [D]
#     top_k: int = 10,
#     ef_search: int = 128,
#     filters: dict[str, Any] | None = None,
# ) -> list[dict[str, Any]]:
#     """
#     带过滤的向量近邻检索, 返回候选列表。

#     每个候选: {patch_id, score, payload}
#     """
#     hits = client.search(
#         collection_name=collection,
#         query_vector=query_vector.tolist(),
#         query_filter=build_filter(filters),
#         limit=top_k,
#         with_payload=True,
#         search_params=qm.SearchParams(hnsw_ef=ef_search, exact=False),
#     )
#     return [
#         {"patch_id": h.payload.get("patch_id"), "score": float(h.score), "payload": h.payload}
#         for h in hits
#     ]
def search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,             # [D]
    top_k: int = 10,
    ef_search: int = 128,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """带过滤的向量近邻检索, 返回候选列表。

    每个候选: {patch_id, score, payload}
    """
    # INT 8 关闭 rescoring 而 Binary 开启，导致Binary反而比INT8的精度高。
    # response = client.query_points(
    #     collection_name=collection,
    #     query=query_vector.tolist(),
    #     query_filter=build_filter(filters),
    #     limit=top_k,
    #     with_payload=True,
    #     search_params=qm.SearchParams(hnsw_ef=ef_search, exact=False),
    # )

    # 修复： 让INT 8 也开启 rescoring
    response = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
        search_params=qm.SearchParams(
            hnsw_ef=ef_search,
            exact=False,
            quantization=qm.QuantizationSearchParams(
                rescore=True,          # 强制用原始向量重排
                oversampling=2.0,      # 粗排候选数 = top_k × 2
            ),
        ),
    )

    return [
        {"patch_id": h.payload.get("patch_id"), "score": float(h.score), "payload": h.payload}
        for h in response.points
    ]