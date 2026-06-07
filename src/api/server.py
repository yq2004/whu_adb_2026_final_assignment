"""
FastAPI 检索服务。

对应论文 §4.5.1 检索接口设计、§4.5.4 完整检索流程。

启动:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000

接口:
    POST /search     主检索接口, 支持过滤 / 重排序
    GET  /health     健康检查
    GET  /collection 返回 collection 状态
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..db.qdrant_client import make_client, search as raw_search
from ..retrieval.search import search_filtered, search_with_rerank

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── 应用状态(在 lifespan 中初始化)──────────────────────────────
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_path = os.environ.get("GEOSEM_CONFIG", "configs/config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    state["cfg"] = cfg
    state["client"] = make_client(cfg["qdrant"]["url"])

    # Clay 编码器很重(1.25 GB), 按需懒加载
    state["encoder"] = None
    logger.info("API started, qdrant=%s collection=%s",
                cfg["qdrant"]["url"], cfg["qdrant"]["collection"])
    yield
    logger.info("API stopped")


app = FastAPI(title="geo-semantic-db", version="0.1.0", lifespan=lifespan)


def _ensure_encoder():
    """按需加载 Clay 编码器。检索时若提供 patch_id 或 vector, 可避免加载模型。"""
    if state["encoder"] is None:
        from ..embedding.clay_encoder import ClayEncoder
        cfg = state["cfg"]
        state["encoder"] = ClayEncoder(
            ckpt_path=cfg["paths"]["clay_ckpt"],
            meta_path=cfg["paths"]["clay_meta"],
            sensor=cfg["data"]["sensor"],
            device=cfg["embedding"]["device"],
            fp16=cfg["embedding"]["fp16"],
            pool=cfg["embedding"]["pool"],
            l2_normalize=cfg["embedding"]["l2_normalize"],
        )
    return state["encoder"]


# ── 请求/响应模型 ──────────────────────────────────────────────

class Filters(BaseModel):
    country: Optional[str] = None
    tile_id: Optional[str] = None
    labels: Optional[list[str]] = None
    bbox: Optional[list[float]] = Field(
        None, description="[min_lon, min_lat, max_lon, max_lat]"
    )
    time_range: Optional[list[str]] = Field(None, description="[iso_start, iso_end]")
    geohash_prefix: Optional[str] = None
    polygon_geojson: Optional[dict] = Field(
        None, description="任意多边形, 应用层精确过滤"
    )


class RerankConfig(BaseModel):
    enabled: bool = False
    w_sem: float = 0.6
    w_geo: float = 0.4
    decay_km: float = 50.0


class SearchRequest(BaseModel):
    # 查询输入三选一(优先级 patch_id > vector > image_b64)
    patch_id: Optional[str] = Field(None, description="使用库内已有 patch 作为 query")
    vector:   Optional[list[float]] = Field(None, description="直接提供查询向量")
    # image_b64 留作扩展:在线对上传影像跑 Clay
    image_b64: Optional[str] = None

    # 在重排序模式下需要(用于计算地理邻近度)
    query_lat: Optional[float] = None
    query_lon: Optional[float] = None

    top_k: int = 10
    ef_search: Optional[int] = None
    filters: Optional[Filters] = None
    rerank: Optional[RerankConfig] = None


class Hit(BaseModel):
    patch_id: str
    score: float
    semantic_score: float
    proximity_score: Optional[float]
    payload: dict[str, Any]


class SearchResponse(BaseModel):
    hits: list[Hit]
    mode: str        # "filter" or "rerank"


# ── 路由 ───────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/collection")
def collection_info():
    cfg = state["cfg"]
    info = state["client"].get_collection(cfg["qdrant"]["collection"])
    return {
        "name": cfg["qdrant"]["collection"],
        "vectors_count": info.vectors_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "status": info.status,
    }


def _resolve_query_vector(req: SearchRequest) -> np.ndarray:
    """根据请求把查询输入归一为 numpy 向量。"""
    cfg = state["cfg"]
    client = state["client"]

    if req.vector is not None:
        return np.asarray(req.vector, dtype=np.float32)

    if req.patch_id is not None:
        # 用 scroll + filter 拉这条 patch 的向量(也可以 retrieve by id)
        from ..db.qdrant_client import patch_id_to_uuid
        from qdrant_client.http import models as qm
        rec = client.retrieve(
            collection_name=cfg["qdrant"]["collection"],
            ids=[patch_id_to_uuid(req.patch_id)],
            with_vectors=True,
        )
        if not rec:
            raise HTTPException(404, f"patch {req.patch_id} not in collection")
        return np.asarray(rec[0].vector, dtype=np.float32)

    if req.image_b64 is not None:
        # 留作扩展:解码 base64、组装成 chips、调 Clay
        raise HTTPException(501, "image_b64 inference not implemented yet")

    raise HTTPException(400, "must provide patch_id, vector, or image_b64")


@app.post("/search", response_model=SearchResponse)
def do_search(req: SearchRequest) -> SearchResponse:
    cfg = state["cfg"]
    client = state["client"]
    collection = cfg["qdrant"]["collection"]
    ef = req.ef_search or cfg["retrieval"]["ef_search"]

    q_vec = _resolve_query_vector(req)
    filters_dict = req.filters.model_dump(exclude_none=True) if req.filters else None
    polygon = None
    if filters_dict and "polygon_geojson" in filters_dict:
        polygon = filters_dict.pop("polygon_geojson")

    use_rerank = req.rerank is not None and req.rerank.enabled
    if use_rerank:
        if req.query_lat is None or req.query_lon is None:
            raise HTTPException(400, "rerank requires query_lat & query_lon")
        results = search_with_rerank(
            client, collection, q_vec,
            query_latlon=(req.query_lat, req.query_lon),
            top_k=req.top_k,
            ef_search=ef,
            filters=filters_dict,
            w_sem=req.rerank.w_sem,
            w_geo=req.rerank.w_geo,
            decay_km=req.rerank.decay_km,
            candidate_factor=cfg["retrieval"]["rerank_candidate_factor"],
        )
        mode = "rerank"
    else:
        results = search_filtered(
            client, collection, q_vec,
            top_k=req.top_k,
            ef_search=ef,
            filters=filters_dict,
            polygon_geojson=polygon,
        )
        mode = "filter"

    return SearchResponse(
        hits=[Hit(**r.__dict__) for r in results],
        mode=mode,
    )
