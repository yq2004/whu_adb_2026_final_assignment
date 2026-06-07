"""
Qdrant collection schema 与建表。

对应论文 §4.4.1 Collection 设计、§4.4.2 索引参数与量化。
"""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)


def build_collection_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """根据 config.yaml 中的 qdrant 配置块, 拼出 recreate_collection 的关键参数。"""
    qcfg = cfg["qdrant"]
    distance_map = {
        "Cosine": qm.Distance.COSINE,
        "Euclid": qm.Distance.EUCLID,
        "Dot": qm.Distance.DOT,
    }

    # HNSW 索引参数(论文 §4.4.2)
    hnsw = qcfg.get("hnsw", {})
    hnsw_config = qm.HnswConfigDiff(
        m=hnsw.get("m", 16),
        ef_construct=hnsw.get("ef_construct", 128),
        full_scan_threshold=hnsw.get("full_scan_threshold", 10000),
    )

    # 量化策略(论文 §4.4.2 / §5.2.2)
    quant_cfg = qcfg.get("quantization", {})
    quant_type = quant_cfg.get("type", "scalar")
    if quant_type == "scalar":
        quantization_config = qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8,
                quantile=quant_cfg.get("quantile", 0.99),
                always_ram=quant_cfg.get("always_ram", True),
            )
        )
    elif quant_type == "binary":
        quantization_config = qm.BinaryQuantization(
            binary=qm.BinaryQuantizationConfig(
                always_ram=quant_cfg.get("always_ram", True),
            )
        )
    else:
        quantization_config = None

    return dict(
        vectors_config=qm.VectorParams(
            size=qcfg["vector_size"],
            distance=distance_map[qcfg["distance"]],
            on_disk=False,                          # 量化向量常驻内存以加速
        ),
        hnsw_config=hnsw_config,
        quantization_config=quantization_config,
        shard_number=qcfg.get("shard_number", 1),
        replication_factor=qcfg.get("replication_factor", 1),
        on_disk_payload=qcfg.get("on_disk_payload", True),
    )


# Payload schema: 论文 §4.4.1 中的字段表
# 字段类型决定 Qdrant 如何为其建索引
PAYLOAD_SCHEMA: dict[str, qm.PayloadSchemaType] = {
    "tile_id":     qm.PayloadSchemaType.KEYWORD,
    "country":     qm.PayloadSchemaType.KEYWORD,
    "labels":      qm.PayloadSchemaType.KEYWORD,    # 多值
    "geohash":     qm.PayloadSchemaType.KEYWORD,
    "acquired_at": qm.PayloadSchemaType.DATETIME,
    "lat":         qm.PayloadSchemaType.FLOAT,
    "lon":         qm.PayloadSchemaType.FLOAT,
}


def create_collection(client: QdrantClient, cfg: dict[str, Any]) -> None:
    """
    建表 + 创建 payload 索引(论文 §4.4.2)。
    使用 recreate_collection 是为了在实验阶段方便迭代;生产环境改为 create_collection。
    """
    qcfg = cfg["qdrant"]
    name = qcfg["collection"]
    params = build_collection_config(cfg)

    logger.info("recreating collection %s", name)
    client.recreate_collection(collection_name=name, **params)

    # 为关键 payload 字段建索引, 让 filter 在 HNSW 搜索过程中高效求值
    index_fields = qcfg.get("payload_index_fields", list(PAYLOAD_SCHEMA.keys()))
    for field in index_fields:
        if field not in PAYLOAD_SCHEMA:
            logger.warning("unknown payload field %s, skip indexing", field)
            continue
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=PAYLOAD_SCHEMA[field],
        )
        logger.info("indexed payload field: %s", field)
