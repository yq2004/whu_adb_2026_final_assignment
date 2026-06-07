"""
数据采样与编码工具。

- stratified_sample: 论文 §5.1.1 的分层抽样
- compute_geohash:   论文 §4.4.1 payload schema 中的 Geohash 字段
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict

import geohash2

from .bigearthnet import PatchRecord

logger = logging.getLogger(__name__)


def stratified_sample(
    records: list[PatchRecord],
    sample_size: int,
    query_size: int,
    seed: int = 42,
) -> tuple[list[PatchRecord], list[PatchRecord]]:
    """
    按 main_label 分层抽样, 返回 (库集, 查询集), 二者严格不重叠。

    保持各类比例与原始分布大体一致, 避免抽样偏差。
    """
    rng = random.Random(seed)
    by_label: dict[str, list[PatchRecord]] = defaultdict(list)
    for r in records:
        by_label[r.main_label].append(r)

    total = len(records)
    pool_target = sample_size + query_size

    pool: list[PatchRecord] = []
    for label, items in by_label.items():
        share = round(len(items) / total * pool_target)
        share = min(share, len(items))
        rng.shuffle(items)
        pool.extend(items[:share])

    rng.shuffle(pool)
    if len(pool) < pool_target:
        logger.warning(
            "stratified pool only %d < requested %d, will use what we have",
            len(pool), pool_target,
        )
    library = pool[:sample_size]
    queries = pool[sample_size:sample_size + query_size]
    logger.info("stratified: library=%d queries=%d", len(library), len(queries))
    return library, queries


def compute_geohash(lat: float, lon: float, precision: int = 6) -> str:
    """
    Geohash 编码;precision=6 对应约 1.2 km × 0.6 km 的格子,
    与 BigEarthNet patch 在地面上的覆盖面积 (1.2 km × 1.2 km) 匹配。
    """
    return geohash2.encode(lat, lon, precision=precision)
