"""
加权重排序工具:把地理距离归一化为 [0, 1] 的"邻近度得分"。

对应论文 §4.5.3 的 proximity_geo 计算。
"""
from __future__ import annotations

import math


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 大圆距离(km)。"""
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def compute_proximity_score(
    q_latlon: tuple[float, float],
    c_latlon: tuple[float, float],
    decay_km: float = 50.0,
) -> float:
    """
    proximity = exp(- distance_km / decay_km)

    decay_km 越小, 对距离越敏感;decay_km = 50 时:
        0 km   → 1.000
        50 km  → 0.368
        200 km → 0.018
    与语义相似度(余弦)同处 [0, 1] 区间, 适合直接做加权融合。
    """
    d = haversine_km(*q_latlon, *c_latlon)
    return math.exp(-d / decay_km)
