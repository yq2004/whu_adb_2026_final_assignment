"""
烟雾测试:不依赖 Clay 与 Qdrant 服务, 验证纯逻辑模块正确。

跑法:
    pytest tests/ -v
或:
    python tests/test_smoke.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# 让 tests/ 能 import 项目内代码
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    aggregate_metrics, is_relevant,
    average_precision_at_k, ndcg_at_k, recall_at_k,
)
from src.retrieval.rerank import compute_proximity_score, haversine_km


def test_is_relevant():
    assert is_relevant(["Urban fabric"], ["Urban fabric", "Inland waters"])
    assert not is_relevant(["Urban fabric"], ["Marine waters"])


def test_recall_at_k():
    q = ["Arable land"]
    # 前 5 个中 4 个相关
    retrieved = [
        ["Arable land", "Pastures"],
        ["Inland waters"],
        ["Arable land"],
        ["Arable land", "Coniferous forest"],
        ["Arable land"],
        ["Marine waters"],
    ]
    assert math.isclose(recall_at_k(q, retrieved, 5), 4 / 5)
    assert math.isclose(recall_at_k(q, retrieved, 3), 2 / 3)


def test_ap_at_k_perfect():
    q = ["X"]
    retrieved = [["X"], ["X"], ["X"]]
    assert math.isclose(average_precision_at_k(q, retrieved, 3), 1.0)


def test_ap_at_k_partial():
    q = ["X"]
    retrieved = [["Y"], ["X"], ["X"]]   # 第 2、3 位命中
    # AP = (0 + 1/2 + 2/3) / 2 = 0.5833...
    ap = average_precision_at_k(q, retrieved, 3)
    assert math.isclose(ap, (1/2 + 2/3) / 2, rel_tol=1e-6)


def test_ndcg_at_k_monotonic():
    q = ["X", "Y"]
    # 第一种排序更好(命中在前)
    good = [["X", "Y"], ["X"], ["Z"]]
    bad  = [["Z"],     ["X"], ["X", "Y"]]
    assert ndcg_at_k(q, good, 3) > ndcg_at_k(q, bad, 3)


def test_aggregate_metrics_shape():
    queries = [["A"], ["B"]]
    retrieved = [[["A"], ["C"]], [["B"], ["B", "D"]]]
    m = aggregate_metrics(queries, retrieved, ks=[2])
    assert "Recall@2" in m and "mAP@2" in m and "NDCG@2" in m


def test_haversine_known_distance():
    # 北京到上海大约 1067 km
    d = haversine_km(39.9042, 116.4074, 31.2304, 121.4737)
    assert 1050 < d < 1100


def test_proximity_score_decay():
    p0 = compute_proximity_score((0, 0), (0, 0), decay_km=50)
    p1 = compute_proximity_score((0, 0), (0, 0.45), decay_km=50)  # ~50 km @ 赤道
    assert math.isclose(p0, 1.0, abs_tol=1e-6)
    assert 0.3 < p1 < 0.45


if __name__ == "__main__":
    # 简易 runner
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(0 if failed == 0 else 1)
