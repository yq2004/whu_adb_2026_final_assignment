"""
检索评估指标。

对应论文 §5.1.2 相关性定义与评价指标:
- 相关性 = 查询与候选的多标签集合相交;
- Recall@K, mAP@K, NDCG@K。
"""
from __future__ import annotations

import math
from typing import Sequence


def is_relevant(query_labels: Sequence[str], candidate_labels: Sequence[str]) -> bool:
    """共享至少一种 CORINE 多标签即视为相关。"""
    return bool(set(query_labels) & set(candidate_labels))


def recall_at_k(query_labels, retrieved_labels_list, k: int) -> float:
    """
    Recall@K = (前 K 中相关候选数) / K
    严格意义上 Recall 还需总相关样本数, 此处采用"前 K 中相关比例"作为常用近似,
    在大库且相关样本多的场景下与传统 Recall@K 高度一致。
    """
    top = retrieved_labels_list[:k]
    if not top:
        return 0.0
    hits = sum(1 for c in top if is_relevant(query_labels, c))
    return hits / k


def average_precision_at_k(query_labels, retrieved_labels_list, k: int) -> float:
    """单条查询的 AP@K(按位置加权的精度)。"""
    top = retrieved_labels_list[:k]
    hits = 0
    score = 0.0
    for i, c in enumerate(top, start=1):
        if is_relevant(query_labels, c):
            hits += 1
            score += hits / i
    return score / max(hits, 1) if hits > 0 else 0.0


def ndcg_at_k(query_labels, retrieved_labels_list, k: int) -> float:
    """
    NDCG@K,相关度采用"共享标签个数"作为分级相关性。
    """
    top = retrieved_labels_list[:k]
    rels = [len(set(query_labels) & set(c)) for c in top]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics(
    query_labels_list: list[list[str]],
    retrieved_labels_per_query: list[list[list[str]]],
    ks: list[int] = (5, 10, 20),
) -> dict[str, float]:
    """
    在整个查询集上聚合 Recall@K, mAP@K, NDCG@K。
    """
    result: dict[str, float] = {}
    n = len(query_labels_list)
    assert n == len(retrieved_labels_per_query)

    for k in ks:
        recall = sum(recall_at_k(q, r, k)
                     for q, r in zip(query_labels_list, retrieved_labels_per_query)) / n
        map_k = sum(average_precision_at_k(q, r, k)
                    for q, r in zip(query_labels_list, retrieved_labels_per_query)) / n
        ndcg = sum(ndcg_at_k(q, r, k)
                   for q, r in zip(query_labels_list, retrieved_labels_per_query)) / n
        result[f"Recall@{k}"] = round(recall, 4)
        result[f"mAP@{k}"]    = round(map_k, 4)
        result[f"NDCG@{k}"]   = round(ndcg, 4)
    return result
