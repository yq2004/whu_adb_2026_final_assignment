"""
批量嵌入生成。

对应论文 §4.3.2 嵌入生成流程、§4.2.3 数据组织。

把扫描到的 PatchRecord 列表按 batch 喂给 ClayEncoder, 把嵌入写成 .npy,
把元数据写成 parquet, 二者通过 patch_id 对齐, 便于下游入库与评估复用。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Any, Callable

from ..data.preprocess import compute_geohash
from .clay_encoder import ClayEncoder

logger = logging.getLogger(__name__)

# Record 可以是 PatchRecord(BigEarthNet) 也可以是 EuroSATRecord;
# 两者字段同名, 协议兼容, 这里用 Any 简化类型标注。
Record = Any
BandReader = Callable[[Record, list[str], int], np.ndarray]


def generate_embeddings(
    records: list[Record],
    encoder: ClayEncoder,
    band_order: list[str],
    target_size: int,
    batch_size: int,
    out_dir: Path,
    split_name: str,
    band_reader: BandReader | None = None,
) -> None:
    """
    批量生成嵌入, 写入:
        {out_dir}/{split_name}_embeddings.npy   shape [N, D]
        {out_dir}/{split_name}_metadata.parquet 一行一 patch

    设计要点(论文 §4.3.2):
    - 按 batch_size 组装 chips, latlons, dates;
    - 跳过读取失败的 patch, 不打断流水线;
    - patch_id 严格对齐, 便于复现与评估。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 默认使用 BigEarthNet 的波段读取函数, 调用方也可注入 EuroSAT 的
    if band_reader is None:
        from ..data.bigearthnet import load_patch_bands
        band_reader = load_patch_bands

    all_emb: list[np.ndarray] = []
    meta_rows: list[dict] = []

    batch_chips: list[np.ndarray] = []
    batch_latlons: list[tuple[float, float]] = []
    batch_dates: list[str | None] = []
    batch_records: list[Record] = []

    def flush():
        if not batch_chips:
            return
        chips = np.stack(batch_chips, axis=0)
        emb = encoder.encode(chips, batch_latlons, batch_dates)
        all_emb.append(emb)
        for r in batch_records:
            meta_rows.append({
                "patch_id": r.patch_id,
                "tile_id": r.tile_id,
                "country": r.country,
                "lat": r.lat,
                "lon": r.lon,
                "geohash": compute_geohash(r.lat, r.lon, precision=6),
                "acquired_at": r.acquired_at,
                "labels": r.labels,
                "main_label": r.main_label,
            })
        batch_chips.clear()
        batch_latlons.clear()
        batch_dates.clear()
        batch_records.clear()

    for r in tqdm(records, desc=f"embedding {split_name}"):
        try:
            chips = band_reader(r, band_order, target_size)
        except Exception as e:
            logger.warning("skip %s: %s", r.patch_id, e)
            continue
        batch_chips.append(chips)
        batch_latlons.append((r.lat, r.lon))
        batch_dates.append(r.acquired_at or None)
        batch_records.append(r)
        if len(batch_chips) >= batch_size:
            flush()
    flush()

    if not all_emb:
        raise RuntimeError("no embeddings produced — check data and Clay setup")

    embeddings = np.concatenate(all_emb, axis=0)
    np.save(out_dir / f"{split_name}_embeddings.npy", embeddings)
    pd.DataFrame(meta_rows).to_parquet(out_dir / f"{split_name}_metadata.parquet")
    logger.info(
        "wrote %d embeddings (dim=%d) to %s",
        embeddings.shape[0], embeddings.shape[1], out_dir,
    )
