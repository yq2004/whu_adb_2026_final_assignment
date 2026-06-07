"""
BigEarthNet 数据加载器。

对应论文 §4.2.1 数据集选择、§4.2.2 影像预处理、§4.2.3 数据组织。

BigEarthNet-S2 每个 patch 是一个目录,目录名形如:
    S2A_MSIL2A_20170613T101031_0_45/
内含:
    *_B01.tif ~ *_B12.tif      (12 个波段, 不同分辨率)
    *_labels_metadata.json     (CORINE 多标签 + 经纬度等)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

# CORINE 19-class nomenclature 主要标签
# 论文 §4.2.1 提到的"经过类别整理后,数据集采用 19 类的精简类别体系"
CORINE_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]


@dataclass
class PatchRecord:
    """单个 BigEarthNet patch 的轻量元数据(用于建立索引清单,不持有像素)。"""
    patch_id: str
    patch_dir: Path
    tile_id: str               # 所属 Sentinel-2 瓦片
    country: Optional[str]
    lat: float                 # 中心经纬度
    lon: float
    acquired_at: str           # ISO datetime
    labels: list[str]          # CORINE 19 类多标签
    main_label: str            # 用于分层抽样的主标签(出现频次最高者)


def _read_metadata_json(meta_path: Path) -> dict:
    with open(meta_path, "r") as f:
        return json.load(f)


def scan_bigearthnet(raw_dir: Path) -> list[PatchRecord]:
    """
    扫描 BigEarthNet 根目录, 抽取每个 patch 的元数据, 返回 PatchRecord 列表。

    注意: 不读取像素, 速度很快, 适合先建清单再分批处理。
    """
    raw_dir = Path(raw_dir)
    records: list[PatchRecord] = []
    for patch_dir in sorted(raw_dir.iterdir()):
        if not patch_dir.is_dir():
            continue
        meta_files = list(patch_dir.glob("*_labels_metadata.json"))
        if not meta_files:
            logger.warning("skip %s: no metadata json", patch_dir.name)
            continue
        meta = _read_metadata_json(meta_files[0])

        labels = meta.get("labels", [])
        if not labels:
            continue
        # 主标签: 出现在 CORINE_19_CLASSES 顺序中最靠前的(确定性, 便于分层抽样)
        main = next((c for c in CORINE_19_CLASSES if c in labels), labels[0])

        # 中心经纬度: BigEarthNet metadata 里有 corners (ulx, uly, lrx, lry) in EPSG:32xxx
        # 真实代码需要 pyproj 把 UTM 转 WGS84;此处假设 metadata 已经存了 wgs84 坐标
        coords = meta.get("coordinates", {})
        lat = float(coords.get("center_lat", 0.0))
        lon = float(coords.get("center_lon", 0.0))

        records.append(PatchRecord(
            patch_id=patch_dir.name,
            patch_dir=patch_dir,
            tile_id=patch_dir.name.split("_MSIL2A_")[0],
            country=meta.get("country"),
            lat=lat,
            lon=lon,
            acquired_at=meta.get("acquisition_date", ""),
            labels=labels,
            main_label=main,
        ))
    logger.info("scanned %d patches from %s", len(records), raw_dir)
    return records


def load_patch_bands(
    record: PatchRecord,
    band_order: list[str],
    target_size: int = 256,
) -> np.ndarray:
    """
    读取并对齐一个 patch 的指定波段, 返回 [C, H, W] 的 float32 数组。

    实现要点(论文 §4.2.2):
    - 不同分辨率波段统一上采样到 10m 网格;
    - 然后整体重采样到 target_size × target_size, 以匹配 Clay 的输入尺寸;
    - 输出 reflectance 值 (Sentinel-2 L2A 通常已是 0-10000 区间, 不在这里做归一化,
      归一化由 Clay 的传感器元数据驱动, 见 src/embedding/clay_encoder.py)。
    """
    bands: list[np.ndarray] = []
    for band_name in band_order:
        tif_path = next(record.patch_dir.glob(f"*_{band_name}.tif"), None)
        if tif_path is None:
            raise FileNotFoundError(f"{band_name} missing for {record.patch_id}")
        with rasterio.open(tif_path) as src:
            data = src.read(
                1,
                out_shape=(target_size, target_size),
                resampling=Resampling.bilinear,
            )
        bands.append(data.astype(np.float32))
    return np.stack(bands, axis=0)   # [C, H, W]
