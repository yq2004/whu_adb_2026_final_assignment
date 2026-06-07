"""
EuroSAT MSI 数据加载器。

EuroSAT (https://github.com/phelber/EuroSAT) 是 BigEarthNet 的轻量替代:
- 27,000 张 Sentinel-2 影像(原始 GeoTIFF, 带地理参考)
- 64×64 像素, 10m/px 分辨率, 地面约 640m × 640m
- 13 个 Sentinel-2 波段
- 单标签, 10 类(Annual Crop, Forest, Industrial, River, SeaLake 等)
- 来自欧洲 34 个国家, 多样性充足
- 总大小约 2 GB

数据下载:
    wget https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip
    unzip 后路径形如:
        data/raw/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/
            AnnualCrop/AnnualCrop_1.tif ... AnnualCrop_3000.tif
            Forest/Forest_1.tif ...
            ... (10 个类目录)

对应论文 §4.2.1 数据集选择 / §4.2.2 影像预处理。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)


# EuroSAT 10 类(单标签)
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# 13 个 Sentinel-2 波段, EuroSAT 把它们按以下顺序存为 GeoTIFF 的 13 个 channel
# 顺序: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
EUROSAT_BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]


@dataclass
class EuroSATRecord:
    """
    单张 EuroSAT 影像的元数据。字段与 PatchRecord 同名, 便于复用 batch_embed。
    """
    patch_id: str          # 形如 "Forest_1234"
    patch_dir: Path        # 这里实际是 .tif 文件路径
    tile_id: str           # EuroSAT 无明确 tile, 用类别名占位
    country: Optional[str] # EuroSAT 标注笼统(34 国总称), 留空
    lat: float             # 中心经纬度(从 GeoTIFF 解析)
    lon: float
    acquired_at: str       # EuroSAT 取景于 2017, 用占位时间戳
    labels: list[str]      # 单标签包成 list, 与下游评估接口兼容
    main_label: str        # 同 labels[0]


def _read_center_latlon(tif_path: Path) -> tuple[float, float]:
    """
    从 GeoTIFF 头里读取中心像素的经纬度。
    EuroSAT 的 .tif 用 UTM 投影(EPSG:326xx), 需要转换到 WGS84(EPSG:4326)。
    """
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        # 中心像素对应的投影坐标
        cx, cy = src.transform * (w / 2, h / 2)
        src_crs = src.crs

    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return float(lat), float(lon)


def scan_eurosat(
    raw_dir: Path,
    acquired_at: str = "2017-08-01T00:00:00Z",
) -> list[EuroSATRecord]:
    """
    扫描 EuroSAT 根目录, 抽出每张 .tif 的元数据。

    raw_dir 应当指向包含 10 个类目录的目录, 例如:
        data/raw/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/
    """
    raw_dir = Path(raw_dir)
    records: list[EuroSATRecord] = []

    for cls in EUROSAT_CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.is_dir():
            logger.warning("missing class dir: %s", cls_dir)
            continue
        for tif_path in sorted(cls_dir.glob(f"{cls}_*.tif")):
            try:
                lat, lon = _read_center_latlon(tif_path)
            except Exception as e:
                logger.warning("skip %s: cannot read georef (%s)", tif_path.name, e)
                continue
            records.append(EuroSATRecord(
                patch_id=tif_path.stem,                   # e.g. "Forest_1234"
                patch_dir=tif_path,                       # 这里直接是 .tif 路径
                tile_id=cls,                              # 用类别名占位
                country=None,
                lat=lat,
                lon=lon,
                acquired_at=acquired_at,
                labels=[cls],
                main_label=cls,
            ))
    logger.info("scanned %d EuroSAT samples from %s", len(records), raw_dir)
    return records


def load_eurosat_bands(
    record: EuroSATRecord,
    band_indices: Optional[list[int]] = None,
    target_size: int = 256,
) -> np.ndarray:
    """
    读取一张 EuroSAT .tif 的指定波段并对齐到 target_size。

    band_indices: 13 个波段中要选哪些(从 1 开始的索引);
                  默认全部 13 个。
    返回 [C, H, W] 的 float32 数组。

    与 BigEarthNet 不同, EuroSAT 一个 .tif 里就包含全部 13 通道,
    所以只需一次 src.read 即可, 不需要"波段对齐"步骤。
    """
    if band_indices is None:
        band_indices = list(range(1, 14))   # 1..13

    with rasterio.open(record.patch_dir) as src:
        data = src.read(
            indexes=band_indices,
            out_shape=(len(band_indices), target_size, target_size),
            resampling=Resampling.bilinear,
        )
    return data.astype(np.float32)        # [C, H, W]


# ─────────────────────────────────────────────────────────────
# 与 PatchRecord 协议兼容:让 batch_embed.py 能直接接受 EuroSATRecord
# 只需提供一个 load_patch_bands_compat 与 BigEarthNet 同签名即可
# ─────────────────────────────────────────────────────────────

def load_patch_bands_compat(
    record: EuroSATRecord,
    band_order: list[str],
    target_size: int = 256,
) -> np.ndarray:
    """
    与 src.data.bigearthnet.load_patch_bands 同名同签名的适配器。

    band_order 用波段名字符串(如 ["B02","B03","B04","B08"])指定要取哪些波段;
    本函数会把它翻译成 EuroSAT 13 通道里对应的 channel 索引。
    """
    name_to_index = {name: i + 1 for i, name in enumerate(EUROSAT_BAND_ORDER)}
    band_indices = []
    for name in band_order:
        if name not in name_to_index:
            raise KeyError(f"unknown band {name}, available: {EUROSAT_BAND_ORDER}")
        band_indices.append(name_to_index[name])
    return load_eurosat_bands(record, band_indices=band_indices, target_size=target_size)
