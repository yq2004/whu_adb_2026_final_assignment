"""
Clay v1.5 编码器封装。

对应论文 §4.3.1 模型选择与加载、§4.3.2 嵌入生成流程。

Clay 官方接口(https://clay-foundation.github.io/model/getting-started/quickstart.html):
    from claymodel.module import ClayMAEModule
    model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
    embeddings = model.encoder(chips, timestamps, wavelengths)

输入:
    chips:       [B, C, H, W]   像素张量
    timestamps:  [B, 4]         [week, hour, lat, lon] 归一化后
    wavelengths: [B, C]         各波段中心波长(nm)
输出:
    embeddings:  [B, N, D]      patch tokens, 取 mean/cls 得到 [B, D]
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class ClayEncoder:
    """对 Clay 模型的最小封装, 便于在批处理和在线检索中复用。"""

    def __init__(
        self,
        ckpt_path: str | Path,
        meta_path: str | Path,
        sensor: str = "sentinel-2-l2a",
        device: str = "cuda",
        fp16: bool = True,
        pool: str = "mean",
        l2_normalize: bool = True,
    ):
        # 延迟导入: claymodel 可能未安装(例如在 CI 烟雾测试中)
        from claymodel.module import ClayMAEModule  # type: ignore

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.pool = pool
        self.l2_normalize = l2_normalize
        self.fp16 = fp16 and self.device.type == "cuda"

        logger.info("loading Clay checkpoint: %s", ckpt_path)
        self.model = ClayMAEModule.load_from_checkpoint(str(ckpt_path), map_location=self.device)
        self.model.eval()
        if self.fp16:
            self.model.half()

        # 加载传感器元数据(波长、归一化参数)
        with open(meta_path, "r") as f:
            self.meta = yaml.safe_load(f)
        self.sensor_meta = self.meta[sensor]

        # 各波段中心波长(nm),Clay 期望的输入之一
        self.wavelengths = torch.tensor(
            [
                self.sensor_meta["bands"]["wavelength"][b] * 1000
                for b in self.sensor_meta["band_order"]
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # 归一化参数
        self.band_mean = torch.tensor(
            [self.sensor_meta["bands"]["mean"][b] for b in self.sensor_meta["band_order"]],
            dtype=torch.float32, device=self.device,
        ).view(1, -1, 1, 1)
        self.band_std = torch.tensor(
            [self.sensor_meta["bands"]["std"][b] for b in self.sensor_meta["band_order"]],
            dtype=torch.float32, device=self.device,
        ).view(1, -1, 1, 1)

    # @staticmethod
    # def _build_timestamps(
    #     latlons: list[tuple[float, float]],
    #     dates: list[Optional[str]],
    #     device: torch.device,
    # ) -> torch.Tensor:
    #     """构造 Clay 期望的 [B, 4] 时间戳张量 = [sin(week), cos(week), lat, lon]
    #     实际格式取决于 Clay 版本,这里给出常见实现;若文档接口有变,只需改这里。
    #     """
    #     out = []
    #     for (lat, lon), date_str in zip(latlons, dates):
    #         if date_str:
    #             try:
    #                 d = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    #                 week = d.isocalendar()[1]
    #             except ValueError:
    #                 week = 1
    #         else:
    #             week = 1
    #         week_norm = week / 52.0
    #         # 用 sin/cos 编码周期性
    #         out.append([
    #             math.sin(2 * math.pi * week_norm),
    #             math.cos(2 * math.pi * week_norm),
    #             math.radians(lat),
    #             math.radians(lon),
    #         ])
    #     return torch.tensor(out, dtype=torch.float32, device=device)

    # @torch.inference_mode()
    # def encode(
    #     self,
    #     chips: np.ndarray,                           # [B, C, H, W] float32
    #     latlons: list[tuple[float, float]],
    #     dates: list[Optional[str]],
    # ) -> np.ndarray:
    #     """
    #     批量推理, 返回 [B, D] 的 numpy 嵌入(已可选 L2 归一化)。
    #     """
    #     x = torch.from_numpy(chips).to(self.device)
    #     # 归一化
    #     x = (x - self.band_mean) / self.band_std
    #     if self.fp16:
    #         x = x.half()

    #     ts = self._build_timestamps(latlons, dates, self.device)
    #     wl = self.wavelengths.unsqueeze(0).expand(x.size(0), -1)
    #     if self.fp16:
    #         ts = ts.half()
    #         wl = wl.half()

    #     out = self.model.encoder(x, ts, wl)          # [B, N, D] or [B, D]
    #     if out.dim() == 3:
    #         if self.pool == "cls":
    #             emb = out[:, 0, :]
    #         else:                                    # mean over patch tokens
    #             emb = out.mean(dim=1)
    #     else:
    #         emb = out                                # 某些版本已经做了池化

    #     emb = emb.float()
    #     if self.l2_normalize:
    #         emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    #     return emb.cpu().numpy()

    # @staticmethod
    # def _build_time(
    #     dates: list,
    #     device: torch.device,
    # ) -> torch.Tensor:
    #     """构造 Clay 期望的 [B, 2] 时间张量, 用 sin/cos 编码 ISO 周。"""
    #     import datetime as dt
    #     out = []
    #     for date_str in dates:
    #         if date_str:
    #             try:
    #                 d = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    #                 week = d.isocalendar()[1]
    #             except ValueError:
    #                 week = 1
    #         else:
    #             week = 1
    #         week_norm = week / 52.0
    #         out.append([
    #             math.sin(2 * math.pi * week_norm),
    #             math.cos(2 * math.pi * week_norm),
    #         ])
    #     return torch.tensor(out, dtype=torch.float32, device=device)

    @staticmethod
    def _build_time(
        dates: list,
        device: torch.device,
    ) -> torch.Tensor:
        """构造 Clay 期望的 [B, 4] 时间张量, 用 sin/cos 编码 ISO 周和小时。"""
        import datetime as dt
        out = []
        for date_str in dates:
            if date_str:
                try:
                    d = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    week = d.isocalendar()[1]
                    hour = d.hour
                except ValueError:
                    week, hour = 1, 0
            else:
                week, hour = 1, 0
            week_norm = week / 52.0
            hour_norm = hour / 24.0
            out.append([
                math.sin(2 * math.pi * week_norm),
                math.cos(2 * math.pi * week_norm),
                math.sin(2 * math.pi * hour_norm),
                math.cos(2 * math.pi * hour_norm),
            ])
        return torch.tensor(out, dtype=torch.float32, device=device)

    # @staticmethod
    # def _build_latlon(
    #     latlons: list[tuple[float, float]],
    #     device: torch.device,
    # ) -> torch.Tensor:
    #     """构造 Clay 期望的 [B, 2] 经纬度张量, 用弧度。"""
    #     out = [[math.radians(lat), math.radians(lon)] for lat, lon in latlons]
    #     return torch.tensor(out, dtype=torch.float32, device=device)
    @staticmethod
    def _build_latlon(
        latlons: list[tuple[float, float]],
        device: torch.device,
    ) -> torch.Tensor:
        """构造 Clay 期望的 [B, 4] 经纬度张量, sin/cos 编码以保证连续性。"""
        out = []
        for lat, lon in latlons:
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            out.append([
                math.sin(lat_rad),
                math.cos(lat_rad),
                math.sin(lon_rad),
                math.cos(lon_rad),
            ])
        return torch.tensor(out, dtype=torch.float32, device=device)

    @torch.inference_mode()
    def encode(
        self,
        chips: np.ndarray,                           # [B, C, H, W] float32
        latlons: list[tuple[float, float]],
        dates: list,
        gsd: float = 10.0,                           # Sentinel-2 默认 10 m/px
    ) -> np.ndarray:
        """批量推理, 返回 [B, D] 的 numpy 嵌入(已可选 L2 归一化)。"""
        # 1) 像素张量并归一化
        x = torch.from_numpy(chips).to(self.device)
        x = (x - self.band_mean) / self.band_std
        if self.fp16:
            x = x.half()

        # 2) 时间、经纬度、波长、gsd 按 Clay encoder 要求的形状组装
        time   = self._build_time(dates, self.device)              # [B, 2]
        latlon = self._build_latlon(latlons, self.device)          # [B, 2]
        waves  = self.wavelengths                                  # [N] —— 注意没有 batch 维
        if self.fp16:
            time   = time.half()
            latlon = latlon.half()
            waves  = waves.half()

        # 3) 打包成 datacube 字典, 调用真正的 encoder
        datacube = {
            "pixels": x,
            "time":   time,
            "latlon": latlon,
            "gsd":    torch.tensor(gsd, dtype=torch.float32, device=self.device),
            "waves":  waves,
        }
        out = self.model.model.encoder(datacube)   # 注意是 model.model.encoder

        # 4) Clay encoder 返回值可能是 tensor 或 tuple, 这里做一下兼容
        if isinstance(out, (tuple, list)):
            out = out[0]                            # 通常第一个就是 patch embeddings [B, L, D]

        # 5) 池化为 [B, D]
        if out.dim() == 3:
            if self.pool == "cls":
                emb = out[:, 0, :]                  # cls token
            else:
                emb = out.mean(dim=1)               # mean pooling
        else:
            emb = out

        emb = emb.float()
        if self.l2_normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.cpu().numpy()

    @property
    def embedding_dim(self) -> int:
        """通过一次 dummy forward 探测嵌入维度,避免硬编码。"""
        n_bands = len(self.sensor_meta["band_order"])
        dummy = np.zeros((1, n_bands, 256, 256), dtype=np.float32)
        emb = self.encode(dummy, [(0.0, 0.0)], [None])
        return int(emb.shape[-1])
