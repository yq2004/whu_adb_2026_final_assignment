"""
Step 1 (EuroSAT 版):扫描 EuroSAT, 分层抽样, 输出库集 + 查询集清单。

EuroSAT 是单标签数据集, main_label 就是它的唯一类别。

用法:
    python scripts/01_preprocess_eurosat.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml


# from ..src.data.eurosat import scan_eurosat
# from ..src.data.preprocess import stratified_sample
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.eurosat import scan_eurosat
from src.data.preprocess import stratified_sample


def main(cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = scan_eurosat(raw_dir)
    if not records:
        raise RuntimeError(
            f"no EuroSAT samples found under {raw_dir}.\n"
            "Expected layout: <raw_dir>/AnnualCrop/AnnualCrop_1.tif ...\n"
            "Download: https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip"
        )

    library, queries = stratified_sample(
        records,
        sample_size=cfg["data"]["sample_size"],
        query_size=cfg["data"]["query_size"],
        seed=cfg["data"]["random_seed"],
    )

    def dump(name: str, rs):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps({
                    "patch_id": r.patch_id,
                    "patch_dir": str(r.patch_dir),
                    "tile_id": r.tile_id,
                    "country": r.country,
                    "lat": r.lat,
                    "lon": r.lon,
                    "acquired_at": r.acquired_at,
                    "labels": r.labels,
                    "main_label": r.main_label,
                }) + "\n")
        logging.info("wrote %s (%d records)", path, len(rs))

    dump("library", library)
    dump("queries", queries)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
