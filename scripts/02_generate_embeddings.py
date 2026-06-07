"""
Step 2:对库集与查询集分别生成 Clay v1.5 嵌入。

用法:
    python scripts/02_generate_embeddings.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.bigearthnet import PatchRecord, load_patch_bands
from src.data.eurosat import load_patch_bands_compat as load_eurosat_bands_compat
from src.embedding.batch_embed import generate_embeddings
from src.embedding.clay_encoder import ClayEncoder


# 支持的数据集 → 波段读取函数
BAND_READERS = {
    "bigearthnet": load_patch_bands,
    "eurosat":     load_eurosat_bands_compat,
}


def load_jsonl(path: Path) -> list[PatchRecord]:
    records: list[PatchRecord] = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            records.append(PatchRecord(
                patch_id=d["patch_id"],
                patch_dir=Path(d["patch_dir"]),
                tile_id=d["tile_id"],
                country=d.get("country"),
                lat=d["lat"],
                lon=d["lon"],
                acquired_at=d.get("acquired_at", ""),
                labels=d["labels"],
                main_label=d["main_label"],
            ))
    return records


def main(cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    processed = Path(cfg["paths"]["processed_dir"])
    emb_dir = Path(cfg["paths"]["embeddings_dir"])

    dataset_name = cfg["data"].get("dataset", "bigearthnet").lower()
    if dataset_name not in BAND_READERS:
        raise ValueError(f"unknown dataset '{dataset_name}', supported: {list(BAND_READERS)}")
    band_reader = BAND_READERS[dataset_name]
    logging.info("dataset = %s, band_reader = %s", dataset_name, band_reader.__module__)

    encoder = ClayEncoder(
        ckpt_path=cfg["paths"]["clay_ckpt"],
        meta_path=cfg["paths"]["clay_meta"],
        sensor=cfg["data"]["sensor"],
        device=cfg["embedding"]["device"],
        fp16=cfg["embedding"]["fp16"],
        pool=cfg["embedding"]["pool"],
        l2_normalize=cfg["embedding"]["l2_normalize"],
    )

    for split in ("library", "queries"):
        records = load_jsonl(processed / f"{split}.jsonl")
        generate_embeddings(
            records=records,
            encoder=encoder,
            band_order=cfg["data"]["band_order"],
            target_size=cfg["data"]["target_size"],
            batch_size=cfg["embedding"]["batch_size"],
            out_dir=emb_dir,
            split_name=split,
            band_reader=band_reader,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
