"""
Step 3:在 Qdrant 中建 collection、建 payload 索引、把库集 upsert 进去。

用法:
    python scripts/03_ingest_qdrant.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db.qdrant_client import ingest, make_client
from src.db.schema import create_collection


def main(cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    embeddings = np.load(emb_dir / "library_embeddings.npy")
    metadata = pd.read_parquet(emb_dir / "library_metadata.parquet")

    # 修正 config 中的 vector_size, 让其与实际嵌入维度对齐
    if embeddings.shape[1] != cfg["qdrant"]["vector_size"]:
        logging.warning(
            "config vector_size=%d, actual=%d — overriding",
            cfg["qdrant"]["vector_size"], embeddings.shape[1],
        )
        cfg["qdrant"]["vector_size"] = int(embeddings.shape[1])

    client = make_client(cfg["qdrant"]["url"])
    create_collection(client, cfg)
    ingest(client, cfg["qdrant"]["collection"], embeddings, metadata)
    logging.info("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
