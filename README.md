# geo-semantic-db

面向地理空间领域的语义数据库系统——论文《语义数据库技术在空间领域应用的思考和实践》第四章的参考实现。

以 **EuroSAT MSI**(默认, ~2 GB)或 **BigEarthNet-S2**(可选, ~59 GB)为数据源,
以 **Clay v1.5** 地理空间基础模型为嵌入提取器,
以 **Qdrant** 为向量数据库,
实现"按内容找内容"的遥感影像相似检索,并支持空间—属性混合过滤与加权重排序。

## 重要观念

**本系统不训练任何模型。** Clay 模型已由 Clay Foundation 在全球遥感数据上预训练并开源,本工作把它作为"嵌入提取器"——把每张影像编码为 768 维语义向量,再以 Qdrant 建索引、供检索。EuroSAT/BigEarthNet 数据集的角色是:
1. **被检索的影像库**(每张图编成向量,塞进 Qdrant)
2. **模拟查询**(留一小部分不入库,当用户的查询请求)
3. **评估时的相关性标准**(用类别标签判断检索结果对不对)

## 与论文章节对应

| 论文章节 | 代码位置 |
| --- | --- |
| 4.2 数据预处理 | `src/data/eurosat.py`(默认)/ `src/data/bigearthnet.py`、`preprocess.py` |
| 4.3 Clay 嵌入生成 | `src/embedding/clay_encoder.py`、`batch_embed.py` |
| 4.4 Qdrant 存储与索引 | `src/db/schema.py`、`qdrant_client.py` |
| 4.5 空间—语义混合检索 | `src/retrieval/search.py`、`rerank.py`、`src/api/server.py` |
| 第 5 章 实验评估 | `src/evaluation/metrics.py`、`scripts/04_run_eval.py` |

## 离线流水线

```
原始 EuroSAT / BigEarthNet patches
        │
        ▼
[01_preprocess_*.py]   扫描、分层抽样 → library.jsonl + queries.jsonl
        │
        ▼
[02_generate_embeddings.py]  Clay 编码器批量推理 → embeddings.npy + metadata.parquet
        │
        ▼
[03_ingest_qdrant.py]  向量 + payload 批量入库, 建 HNSW + 量化索引 → Qdrant
```

## 在线检索

```
查询 (patch_id / 影像 / 经纬度+时间)
        │
        ▼
[FastAPI: /search]
   ├── Clay encoder 生成查询向量 (按需在线)
   ├── 构造 Qdrant Filter (类别 / 标签 / 时间 / 地理 bbox)
   ├── Qdrant HNSW + payload 过滤检索 top_k × 5
   ├── (可选) 应用层精确空间过滤 (任意多边形, shapely)
   ├── (可选) 加权重排序  S = w_sem · sim_sem + w_geo · proximity_geo
   └── 返回 top_k
```

## 快速开始(EuroSAT)

```bash
# 1. 启动 Qdrant
docker compose up -d qdrant

# 2. 安装依赖
pip install -r requirements.txt
pip install git+https://github.com/Clay-foundation/model.git   # claymodel

# 3. 下载 Clay v1.5 权重 (1.25 GB)
mkdir -p weights
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt -P weights/
# Clay 元数据 (随 Clay 仓库提供):
wget https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml -P weights/

# 4. 下载 EuroSAT GeoTIFF 完整版 (~2 GB)
mkdir -p data/raw
wget https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip
unzip EuroSATallBands.zip -d data/raw/eurosat/

# 5. 跑离线流水线
python scripts/01_preprocess_eurosat.py    --config configs/config.yaml
python scripts/02_generate_embeddings.py   --config configs/config.yaml
python scripts/03_ingest_qdrant.py         --config configs/config.yaml

# 6. 启动检索服务
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 7. 跑评估实验
python scripts/04_run_eval.py --config configs/config.yaml
```

## 切换到 BigEarthNet

只需两步:
1. 把 `configs/config.yaml` 中的 `data.dataset` 改为 `bigearthnet`,`paths.raw_dir` 指向 BigEarthNet 根目录。
2. 用 `scripts/01_preprocess.py` 替代 `01_preprocess_eurosat.py`。

step 2/3/4 完全不变,这是因为两套数据走的是同一个抽象(`PatchRecord` 与 `EuroSATRecord` 字段一致)。

## 数据集差异(选择参考)

| 维度 | EuroSAT MSI(默认) | BigEarthNet-S2 |
| --- | --- | --- |
| 大小 | ~2 GB | ~59 GB |
| 样本数 | 27,000 | 590,326 |
| 像素 | 64×64 | 120×120(10m 波段) |
| 地面覆盖 | 640m × 640m | 1.2km × 1.2km |
| 标签 | **单标签**, 10 类 | **多标签**, 19 类 |
| 地理覆盖 | 欧洲 34 国 | 欧洲 10 国 |
| 数据格式 | 单 .tif 含 13 通道, 自带 GeoTIFF 坐标 | 12 个分波段 .tif + JSON 元数据 |

论文 §5.1.2 中的相关性定义在两者上略有差异:
- EuroSAT 单标签 → "同类即相关" (set 交集自然退化)
- BigEarthNet 多标签 → "共享至少一种标签即相关"
代码层面 `is_relevant()` 用 set 交集判断, 两种数据天然兼容, 无需改动。

## 配置

所有可调参数集中在 `configs/config.yaml`,包括数据集类型、路径、Clay 模型路径、
Qdrant 连接参数、HNSW/量化策略、批大小、评估参数等。

## 许可

代码基于 Apache-2.0;Clay 权重 Apache-2.0;EuroSAT MIT;BigEarthNet 遵守其原始许可。
