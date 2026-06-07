# data/embeddings — Clay 嵌入向量

本目录存放由 Clay v1.5 编码器生成的嵌入文件（二进制格式，~87 MB），未上传至仓库。

## 文件说明

| 文件 | 内容 |
|---|---|
| `library_embeddings.npy` | 库集影像的 768 维嵌入矩阵，shape = [20000, 768] |
| `library_metadata.parquet` | 库集元数据（patch_id、labels、lat、lon 等） |
| `queries_embeddings.npy` | 查询集影像的嵌入矩阵，shape = [2000, 768] |
| `queries_metadata.parquet` | 查询集元数据 |

## 如何生成

在完成原始数据下载（见 `data/raw/README.md`）和 Clay 权重下载（见 `weights/README.md`）后，依次执行：

```bash
python scripts/01_preprocess_eurosat.py --config configs/config.yaml
python scripts/02_generate_embeddings.py --config configs/config.yaml
```
