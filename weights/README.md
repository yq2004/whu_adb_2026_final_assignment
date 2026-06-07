# weights — Clay v1.5 模型权重

本目录存放 Clay 地理空间基础模型的权重文件（~1.25 GB），因体积较大未上传至仓库。

## 下载方式

```bash
mkdir -p weights

# Clay v1.5 权重（来自 HuggingFace）
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt \
     -P weights/

# Clay 传感器元数据（来自 Clay 官方仓库）
wget https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml \
     -P weights/
```

## 文件说明

| 文件 | 大小 | 说明 |
|---|---|---|
| `clay-v1.5.ckpt` | ~1.25 GB | PyTorch Lightning checkpoint，Clay MAE 编码器权重 |
| `metadata.yaml` | < 1 KB | 各传感器的波段波长与归一化参数，**已随仓库提供** |

## Clay 安装

除权重外，还需安装 Clay Python 包：

```bash
pip install git+https://github.com/Clay-foundation/model.git
```

详见 [Clay 官方文档](https://clay-foundation.github.io/model/getting-started/quickstart.html)。
