# data/raw — 原始遥感影像

本目录存放原始卫星影像文件，因体积较大（~9 GB）未上传至仓库，请按以下步骤自行下载。

## EuroSAT MSI（默认，~2 GB）

```bash
mkdir -p data/raw
wget https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip \
     -P data/raw/
unzip data/raw/EuroSATallBands.zip -d data/raw/eurosat/
```

解压后目录结构应为：

```
data/raw/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

## BigEarthNet-S2（可选，~59 GB）

从 [BigEarthNet 官网](https://bigearth.net/) 申请并下载 `BigEarthNet-S2` 数据集，
解压到 `data/raw/BigEarthNet-S2/`，然后将 `configs/config.yaml` 中
`data.dataset` 改为 `bigearthnet`，`paths.raw_dir` 指向该目录即可。
