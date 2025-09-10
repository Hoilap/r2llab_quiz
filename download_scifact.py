# 下载 SciFact 数据集脚本
from beir import util, LoggingHandler
import logging
import pathlib
import os

logging.basicConfig(level=logging.INFO)

data_path = os.path.join(os.path.dirname(__file__), "datasets")
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(data_path, dataset)

if not os.path.exists(out_dir):
    print(f"Downloading {dataset} dataset...")
    data_dir = util.download_and_unzip(url, data_path)
    print(f"Downloaded and extracted to {data_dir}")
else:
    print(f"Dataset already exists at {out_dir}")
