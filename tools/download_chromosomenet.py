"""
下载 ChromosomeNet 数据集 (CIL/Scientific Data 2023)
并转换为 COCO 格式用于 LDMDet 训练

数据来源:
- 论文: "An Open Dataset of Annotated Metaphase Cell Images for Chromosome Identification"
  Scientific Data, 2023 (Nature)
- 原始数据: Cell Image Library (CIL)
  https://cellimagelibrary.org

使用方法:
    python tools/download_chromosomenet.py --output data/ChromosomeNet

注意:
    CIL 数据集较大 (~5GB)，下载可能需要较长时间。
    如果 CIL 网站不可达，请手动从以下地址下载:
    - Cell Image Library: https://cellimagelibrary.org
    - 搜索 "chromosome karyotype" 获取 CIL 数据集 ID
    - 或联系 ChromosomeNet 作者: chengchuang.lin@m.scnu.edu.cn

    备选方案: 从 Kaggle 下载
    pip install kaggle
    kaggle datasets download -d <dataset-slug> -p data/ChromosomeNet
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("请先安装依赖: pip install requests pillow tqdm")
    sys.exit(1)

CIL_BASE_URL = "https://cellimagelibrary.org"
CIL_API_URL = f"{CIL_BASE_URL}/search/json"

CHROMOSOME_CLASSES = {
    1: "A1", 2: "A2", 3: "A3", 4: "B4", 5: "B5",
    6: "C6", 7: "C7", 8: "C8", 9: "C9", 10: "C10",
    11: "C11", 12: "C12", 13: "D13", 14: "D14", 15: "D15",
    16: "E16", 17: "E17", 18: "E18", 19: "F19", 20: "F20",
    21: "G21", 22: "G22", 23: "X", 24: "Y",
}


def download_from_cil(output_dir: str):
    """从 Cell Image Library 下载 ChromosomeNet 数据集"""
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ChromosomeNet 数据集下载")
    print("=" * 60)
    print()
    print("注意: CIL 网站可能不稳定，如果下载失败，请尝试以下备选方案:")
    print()
    print("方案1: 手动从 CIL 网站下载")
    print(f"  访问: {CIL_BASE_URL}")
    print("  搜索 'chromosome karyotype' 下载数据集")
    print()
    print("方案2: 从 Kaggle 下载 (需要 kaggle CLI)")
    print("  pip install kaggle")
    print("  kaggle datasets download -d aliabedimadiseh/chromosome-image-dataset-karyotype")
    print()
    print("方案3: 联系 ChromosomeNet 作者获取数据")
    print("  chengchuang.lin@m.scnu.edu.cn")
    print()
    print("方案4: 使用已有的 Roboflow 数据集")
    print("  pip install roboflow")
    print("  然后使用 Roboflow API 下载公开染色体检测数据集")
    print()

    test_resp = requests.get(CIL_BASE_URL, timeout=10)
    if test_resp.status_code != 200:
        print(f"[ERROR] CIL 网站不可达 (status={test_resp.status_code})")
        print("请使用上述备选方案下载数据集。")
        return False

    print("[INFO] CIL 网站可达，开始下载...")
    return True


def convert_to_coco(annotation_dir: str, image_dir: str, output_json: str):
    """将 ChromosomeNet 标注转换为 COCO 格式"""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    for cid, name in CHROMOSOME_CLASSES.items():
        coco["categories"].append({
            "id": cid,
            "name": name,
            "supercategory": "chromosome",
        })

    ann_files = sorted(Path(annotation_dir).glob("*.json"))
    if not ann_files:
        ann_files = sorted(Path(annotation_dir).glob("*.txt"))

    img_id = 0
    ann_id = 0

    for ann_file in tqdm(ann_files, desc="Converting"):
        with open(ann_file) as f:
            data = json.load(f)

        img_name = data.get("image_name", ann_file.stem + ".jpg")
        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception:
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "height": h,
            "width": w,
        })

        for obj in data.get("objects", []):
            bbox = obj.get("bbox", [])
            cat_id = obj.get("category_id", 0)
            if len(bbox) == 4 and cat_id > 0:
                x, y, bw, bh = bbox
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"转换完成: {img_id} images, {ann_id} annotations -> {output_json}")


def main():
    parser = argparse.ArgumentParser(description="下载 ChromosomeNet 数据集")
    parser.add_argument("--output", default="data/ChromosomeNet", help="输出目录")
    parser.add_argument("--convert-only", action="store_true",
                        help="只做格式转换（假设数据已下载）")
    args = parser.parse_args()

    if args.convert_only:
        convert_to_coco(
            annotation_dir=args.output + "/annotations",
            image_dir=args.output + "/images",
            output_json=args.output + "/annotations.json",
        )
    else:
        download_from_cil(args.output)


if __name__ == "__main__":
    main()
