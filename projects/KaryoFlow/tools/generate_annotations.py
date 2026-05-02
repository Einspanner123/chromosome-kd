"""
从 COCO 标注生成 KaryoFlow 排列标注

读取 COCO 格式的染色体检测标注，生成 Lehmer code 排列标注。

用法:
    python tools/generate_annotations.py \
        --data-root /data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco \
        --output-dir /data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from karyoflow.constants import CLASS_TO_SLOTS, ID_TO_CLASS, NUM_SLOTS, SLOT_ORDER
from karyoflow.lehmer import permutation_to_lehmer, validate_permutation


def generate_karyoflow_annotations(coco_json_path: str) -> dict:
    """从 COCO 标注生成 KaryoFlow 排列标注

    Args:
        coco_json_path: COCO 格式标注文件路径

    Returns:
        karyoflow_annotations: dict
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # 按 image_id 分组标注
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    results = {
        "images": coco["images"],
        "categories": coco.get("categories", []),
        "karyoflow_annotations": {},
        "stats": {"total": 0, "valid": 0, "invalid": 0, "skipped": 0},
    }

    for img in coco["images"]:
        img_id = img["id"]
        anns = anns_by_image.get(img_id, [])
        results["stats"]["total"] += 1

        if len(anns) == 0:
            results["stats"]["skipped"] += 1
            continue

        # 按类别分组
        groups = defaultdict(list)
        for ann in anns:
            cat_id = ann["category_id"]
            cls_name = ID_TO_CLASS.get(cat_id)
            if cls_name is None:
                continue
            bbox = ann["bbox"]  # [x, y, w, h] COCO format
            area = bbox[2] * bbox[3]
            groups[cls_name].append({
                "ann": ann,
                "area": area,
                "ann_id": ann["id"],
            })

        # 组内按面积降序 (大的在前 = 标准核型图约定)
        for cls_name in groups:
            groups[cls_name].sort(key=lambda x: -x["area"])

        # 按 SLOT_ORDER 生成排列
        # slot_to_detection[slot_idx] = 原始标注在 anns 列表中的索引
        slot_assignments = []  # (slot_idx, ann_index_in_anns)
        slot_class_counters = defaultdict(int)  # 追踪每个类别已分配几个

        # 构建 ann → anns 列表中索引的映射
        ann_id_to_idx = {ann["id"]: idx for idx, ann in enumerate(anns)}

        for slot_idx, slot_class in enumerate(SLOT_ORDER):
            counter = slot_class_counters[slot_class]
            if slot_class in groups and counter < len(groups[slot_class]):
                entry = groups[slot_class][counter]
                ann_idx = ann_id_to_idx[entry["ann_id"]]
                slot_assignments.append((slot_idx, ann_idx))
                slot_class_counters[slot_class] += 1
            else:
                # 该槽位无对应染色体 (异常核型或标注缺失)
                slot_assignments.append((slot_idx, -1))

        # 检查是否所有 46 个槽位都分配了
        valid = all(idx >= 0 for _, idx in slot_assignments)
        num_assigned = sum(1 for _, idx in slot_assignments if idx >= 0)

        if valid:
            # detection_indices: 每个 slot 对应原始标注列表中的索引
            detection_indices = [idx for _, idx in slot_assignments]

            # 我们需要的排列是: 将 46 个被选中的检测结果重新编号为 0~45，
            # 然后找到这些编号的排列
            # 即: 被选中的检测按原始顺序排序 → 得到 rank 映射
            selected = sorted(set(detection_indices))
            if len(selected) == NUM_SLOTS:
                # 原始 ann index → 0~45 的映射
                idx_to_rank = {idx: rank for rank, idx in enumerate(selected)}
                # 排列: slot i 对应的 rank
                permutation = [idx_to_rank[detection_indices[i]] for i in range(NUM_SLOTS)]

                if validate_permutation(permutation):
                    lehmer_code = permutation_to_lehmer(permutation)
                    results["karyoflow_annotations"][str(img_id)] = {
                        "detection_indices": detection_indices,
                        "permutation": permutation,
                        "lehmer_code": lehmer_code,
                        "num_chromosomes": len(anns),
                        "valid": True,
                    }
                    results["stats"]["valid"] += 1
                else:
                    results["karyoflow_annotations"][str(img_id)] = {
                        "detection_indices": detection_indices,
                        "lehmer_code": None,
                        "num_chromosomes": len(anns),
                        "valid": False,
                        "reason": "invalid_permutation",
                    }
                    results["stats"]["invalid"] += 1
            else:
                results["karyoflow_annotations"][str(img_id)] = {
                    "detection_indices": detection_indices,
                    "lehmer_code": None,
                    "num_chromosomes": len(anns),
                    "valid": False,
                    "reason": f"duplicate_slots_{len(selected)}_unique/{NUM_SLOTS}",
                }
                results["stats"]["invalid"] += 1
        else:
            results["karyoflow_annotations"][str(img_id)] = {
                "detection_indices": [idx for _, idx in slot_assignments],
                "lehmer_code": None,
                "num_chromosomes": len(anns),
                "num_assigned": num_assigned,
                "valid": False,
                "reason": f"incomplete_{num_assigned}/{NUM_SLOTS}",
            }
            results["stats"]["invalid"] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate KaryoFlow annotations from COCO")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data-root)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_root

    for split in args.splits:
        coco_path = os.path.join(args.data_root, split, "_annotations.coco.json")
        if not os.path.exists(coco_path):
            print(f"  Skipping {split}: {coco_path} not found")
            continue

        print(f"Processing {split}...")
        results = generate_karyoflow_annotations(coco_path)

        output_path = os.path.join(output_dir, f"{split}_karyoflow.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        stats = results["stats"]
        print(f"  Total: {stats['total']}, Valid: {stats['valid']}, "
              f"Invalid: {stats['invalid']}, Skipped: {stats['skipped']}")
        print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
