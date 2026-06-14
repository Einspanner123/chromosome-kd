"""将生成结果导出为COCO格式

将generate.py的输出转换为COCO标注格式，可直接用于LDMDet训练数据增强。

用法:
  python projects/ChromoGen/tools/export_coco.py \
    --generation_dir generated/ \
    --output_dir data/Chromosome20240904_NoAug_NoResize_coco/train_augmented/ \
    --score_threshold 0.3
"""

import argparse
import json
import os
import shutil
from typing import Dict, List

CHROMO_CLASSES = [
    'A1',
    'A2',
    'A3',
    'B4',
    'B5',
    'C10',
    'C11',
    'C12',
    'C6',
    'C7',
    'C8',
    'C9',
    'D13',
    'D14',
    'D15',
    'E16',
    'E17',
    'E18',
    'F19',
    'F20',
    'G21',
    'G22',
    'X',
    'Y',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export generated results to COCO format'
    )
    parser.add_argument(
        '--generation_dir', type=str, required=True, help='生成结果目录'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True, help='COCO格式输出目录'
    )
    parser.add_argument(
        '--score_threshold', type=float, default=0.3, help='bbox置信度阈值'
    )
    parser.add_argument(
        '--merge_with',
        type=str,
        default=None,
        help='原始COCO标注文件路径，合并生成数据到原始数据集',
    )
    return parser.parse_args()


def create_coco_categories() -> List[Dict]:
    """创建COCO格式的categories列表"""
    categories = []
    for i, name in enumerate(CHROMO_CLASSES):
        categories.append(
            {
                'id': i + 1,  # COCO category_id从1开始
                'name': name,
                'supercategory': 'chromosome',
            }
        )
    return categories


def export_to_coco(
    generation_dir: str, output_dir: str, score_threshold: float
):
    """将生成结果导出为COCO格式"""
    info_path = os.path.join(generation_dir, 'generation_info.json')
    with open(info_path) as f:
        gen_data = json.load(f)

    # 创建输出目录
    img_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_output_dir, exist_ok=True)

    # COCO结构
    coco = {
        'info': {'description': 'ChromoGen Generated Dataset'},
        'licenses': [{'id': 1, 'name': 'Generated'}],
        'categories': create_coco_categories(),
        'images': [],
        'annotations': [],
    }

    ann_id = 1
    for item in gen_data:
        image_id = item['image_id']

        # 复制图像
        src_path = item['image_path']
        dst_filename = f'gen_{image_id:05d}.png'
        dst_path = os.path.join(img_output_dir, dst_filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

        # 添加图像信息
        from PIL import Image

        try:
            img = Image.open(
                src_path if os.path.exists(src_path) else dst_path
            )
            w, h = img.size
        except Exception:
            w, h = 768, 768

        coco['images'].append(
            {
                'id': image_id,
                'file_name': dst_filename,
                'width': w,
                'height': h,
            }
        )

        # 添加标注
        if 'bboxes' in item and 'labels' in item and 'scores' in item:
            bboxes = item['bboxes']  # cxcywh归一化
            labels = item['labels']
            scores = item['scores']

            for bbox, label, score in zip(bboxes, labels, scores):
                if score < score_threshold:
                    continue

                # cxcywh归一化 → xyxy绝对坐标 → COCO xywh绝对坐标
                cx, cy, bw, bh = bbox
                cx_abs = cx * w
                cy_abs = cy * h
                bw_abs = bw * w
                bh_abs = bh * h
                x = cx_abs - bw_abs / 2
                y = cy_abs - bh_abs / 2

                coco['annotations'].append(
                    {
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': label + 1,  # COCO从1开始
                        'bbox': [
                            round(x, 2),
                            round(y, 2),
                            round(bw_abs, 2),
                            round(bh_abs, 2),
                        ],
                        'area': round(bw_abs * bh_abs, 2),
                        'iscrowd': 0,
                        'score': round(score, 4),
                    }
                )
                ann_id += 1

    # 保存COCO标注
    ann_path = os.path.join(output_dir, '_annotations.coco.json')
    with open(ann_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(
        f'Exported {len(coco["images"])} images, {len(coco["annotations"])} annotations'
    )
    print(f'Annotation file: {ann_path}')
    return coco


def merge_with_original(
    coco_gen: Dict, original_ann_path: str, output_path: str
):
    """将生成的COCO标注合并到原始数据集"""
    with open(original_ann_path) as f:
        coco_orig = json.load(f)

    # 偏移image_id和annotation_id
    max_img_id = max(img['id'] for img in coco_orig['images'])
    max_ann_id = max(ann['id'] for ann in coco_orig['annotations'])

    for img in coco_gen['images']:
        img['id'] += max_img_id
        img['file_name'] = os.path.join('generated', img['file_name'])

    for ann in coco_gen['annotations']:
        ann['id'] += max_ann_id
        ann['image_id'] += max_img_id

    coco_merged = copy.deepcopy(coco_orig)
    coco_merged['images'].extend(coco_gen['images'])
    coco_merged['annotations'].extend(coco_gen['annotations'])

    with open(output_path, 'w') as f:
        json.dump(coco_merged, f, indent=2)

    print(
        f'Merged: {len(coco_orig["images"])} original + {len(coco_gen["images"])} generated = {len(coco_merged["images"])} total'
    )
    print(f'Merged annotation file: {output_path}')


if __name__ == '__main__':
    import copy

    args = parse_args()

    coco = export_to_coco(
        args.generation_dir, args.output_dir, args.score_threshold
    )

    if args.merge_with:
        merge_path = args.merge_with.replace('.json', '_merged.json')
        merge_with_original(coco, args.merge_with, merge_path)
