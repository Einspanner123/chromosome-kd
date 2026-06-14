"""染色体COCO数据集封装

为ChromoGen提供数据加载，输出：
  - pixel_values: [3, H, W] 图像张量
  - class_labels: [24] 类别索引
  - counts: [24] 各类数量
  - bboxes: [N, 4] xyxy格式bbox (可选，enable_bbox_head时)
  - labels: [N] 类别标签 (可选，enable_bbox_head时)
"""

import json
import os
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..constants import NUM_CLASSES


class ChromoGenDataset(Dataset):
    """染色体核型图像数据集

    读取COCO格式标注，输出训练所需的图像和条件信息。
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        img_dir: str = 'train',
        image_size: int = 768,
        enable_bbox: bool = True,
        center_crop: bool = True,
    ):
        """
        Args:
            data_root: 数据集根目录
            ann_file: COCO标注文件路径 (相对于data_root)
            img_dir: 图像目录名 (相对于data_root)
            image_size: 输出图像尺寸 (正方形)
            enable_bbox: 是否加载bbox标注
            center_crop: 是否中心裁剪为正方形
        """
        super().__init__()

        self.data_root = data_root
        self.img_dir = img_dir
        self.image_size = image_size
        self.enable_bbox = enable_bbox
        self.center_crop = center_crop

        # 加载COCO标注
        ann_path = os.path.join(data_root, ann_file)
        with open(ann_path) as f:
            coco_data = json.load(f)

        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = coco_data['annotations']

        # 按image_id分组标注
        self.img_to_anns: Dict[int, List[dict]] = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # 有效的图像ID列表 (有标注的)
        self.img_ids = sorted(self.img_to_anns.keys())

        # 类别名到索引的映射 (COCO category_id从1开始)
        self.cat_id_to_idx = {}
        for i, cat in enumerate(coco_data.get('categories', [])):
            self.cat_id_to_idx[cat['id']] = i

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size)
                if center_crop
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 归一化到[-1, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        anns = self.img_to_anns[img_id]

        # 加载图像
        img_path = os.path.join(
            self.data_root, self.img_dir, img_info['file_name']
        )
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # 预处理图像
        pixel_values = self.transform(image)

        # 统计各类数量
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        bboxes = []
        labels = []

        for ann in anns:
            cat_idx = self.cat_id_to_idx.get(ann['category_id'], -1)
            if cat_idx >= 0:
                counts[cat_idx] += 1

            if self.enable_bbox and 'bbox' in ann:
                # COCO bbox: [x, y, w, h] → xyxy
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])
                labels.append(cat_idx if cat_idx >= 0 else 0)

        # 类别标签: [0, 1, 2, ..., 23]
        class_labels = torch.arange(NUM_CLASSES, dtype=torch.long)

        result = {
            'pixel_values': pixel_values,
            'class_labels': class_labels,
            'counts': counts,
            'img_id': img_id,
            'orig_size': (orig_h, orig_w),
        }

        if self.enable_bbox and len(bboxes) > 0:
            result['bboxes'] = torch.tensor(bboxes, dtype=torch.float32)
            result['labels'] = torch.tensor(labels, dtype=torch.long)
        elif self.enable_bbox:
            result['bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
            result['labels'] = torch.zeros((0,), dtype=torch.long)

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """自定义collate函数，处理变长bbox"""
    result = {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'class_labels': torch.stack([item['class_labels'] for item in batch]),
        'counts': torch.stack([item['counts'] for item in batch]),
        'img_ids': [item['img_id'] for item in batch],
        'orig_sizes': [item['orig_size'] for item in batch],
    }

    if 'bboxes' in batch[0]:
        result['bboxes'] = [item['bboxes'] for item in batch]
        result['labels'] = [item['labels'] for item in batch]

    return result
