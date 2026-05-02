"""
KaryoFlow 数据集

读取 COCO 图像 + KaryoFlow 排列标注，提供训练/评估所需的数据。
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import HOMOLOG_PAIRS, NUM_SLOTS


class KaryoFlowDataset(Dataset):
    """KaryoFlow 训练/评估数据集

    Args:
        coco_json: COCO 标注文件路径
        karyoflow_json: KaryoFlow 排列标注文件路径
        image_dir: 图像目录
        crop_size: 染色体 crop 尺寸
        augment: 是否启用数据增强
        only_valid: 只使用有效 (N=46) 的样本
    """

    def __init__(
        self,
        coco_json: str,
        karyoflow_json: str,
        image_dir: str,
        crop_size: int = 64,
        augment: bool = True,
        only_valid: bool = True,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.augment = augment

        # 加载 COCO 标注
        with open(coco_json, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # 加载 KaryoFlow 排列标注
        with open(karyoflow_json, "r") as f:
            kf_data = json.load(f)

        self.kf_annotations = kf_data["karyoflow_annotations"]

        # 过滤有效样本
        self.valid_image_ids = []
        for img_id_str, ann in self.kf_annotations.items():
            if only_valid and not ann.get("valid", False):
                continue
            self.valid_image_ids.append(int(img_id_str))

        # 图像预处理
        self.crop_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.valid_image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.valid_image_ids[idx]
        img_info = self.images[img_id]
        anns = self.anns_by_image[img_id]
        kf_ann = self.kf_annotations[str(img_id)]

        # 加载图像
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # 获取排列标注
        detection_indices = kf_ann["detection_indices"]  # slot → 原始标注 idx (用于 crop)
        permutation = kf_ann["permutation"]              # slot → 0~45 rank (用于训练)

        # 数据增强: 同源对交换
        if self.augment:
            detection_indices, permutation = self._augment_homolog_swap(
                detection_indices, permutation
            )

        # 裁剪每个检测到的染色体
        crops = []
        bboxes = []
        labels = []

        for slot_idx in range(NUM_SLOTS):
            ann_idx = detection_indices[slot_idx]
            ann = anns[ann_idx]

            # COCO bbox [x, y, w, h] → PIL crop [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            # 边界保护
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))

            crop = image.crop((x1, y1, x2, y2))

            # 数据增强: crop jitter
            if self.augment:
                crop = self._augment_crop_jitter(crop, image, x1, y1, x2, y2, img_w, img_h)

            crop_tensor = self.crop_transform(crop)
            crops.append(crop_tensor)
            bboxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

        return {
            "image_id": img_id,
            "crops": torch.stack(crops),              # (N, 3, crop_size, crop_size)
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),  # (N, 4) xyxy
            "labels": torch.tensor(labels, dtype=torch.long),     # (N,) category_id
            "permutation": torch.tensor(permutation, dtype=torch.long),  # (N,) 0~45 排列
        }

    def _augment_homolog_swap(
        self, detection_indices: List[int], permutation: List[int]
    ) -> Tuple[List[int], List[int]]:
        """同源对随机交换 (排列空间数据增强)

        对每对同源染色体，以 50% 概率交换对内两条的位置。
        同时更新 detection_indices 和 permutation。
        """
        det_indices = list(detection_indices)
        perm = list(permutation)

        for slot_a, slot_b in HOMOLOG_PAIRS:
            if random.random() < 0.5:
                det_indices[slot_a], det_indices[slot_b] = (
                    det_indices[slot_b],
                    det_indices[slot_a],
                )
                perm[slot_a], perm[slot_b] = perm[slot_b], perm[slot_a]

        return det_indices, perm

    def _augment_crop_jitter(
        self, crop, image, x1, y1, x2, y2, img_w, img_h
    ):
        """裁剪框轻微抖动 (模拟检测器不完美)"""
        w = x2 - x1
        h = y2 - y1
        jitter = 0.05  # 5% 抖动

        dx = int(w * jitter * (random.random() * 2 - 1))
        dy = int(h * jitter * (random.random() * 2 - 1))

        nx1 = max(0, x1 + dx)
        ny1 = max(0, y1 + dy)
        nx2 = min(img_w, x2 + dx)
        ny2 = min(img_h, y2 + dy)

        if nx2 > nx1 and ny2 > ny1:
            return image.crop((nx1, ny1, nx2, ny2))
        return crop
