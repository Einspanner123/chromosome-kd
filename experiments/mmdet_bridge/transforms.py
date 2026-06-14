"""Custom data augmentation transforms for chromosome detection.

- CLAHE: Contrast Limited Adaptive Histogram Equalization for G-banding enhancement
- SmallObjectCopyPaste: Selective copy-paste that only copies small objects
"""

import random
import warnings
from typing import Optional

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks


@TRANSFORMS.register_module()
class CLAHE(BaseTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE enhances local contrast in images, which is particularly effective
    for G-banding chromosome images where band pattern clarity is crucial
    for classification. Unlike global histogram equalization, CLAHE operates
    on small tiles and limits contrast amplification to avoid noise enhancement.

    Reference: ChroSegNet (Applied Sciences 2023) uses CLAHE + grayscale
    adjustment to enhance chromosome band features and edges.

    Required Keys:
        - img

    Modified Keys:
        - img

    Args:
        prob (float): Probability of applying CLAHE. Defaults to 0.5.
        clip_limit (float): Threshold for contrast limiting. Higher values
            give more contrast. Defaults to 2.0.
        tile_grid_size (tuple): Size of grid for histogram equalization.
            Defaults to (8, 8).
    """

    def __init__(
        self,
        prob: float = 0.5,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ) -> None:
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    @cache_randomness
    def _random_prob(self):
        return np.random.random()

    def transform(self, results: dict) -> dict:
        if self._random_prob() >= self.prob:
            return results

        img = results['img']
        # CLAHE works on grayscale or per-channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

        if img.ndim == 2:
            # Grayscale
            results['img'] = clahe.apply(img)
        elif img.ndim == 3:
            # Convert to LAB, apply CLAHE on L channel, convert back
            # This preserves color while enhancing luminance contrast
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            results['img'] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'prob={self.prob}, clip_limit={self.clip_limit}, '
            f'tile_grid_size={self.tile_grid_size})'
        )


@TRANSFORMS.register_module()
class SmallObjectCopyPaste(BaseTransform):
    """Copy-Paste augmentation that selectively copies small objects.

    Extends the standard CopyPaste by filtering source objects based on
    bbox area, only pasting small objects to increase small object density.
    This is particularly effective for chromosome detection where 22% of
    objects are small and 4.2% are extremely small.

    Reference: "Simple Copy-Paste is a Strong Data Augmentation Method for
    Instance Segmentation" (CVPR 2021) + small object selective strategy
    from "Augmentation for Small Object Detection" (CVPR 2019).

    Required Keys:
        - img
        - gt_bboxes (BaseBoxes[torch.float32])
        - gt_bboxes_labels (np.int64)
        - gt_ignore_flags (bool)
        - gt_masks (BitmapMasks) (optional)
        - mix_results (list): Required by MultiImageMixDataset

    Modified Keys:
        - img
        - gt_bboxes
        - gt_bboxes_labels
        - gt_ignore_flags
        - gt_masks (optional)

    Args:
        max_num_pasted (int): Maximum number of pasted objects. Defaults to 8.
        bbox_occluded_thr (int): Threshold of occluded bbox. Defaults to 10.
        mask_occluded_thr (int): Threshold of occluded mask. Defaults to 300.
        paste_by_box (bool): Whether to use boxes as masks when masks are
            not available. Defaults to True.
        area_thr (float): Maximum bbox area threshold. Only objects with
            area < area_thr will be considered for pasting.
            Defaults to 2500.0 (50x50 pixels, covering ~78% of objects).
        area_ratio_thr (float): If set, use ratio relative to image area
            instead of absolute pixel area. Objects with area <
            area_ratio_thr * image_area will be selected.
            Defaults to None (use absolute area_thr).
    """

    def __init__(
        self,
        max_num_pasted: int = 8,
        bbox_occluded_thr: int = 10,
        mask_occluded_thr: int = 300,
        paste_by_box: bool = True,
        area_thr: float = 2500.0,
        area_ratio_thr: Optional[float] = None,
    ) -> None:
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.paste_by_box = paste_by_box
        self.area_thr = area_thr
        self.area_ratio_thr = area_ratio_thr

    @cache_randomness
    def get_indexes(self, dataset) -> int:
        """Randomly select a source image index."""
        return random.randint(0, len(dataset) - 1)

    def _compute_areas(self, bboxes, img_shape):
        """Compute bbox areas with threshold."""
        if hasattr(bboxes, 'tensor'):
            bbox_tensor = bboxes.tensor
        else:
            bbox_tensor = bboxes

        widths = bbox_tensor[:, 2] - bbox_tensor[:, 0]
        heights = bbox_tensor[:, 3] - bbox_tensor[:, 1]
        areas = widths * heights

        if self.area_ratio_thr is not None:
            img_area = img_shape[0] * img_shape[1]
            thr = self.area_ratio_thr * img_area
        else:
            thr = self.area_thr

        # Convert to numpy if tensor
        if hasattr(areas, 'numpy'):
            areas = areas.numpy()

        return areas, thr

    @cache_randomness
    def _get_selected_inds(self, num_small: int) -> np.ndarray:
        """Randomly select indices from small objects."""
        max_num = min(num_small, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num + 1)
        if num_pasted == 0:
            return np.array([], dtype=np.int64)
        return np.random.choice(num_small, size=num_pasted, replace=False)

    def get_gt_masks(self, results: dict) -> BitmapMasks:
        """Get gt_masks or generate from bboxes."""
        if results.get('gt_masks') is not None:
            if self.paste_by_box:
                warnings.warn(
                    'gt_masks is already contained in results, '
                    'so paste_by_box is disabled.'
                )
            return results['gt_masks']
        else:
            if not self.paste_by_box:
                raise RuntimeError('results does not contain masks.')
            return results['gt_bboxes'].create_masks(results['img'].shape[:2])

    @staticmethod
    def _filter_by_area(results: dict, small_inds: np.ndarray) -> dict:
        """Filter results to only contain selected small objects."""
        filtered = dict(results)
        filtered['gt_bboxes'] = results['gt_bboxes'][small_inds]
        filtered['gt_bboxes_labels'] = results['gt_bboxes_labels'][small_inds]
        filtered['gt_masks'] = results['gt_masks'][small_inds]
        filtered['gt_ignore_flags'] = results['gt_ignore_flags'][small_inds]
        return filtered

    def transform(self, results: dict) -> dict:
        """Apply small object copy-paste."""
        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, (
            f'SmallObjectCopyPaste only supports 1 mix image, got {num_images}'
        )

        src_results = results['mix_results'][0]

        # Ensure masks exist (generate from bboxes if paste_by_box)
        src_results['gt_masks'] = self.get_gt_masks(src_results)
        src_bboxes = src_results['gt_bboxes']

        if len(src_bboxes) == 0:
            return results

        # Filter small objects from source
        areas, thr = self._compute_areas(src_bboxes, src_results['img'].shape)
        small_inds = np.where(areas < thr)[0]

        if len(small_inds) == 0:
            return results

        # Select from small objects
        selected_small_inds = small_inds[
            self._get_selected_inds(len(small_inds))
        ]
        selected_src = self._filter_by_area(src_results, selected_small_inds)

        return self._copy_paste(results, selected_src)

    def _copy_paste(self, dst_results: dict, src_results: dict) -> dict:
        """Copy-paste source objects onto destination image."""
        dst_img = dst_results['img']
        dst_h, dst_w = dst_img.shape[:2]
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_ignore_flags = dst_results['gt_ignore_flags']

        # Ensure dst masks exist
        dst_masks = self.get_gt_masks(dst_results)
        dst_results['gt_masks'] = dst_masks

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = src_results['gt_masks']
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # Resize src to match dst if shapes differ (after different augmentations)
        src_h, src_w = src_img.shape[:2]
        if src_h != dst_h or src_w != dst_w:
            scale_y = dst_h / src_h
            scale_x = dst_w / src_w
            src_img = cv2.resize(src_img, (dst_w, dst_h))

            # Rescale bboxes
            if hasattr(src_bboxes, 'tensor'):
                bbox_tensor = src_bboxes.tensor.clone()
                bbox_tensor[:, 0] *= scale_x
                bbox_tensor[:, 1] *= scale_y
                bbox_tensor[:, 2] *= scale_x
                bbox_tensor[:, 3] *= scale_y
                src_bboxes = type(src_bboxes)(bbox_tensor)

            # Rescale masks
            resized_masks = []
            for mask in src_masks.masks:
                resized = cv2.resize(
                    mask.astype(np.float32),
                    (dst_w, dst_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                resized_masks.append((resized > 0.5).astype(np.uint8))
            src_masks = BitmapMasks(np.array(resized_masks), dst_h, dst_w)

        # Update masks: zero out dst pixels covered by src objects
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = dst_masks
        updated_dst_masks.masks = np.where(
            composed_mask, 0, updated_dst_masks.masks
        )
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # Filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (
            (l1_distance <= self.bbox_occluded_thr).all(dim=-1).numpy()
        )
        masks_inds = (
            updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        )
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects onto destination
        img = (
            dst_img * (1 - composed_mask[..., np.newaxis])
            + src_img * composed_mask[..., np.newaxis]
        ).astype(np.uint8)
        bboxes = src_bboxes.cat([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks]
        )
        ignore_flags = np.concatenate(
            [dst_ignore_flags[valid_inds], src_ignore_flags]
        )

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(
            masks, masks.shape[1], masks.shape[2]
        )
        dst_results['gt_ignore_flags'] = ignore_flags

        return dst_results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'paste_by_box={self.paste_by_box}, '
        if self.area_ratio_thr is not None:
            repr_str += f'area_ratio_thr={self.area_ratio_thr})'
        else:
            repr_str += f'area_thr={self.area_thr})'
        return repr_str
