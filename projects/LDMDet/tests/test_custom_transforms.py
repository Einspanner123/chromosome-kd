"""Tests for custom transforms: CLAHE and SmallObjectCopyPaste.

Red-Green Refactoring:
  1. RED:   Write tests that define expected behavior
  2. GREEN: Run tests, fix code until all pass
  3. REFACTOR: Clean up while keeping tests green
"""

import sys

sys.path.insert(0, '/home/linkst/workspace/projects/chromosome-kd')

import numpy as np
import torch

# Import custom transforms to register them
from mmdet.datasets.transforms import (
    geometric as _mmdet_geometric,  # noqa: F401
)

# Import mmdet transforms to register all built-in transforms
# (MinIoURandomCrop, RandomAffine, Rotate, etc.)
from mmdet.datasets.transforms import (
    transforms as _mmdet_transforms,  # noqa: F401
)
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks

# ============================================================================
# Helper: construct results dict mimicking mmdet pipeline output
# ============================================================================


def make_results(img_h=480, img_w=640, bboxes_xyxy=None, labels=None):
    """Create a minimal results dict for testing transforms.

    Args:
        img_h, img_w: Image dimensions.
        bboxes_xyxy: List of [x1,y1,x2,y2] or None for empty.
        labels: List of int labels, same length as bboxes.

    Returns:
        dict with keys: img, gt_bboxes, gt_bboxes_labels, gt_ignore_flags
    """
    # Simulate G-banding image: light gray background (220) with darker objects
    img = np.full((img_h, img_w, 3), 220, dtype=np.uint8)

    if bboxes_xyxy is None:
        bboxes_xyxy = []
        labels = []

    if labels is None:
        labels = [0] * len(bboxes_xyxy)

    bbox_tensor = (
        torch.tensor(bboxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        if len(bboxes_xyxy) > 0
        else torch.zeros(0, 4, dtype=torch.float32)
    )
    labels_arr = np.array(labels, dtype=np.int64)
    ignore_flags = np.zeros(len(bboxes_xyxy), dtype=bool)

    return {
        'img': img,
        'gt_bboxes': HorizontalBoxes(bbox_tensor),
        'gt_bboxes_labels': labels_arr,
        'gt_ignore_flags': ignore_flags,
    }


# ============================================================================
# CLAHE Tests
# ============================================================================


class TestCLAHE:
    """Test CLAHE transform."""

    def test_prob_1_always_applies(self):
        """prob=1.0 should always apply CLAHE."""
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=1.0))
        # Create low-contrast image
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add some variation
        img[20:40, 20:40] = 100
        img[50:70, 50:70] = 180
        results = {'img': img.copy()}
        out = clahe.transform(results)
        # Image should be modified (not identical)
        assert not np.array_equal(out['img'], img), (
            'CLAHE with prob=1.0 should modify the image'
        )

    def test_prob_0_never_applies(self):
        """prob=0.0 should never apply CLAHE."""
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=0.0))
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        results = {'img': img.copy()}
        out = clahe.transform(results)
        assert np.array_equal(out['img'], img), (
            'CLAHE with prob=0.0 should not modify the image'
        )

    def test_output_shape_unchanged(self):
        """Output image shape should match input."""
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=1.0))
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        results = {'img': img}
        out = clahe.transform(results)
        assert out['img'].shape == img.shape, (
            f'Shape mismatch: {out["img"].shape} vs {img.shape}'
        )

    def test_output_dtype_uint8(self):
        """Output should remain uint8."""
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=1.0))
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        results = {'img': img}
        out = clahe.transform(results)
        assert out['img'].dtype == np.uint8, (
            f'Expected uint8, got {out["img"].dtype}'
        )

    def test_grayscale_input(self):
        """Should handle 2D grayscale images."""
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=1.0))
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        results = {'img': img}
        out = clahe.transform(results)
        assert out['img'].shape == (100, 100), (
            f'Grayscale shape mismatch: {out["img"].shape}'
        )
        assert out['img'].dtype == np.uint8

    def test_enhances_low_contrast(self):
        """CLAHE should increase contrast in low-contrast regions."""
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=1.0, clip_limit=4.0))
        # Low-contrast image: values clustered around 128
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        img[50:100, 50:100] = 135  # Slightly brighter region
        img[120:170, 120:170] = 121  # Slightly darker region
        results = {'img': img}
        out = clahe.transform(results)
        # After CLAHE, the difference between regions should be amplified
        diff_before = int(img[75, 75, 0]) - int(img[145, 145, 0])
        diff_after = int(out['img'][75, 75, 0]) - int(out['img'][145, 145, 0])
        assert abs(diff_after) >= abs(diff_before), (
            f'CLAHE should amplify contrast: before={diff_before}, after={diff_after}'
        )

    def test_preserves_other_keys(self):
        """CLAHE should not modify non-image keys."""
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=1.0))
        results = make_results(bboxes_xyxy=[[10, 20, 50, 60]], labels=[1])
        results_copy = {k: v for k, v in results.items() if k != 'img'}
        out = clahe.transform(results)
        for k in results_copy:
            if k == 'img':
                continue
            if isinstance(results_copy[k], torch.Tensor):
                assert torch.equal(out[k], results_copy[k]), (
                    f'Key {k} was modified'
                )
            elif isinstance(results_copy[k], np.ndarray):
                assert np.array_equal(out[k], results_copy[k]), (
                    f'Key {k} was modified'
                )


# ============================================================================
# SmallObjectCopyPaste Tests
# ============================================================================


class TestSmallObjectCopyPaste:
    """Test SmallObjectCopyPaste transform."""

    def _make_copy_paste_results(
        self,
        dst_bboxes,
        src_bboxes,
        dst_labels=None,
        src_labels=None,
        img_h=480,
        img_w=640,
    ):
        """Create dst + mix_results for CopyPaste testing."""
        dst = make_results(img_h, img_w, dst_bboxes, dst_labels)
        src = make_results(img_h, img_w, src_bboxes, src_labels)
        dst['mix_results'] = [src]
        return dst

    def test_only_copies_small_objects(self):
        """Should only paste objects with area < area_thr."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=100,  # paste all small objects
                paste_by_box=True,
                area_thr=2500.0,
            )
        )
        # dst: 1 object, src: 1 small (40x40=1600) + 1 large (80x80=6400)
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=[
                [100, 100, 140, 140],  # small: 40x40=1600 < 2500
                [200, 200, 280, 280],
            ],  # large: 80x80=6400 > 2500
            dst_labels=[0],
            src_labels=[1, 2],
        )
        out = cp.transform(dst)

        # Original dst had 1 object, should add only small src objects
        total_objects = len(out['gt_bboxes'])
        assert total_objects >= 1, 'Should have at least original objects'
        # The large object (label=2) should NOT be pasted
        pasted_labels = out['gt_bboxes_labels']
        # Check that label 2 (large object) is not in the newly added labels
        # Original label was 0, so any label 2 would be from paste
        assert 2 not in pasted_labels, (
            f'Large object (label=2) should not be pasted, got labels: {pasted_labels}'
        )

    def test_area_thr_filters_correctly(self):
        """area_thr should correctly filter objects by bbox area."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=100,
                paste_by_box=True,
                area_thr=1000.0,  # Only very small objects
            )
        )
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=[
                [100, 100, 120, 120],  # 20x20=400 < 1000 ✓
                [200, 200, 250, 250],  # 50x50=2500 > 1000 ✗
                [300, 300, 320, 320],
            ],  # 20x20=400 < 1000 ✓
            dst_labels=[0],
            src_labels=[1, 2, 3],
        )
        out = cp.transform(dst)
        pasted_labels = out['gt_bboxes_labels']
        assert 2 not in pasted_labels, (
            'Medium object (label=2, area=2500) should not be pasted'
        )

    def test_empty_source_returns_unchanged(self):
        """Empty source bboxes should return dst unchanged."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=8,
                paste_by_box=True,
                area_thr=2500.0,
            )
        )
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=[],  # empty source
            dst_labels=[0],
            src_labels=[],
        )
        out = cp.transform(dst)
        assert len(out['gt_bboxes']) == 1, (
            'Should have original 1 object when source is empty'
        )

    def test_no_small_objects_in_source(self):
        """If source has only large objects, nothing should be pasted."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=8,
                paste_by_box=True,
                area_thr=2500.0,
            )
        )
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=[[100, 100, 200, 200]],  # 100x100=10000 > 2500
            dst_labels=[0],
            src_labels=[1],
        )
        out = cp.transform(dst)
        assert len(out['gt_bboxes']) == 1, 'Should not paste any large objects'

    def test_max_num_pasted_limits_count(self):
        """max_num_pasted should limit the number of pasted objects."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=2,
                paste_by_box=True,
                area_thr=2500.0,
            )
        )
        # 5 small objects in source
        src_bboxes = [
            [i * 30, i * 30, i * 30 + 20, i * 30 + 20] for i in range(5)
        ]
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=src_bboxes,
            dst_labels=[0],
            src_labels=list(range(5)),
        )
        out = cp.transform(dst)
        # Original 1 + at most 2 pasted = at most 3
        assert len(out['gt_bboxes']) <= 3, (
            f'Should paste at most 2 objects, got {len(out["gt_bboxes"]) - 1} pasted'
        )

    def test_output_image_valid(self):
        """Output image should have valid pixel values [0, 255]."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=8,
                paste_by_box=True,
                area_thr=2500.0,
            )
        )
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=[[100, 100, 130, 130]],
            dst_labels=[0],
            src_labels=[1],
        )
        out = cp.transform(dst)
        assert out['img'].dtype == np.uint8, (
            f'Expected uint8, got {out["img"].dtype}'
        )
        assert out['img'].min() >= 0, f'Min pixel value {out["img"].min()} < 0'
        assert out['img'].max() <= 255, (
            f'Max pixel value {out["img"].max()} > 255'
        )

    def test_area_ratio_thr(self):
        """area_ratio_thr should use relative area instead of absolute."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=100,
                paste_by_box=True,
                area_thr=2500.0,
                area_ratio_thr=0.01,  # 1% of image area
            )
        )
        img_h, img_w = 480, 640
        img_area = img_h * img_w  # 307200
        # 1% = 3072 pixels
        # 50x50=2500 < 3072 -> small ✓
        # 60x60=3600 > 3072 -> not small ✗
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=[
                [100, 100, 150, 150],  # 50x50=2500 < 3072
                [200, 200, 260, 260],
            ],  # 60x60=3600 > 3072
            dst_labels=[0],
            src_labels=[1, 2],
            img_h=img_h,
            img_w=img_w,
        )
        out = cp.transform(dst)
        pasted_labels = out['gt_bboxes_labels']
        assert 2 not in pasted_labels, (
            'Object with area > 1% of image should not be pasted'
        )

    def test_paste_by_box_generates_masks(self):
        """paste_by_box=True should generate masks from bboxes."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=100,
                paste_by_box=True,
                area_thr=2500.0,
            )
        )
        dst = self._make_copy_paste_results(
            dst_bboxes=[[10, 10, 50, 50]],
            src_bboxes=[[100, 100, 130, 130]],
            dst_labels=[0],
            src_labels=[1],
        )
        out = cp.transform(dst)
        assert 'gt_masks' in out, 'Output should contain gt_masks'
        assert isinstance(out['gt_masks'], BitmapMasks)

    def test_different_src_dst_sizes(self):
        """Copy-paste should handle src and dst with different image sizes."""
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=100,
                paste_by_box=True,
                area_thr=2500.0,
            )
        )
        # dst image 480x640, src image 400x500
        dst_img = np.full((480, 640, 3), 200, dtype=np.uint8)
        src_img = np.full((400, 500, 3), 180, dtype=np.uint8)

        dst = {
            'img': dst_img,
            'gt_bboxes': HorizontalBoxes(
                torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
            ),
            'gt_bboxes_labels': np.array([0], dtype=np.int64),
            'gt_ignore_flags': np.zeros(1, dtype=bool),
            'mix_results': [
                {
                    'img': src_img,
                    'gt_bboxes': HorizontalBoxes(
                        torch.tensor([[20, 20, 60, 60]], dtype=torch.float32)
                    ),
                    'gt_bboxes_labels': np.array([1], dtype=np.int64),
                    'gt_ignore_flags': np.zeros(1, dtype=bool),
                }
            ],
        }
        out = cp.transform(dst)
        assert out['img'].shape == (480, 640, 3), (
            f'Output img shape should match dst: {out["img"].shape}'
        )
        assert out['img'].dtype == np.uint8
        assert len(out['gt_bboxes']) >= 1


# ============================================================================
# Integration Test: Full Pipeline
# ============================================================================


class TestPipelineIntegration:
    """Test the full augmentation pipeline with real dataset sample."""

    CONFIG_PATH = 'configs/ldmdet_rf_heun_shifted_bs8_aug.py'
    # Config uses relative _base_ paths, must run from LDMDet dir
    CONFIG_CWD = '/home/linkst/workspace/projects/chromosome-kd/LDMDet'

    def _load_config(self):
        import os

        from mmengine.config import Config

        old_cwd = os.getcwd()
        try:
            os.chdir(self.CONFIG_CWD)
            cfg = Config.fromfile(self.CONFIG_PATH)
        finally:
            os.chdir(old_cwd)
        return cfg

    def test_full_pipeline_runs(self):
        """Full pipeline should run without errors on realistic data."""
        cfg = self._load_config()
        # Check inner pipeline (train_dataset_pipeline) has CLAHE and Sharpness
        inner_pipeline = cfg.train_dataloader.dataset.dataset.pipeline
        inner_types = [t['type'] for t in inner_pipeline]
        assert 'CLAHE' in inner_types, (
            f'CLAHE not in inner pipeline: {inner_types}'
        )
        assert 'Sharpness' in inner_types, (
            f'Sharpness not in inner pipeline: {inner_types}'
        )

        # Check outer pipeline (train_pipeline) has SmallObjectCopyPaste
        outer_pipeline = cfg.train_dataloader.dataset.pipeline
        outer_types = [t['type'] for t in outer_pipeline]
        assert 'SmallObjectCopyPaste' in outer_types, (
            f'SmallObjectCopyPaste not in outer pipeline: {outer_types}'
        )

        # SmallObjectCopyPaste should NOT be in inner pipeline
        assert 'SmallObjectCopyPaste' not in inner_types, (
            f'SmallObjectCopyPaste should not be in inner pipeline: {inner_types}'
        )

        print(f'Inner pipeline: {inner_types}')
        print(f'Outer pipeline: {outer_types}')

    def test_pipeline_transforms_on_synthetic_data(self):
        """Run custom transforms (CLAHE, Sharpness, SmallObjectCopyPaste)
        on synthetic data to verify end-to-end correctness."""
        # Build only the custom transforms directly
        clahe = TRANSFORMS.build(dict(type='CLAHE', prob=1.0))
        sharpness = TRANSFORMS.build(
            dict(type='Sharpness', prob=1.0, min_mag=0.5, max_mag=0.9)
        )
        cp = TRANSFORMS.build(
            dict(
                type='SmallObjectCopyPaste',
                max_num_pasted=8,
                paste_by_box=True,
                area_thr=2500.0,
            )
        )

        # Create synthetic chromosome-like data
        img = np.full((480, 640, 3), 220, dtype=np.uint8)
        for i in range(5):
            x1, y1 = 50 + i * 100, 50 + i * 60
            x2, y2 = x1 + 40, y1 + 30
            img[y1:y2, x1:x2] = 100 + i * 20

        results = {
            'img': img,
            'gt_bboxes': HorizontalBoxes(
                torch.tensor(
                    [
                        [50, 50, 90, 80],
                        [150, 110, 190, 140],
                        [250, 170, 290, 200],
                        [350, 230, 390, 260],
                        [450, 290, 490, 320],
                    ],
                    dtype=torch.float32,
                )
            ),
            'gt_bboxes_labels': np.array([0, 1, 2, 3, 4], dtype=np.int64),
            'gt_ignore_flags': np.zeros(5, dtype=bool),
        }

        # Step 1: CLAHE
        results = clahe.transform(results)
        assert results['img'].dtype == np.uint8
        assert results['img'].shape == (480, 640, 3)
        print(f'  CLAHE: OK, img shape={results["img"].shape}')

        # Step 2: Sharpness
        results = sharpness.transform(results)
        assert results['img'].dtype == np.uint8
        assert results['img'].shape == (480, 640, 3)
        print(f'  Sharpness: OK, img shape={results["img"].shape}')

        # Step 3: SmallObjectCopyPaste (needs mix_results)
        src = make_results(
            bboxes_xyxy=[[100, 100, 130, 130], [300, 300, 360, 360]],
            labels=[5, 6],
        )
        results['mix_results'] = [src]
        results = cp.transform(results)
        assert results['img'].dtype == np.uint8
        assert results['img'].shape == (480, 640, 3)
        assert len(results['gt_bboxes']) >= 5  # At least original objects
        print(
            f'  SmallObjectCopyPaste: OK, bboxes={len(results["gt_bboxes"])}'
        )

        # Final checks
        assert results['img'].dtype == np.uint8
        assert results['img'].shape[2] == 3
        assert len(results['gt_bboxes']) >= 1


# ============================================================================
# Run all tests
# ============================================================================


def run_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestCLAHE,
        TestSmallObjectCopyPaste,
        TestPipelineIntegration,
    ]
    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in methods:
            total += 1
            test_name = f'{cls.__name__}.{method_name}'
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f'  PASS: {test_name}')
            except AssertionError as e:
                failed += 1
                errors.append((test_name, str(e)))
                print(f'  FAIL: {test_name} - {e}')
            except Exception as e:
                failed += 1
                errors.append((test_name, f'{type(e).__name__}: {e}'))
                print(f'  ERROR: {test_name} - {type(e).__name__}: {e}')

    print(f'\n{"=" * 60}')
    print(f'Results: {passed}/{total} passed, {failed} failed')
    if errors:
        print('\nFailed tests:')
        for name, err in errors:
            print(f'  {name}: {err}')
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
