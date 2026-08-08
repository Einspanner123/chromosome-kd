import importlib.util
from pathlib import Path

import numpy as np


_PATH = (Path(__file__).resolve().parents[1] /
         'experiments/analysis/precision_bottleneck_diagnosis.py')
_SPEC = importlib.util.spec_from_file_location('precision_diag', _PATH)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


def test_pairwise_iou_known_values():
    boxes = np.asarray([[0, 0, 10, 10], [5, 5, 15, 15]])
    values = diag.pairwise_iou(boxes[:1], boxes)
    assert np.isclose(values[0, 0], 1.0)
    assert np.isclose(values[0, 1], 25 / 175)


def test_oracles_change_only_supported_components():
    gt = {1: [{
        'image_id': 1, 'category_id': 2, 'bbox': [0, 0, 10, 10],
        'xyxy': np.asarray([0, 0, 10, 10]), 'area': 100,
    }]}
    pred = [{
        'image_id': 1, 'category_id': 1, 'bbox': [1, 1, 8, 8],
        'score': 0.9,
    }]
    variants = diag.build_oracles(pred, gt)
    assert variants['classification'][0]['category_id'] == 2
    assert variants['classification'][0]['bbox'] == pred[0]['bbox']
    assert variants['localization'][0] == pred[0]
    assert variants['joint'][0]['category_id'] == 2
    assert variants['joint'][0]['bbox'] == gt[1][0]['bbox']


def test_quality_oracle_only_changes_scores():
    gt = {1: [{
        'category_id': 2, 'bbox': [0, 0, 10, 10],
        'xyxy': diag.xywh_to_xyxy([0, 0, 10, 10]),
    }]}
    pred = [{
        'image_id': 1, 'category_id': 2,
        'bbox': [0, 0, 5, 10], 'score': 0.8,
    }]
    variants = diag.build_quality_oracles(pred, gt, betas=(1.0,))
    result = variants['quality_beta_1'][0]
    assert result['bbox'] == pred[0]['bbox']
    assert result['category_id'] == pred[0]['category_id']
    assert abs(result['score'] - 0.4) < 1e-8
    assert abs(variants['quality_only'][0]['score'] - 0.5) < 1e-8


def test_overlap_and_area_strata_are_deterministic():
    grouped = {1: [
        {'xyxy': np.asarray([0, 0, 10, 10]), 'area': 100},
        {'xyxy': np.asarray([9, 0, 19, 10]), 'area': 100},
        {'xyxy': np.asarray([100, 100, 200, 200]), 'area': 10000},
    ]}
    diag.annotate_gt_strata(grouped)
    assert grouped[1][0]['overlap_stratum'] == 'near_0.01-0.20'
    assert grouped[1][2]['overlap_stratum'] == 'isolated_<0.01'
    assert grouped[1][0]['area_stratum'] == 'small'
    assert grouped[1][2]['area_stratum'] == 'large'
