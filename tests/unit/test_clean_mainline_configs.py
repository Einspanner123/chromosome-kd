from pathlib import Path

from mmengine.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path: str) -> Config:
    return Config.fromfile(str(PROJECT_ROOT / relative_path))


def test_clean_a4_uses_only_random_coupling():
    cfg = _load('experiments/configs/ldmdet/a4_dpm_pp_random_chr2024.py')
    coupling = dict(cfg.model.bbox_head.coupling)

    assert coupling == {'type': 'random'}
    assert cfg.model.bbox_head.solver_type == 'dpm_solver_pp'
    assert cfg.model.bbox_head.sampling_timesteps == 4
    assert cfg.val_evaluator.ann_file.endswith(
        'valid/_annotations.coco.json')
    assert cfg.test_evaluator.ann_file.endswith(
        'test/_annotations.coco.json')


def test_ocgr_clean_config_keeps_random_coupling():
    cfg = _load(
        'experiments/configs/ldmdet/directions/georel/'
        'ocgr_random_chr2024.py')

    assert dict(cfg.model.bbox_head.coupling) == {'type': 'random'}
    assert cfg.model.bbox_head.geometric_relation_start_head == 3
    assert cfg.model.bbox_head.single_head.geometric_relation_attention is True
