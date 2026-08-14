from pathlib import Path

from tools.experiments.matrix import resolve_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
D1_MATRIX = PROJECT_ROOT / "experiments/configs/matrices/d1_inhouse1700.yaml"
D1_DISTILL_MATRIX = (
    PROJECT_ROOT / "experiments/configs/matrices/d1_inhouse1700_head_distill.yaml"
)


def test_karyoflow_mainline_identity():
    cfg, _ = resolve_config(D1_MATRIX, "karyoflow", 42)
    head = cfg.model.bbox_head

    assert dict(head.coupling) == {"type": "random"}
    assert head.diffusion_type == "rectified_flow"
    assert head.solver_type == "dpm_solver_pp"
    assert head.sampling_timesteps == 4
    assert cfg.experiment.replication_unit == "independent_training_seed"
    assert cfg.test_evaluator.ann_file.endswith("test/_annotations.coco.json")


def test_lqcr_is_parent_matched_and_final_only():
    cfg, _ = resolve_config(D1_MATRIX, "karyoflow_lqcr", 42)
    head = cfg.model.bbox_head

    assert cfg.experiment.parent_method_id == "karyoflow_r50"
    assert cfg.experiment.pairing == "same_training_seed"
    assert head.quality_only_training is True
    assert head.quality_calibration_mode == "final_only"


def test_head_distillation_restored_identity():
    cfg, _ = resolve_config(D1_DISTILL_MATRIX, "karyoflow_h3_distill", 42)
    head = cfg.model.bbox_head

    assert cfg.experiment.parent_checkpoint_binding == "teacher_checkpoint"
    assert cfg.experiment.pairing == "same_training_seed"
    assert head.use_distillation is True
    assert head.num_heads == 3
    assert dict(head.distill_head_map) == {0: 0, 1: 2, 2: 5}
