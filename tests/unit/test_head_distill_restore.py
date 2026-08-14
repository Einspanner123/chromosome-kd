"""Regression tests for the restored parent-matched head distillation."""

import torch

from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.single_head import SingleDiffusionDetHead
from tools.experiments.matrix import resolve_config, scientific_hash


def make_head(num_heads, use_distillation=False):
    single = SingleDiffusionDetHead(
        num_classes=3,
        feat_channels=8,
        dim_feedforward=16,
        num_cls_convs=1,
        num_reg_convs=1,
        num_heads=2,
        time_conditioning='scale_shift',
        use_sdpa=False,
    )
    return DiffusionDetHead(
        num_classes=3,
        feat_channels=8,
        num_proposals=2,
        num_heads=num_heads,
        sampling_timesteps=1,
        solver_type='euler',
        diffusion_type='rectified_flow',
        single_head=single,
        roi_extractor=None,
        criterion=None,
        use_distillation=use_distillation,
        distill_head_map={0: 0, 1: 2, 2: 5},
    )


def test_teacher_is_frozen_unregistered_and_head_mapping_is_exact():
    teacher = make_head(6)
    student = make_head(3, use_distillation=True)
    with torch.no_grad():
        for index, head in enumerate(teacher.head_series):
            for parameter in head.parameters():
                parameter.fill_(index + 1.0)
    student.set_teacher(teacher)
    mapped = student.init_student_from_teacher()

    assert mapped
    assert not any(
        name.startswith('_teacher') for name in student.state_dict()
    )
    assert not any(
        name.startswith('_teacher') for name, _ in student.named_parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in teacher.parameters()
    )
    for student_index, teacher_index in {0: 0, 1: 2, 2: 5}.items():
        student_state = student.head_series[student_index].state_dict()
        teacher_state = teacher.head_series[teacher_index].state_dict()
        assert student_state.keys() == teacher_state.keys()
        for name in student_state:
            assert torch.equal(student_state[name], teacher_state[name])


def test_teacher_follows_student_dtype_conversion():
    teacher = make_head(6)
    student = make_head(3, use_distillation=True)
    student.set_teacher(teacher)
    student.double()
    assert next(student.parameters()).dtype == torch.float64
    assert next(teacher.parameters()).dtype == torch.float64


def test_parent_checkpoint_uses_teacher_binding_without_changing_science_hash():
    matrix = 'experiments/configs/matrices/d2_taichung_head_distill.yaml'
    base, _ = resolve_config(matrix, 'karyoflow_ot_h3_distill', 335778785)
    bound, _ = resolve_config(
        matrix,
        'karyoflow_ot_h3_distill',
        335778785,
        '/tmp/parent-checkpoint.pth',
    )
    assert bound.get('load_from') is None
    assert bound.model.teacher_checkpoint == '/tmp/parent-checkpoint.pth'
    assert scientific_hash(bound) == scientific_hash(base)
