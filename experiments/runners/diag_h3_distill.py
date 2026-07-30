"""诊断 H3 Distill: 直接测试 init_student_from_teacher + init_weights 覆盖

绕过完整 model 构建 (避免 DetDataPreprocessor 注册问题),
直接构建 Student/Teacher head, 手动加载 +DPM-Solver++ checkpoint, 验证:
1. init_student_from_teacher 是否把 Teacher head 权重复制到 Student
2. mmengine init_weights 是否会覆盖 (Student head 是 nn.Module, 应该不会)
"""

import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--teacher-ckpt',
        default='work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
    )
    args = parser.parse_args()

    print('=' * 70)
    print('H3 Distill 诊断: init_student_from_teacher + init_weights 覆盖')
    print('=' * 70)

    from ldmdet.core.head import DiffusionDetHead
    from ldmdet.core.single_head import SingleDiffusionDetHead
    from ldmdet.core.roi_extractor import SingleRoIExtractor

    # 构建 Student (H=3) 和 Teacher (H=6) 的辅助函数
    def make_head(num_heads):
        single_head = SingleDiffusionDetHead(
            num_classes=24,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,  # 与 +DPM-Solver++ 配置一致 (ldmdet_baseline.py)
            use_focal_loss=True,
            use_normalized_classifier=False,
            time_conditioning='adaln_zero',
        )
        roi_extractor = SingleRoIExtractor(
            featmap_strides=[16],
            out_channels=256,
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True),
        )
        return DiffusionDetHead(
            num_classes=24,
            feat_channels=256,
            num_proposals=500,
            num_heads=num_heads,
            single_head=single_head,
            roi_extractor=roi_extractor,
            criterion=None,
            use_distillation=(num_heads == 3),  # Student 启用蒸馏
            distill_lambda=0.05,
            distill_head_map={0: 0, 1: 2, 2: 5},
        )

    # 1. 构建 Student + Teacher
    print('\n[1] 构建 Student (H=3) + Teacher (H=6)...')
    student = make_head(num_heads=3)
    teacher = make_head(num_heads=6)
    print(f'  Student head_series 数量: {len(student.head_series)}')
    print(f'  Teacher head_series 数量: {len(teacher.head_series)}')

    # 2. 加载 +DPM-Solver++ checkpoint 到 Teacher
    print(f'\n[2] 加载 +DPM-Solver++ checkpoint 到 Teacher: {args.teacher_ckpt}')
    if not os.path.exists(args.teacher_ckpt):
        print(f'  ✗ 文件不存在!')
        return

    checkpoint = torch.load(args.teacher_ckpt, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    # 提取 bbox_head.* 前缀的权重
    prefix = 'bbox_head.'
    head_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            head_state[k[len(prefix):]] = v

    print(f'  checkpoint 中 bbox_head.* 键数: {len(head_state)}')
    print(f'  Teacher state_dict 键数: {len(teacher.state_dict())}')

    missing, unexpected = teacher.load_state_dict(head_state, strict=False)
    print(f'  Teacher missing keys: {len(missing)}')
    print(f'    前 10 个: {missing[:10]}')
    print(f'  Teacher unexpected keys: {len(unexpected)}')
    print(f'    前 10 个: {unexpected[:10]}')

    # 分析 missing/unexpected 的模式
    print(f'\n  --- missing keys 模式分析 ---')
    missing_prefixes = set()
    for k in missing:
        parts = k.split('.')
        if len(parts) >= 3:
            missing_prefixes.add('.'.join(parts[:3]))
        else:
            missing_prefixes.add(k)
    for p in sorted(missing_prefixes):
        print(f'    {p}')

    print(f'\n  --- unexpected keys 模式分析 ---')
    unexpected_prefixes = set()
    for k in unexpected:
        parts = k.split('.')
        if len(parts) >= 3:
            unexpected_prefixes.add('.'.join(parts[:3]))
        else:
            unexpected_prefixes.add(k)
    for p in sorted(unexpected_prefixes):
        print(f'    {p}')

    # 检查 Teacher head 0/2/5 是否加载了非随机权重
    for t_idx in [0, 2, 5]:
        w = teacher.head_series[t_idx].reg_head[0][0].weight
        print(f'  Teacher head[{t_idx}] reg_head[0][0].weight: mean={w.mean().item():.6f}, std={w.std().item():.6f}')

    # 3. 注入 Teacher 并调用 init_student_from_teacher
    print(f'\n[3] 注入 Teacher + 调用 init_student_from_teacher...')
    student.set_teacher(teacher)
    student.init_student_from_teacher()

    # 4. 检查 Student head 是否与 Teacher 对应 head 一致
    print(f'\n[4] Student head vs Teacher 对应 head 权重对比 (init_student_from_teacher 后):')
    distill_head_map = {0: 0, 1: 2, 2: 5}
    all_identical = True
    for s_idx, t_idx in distill_head_map.items():
        s_w = student.head_series[s_idx].reg_head[0][0].weight
        t_w = teacher.head_series[t_idx].reg_head[0][0].weight
        identical = torch.equal(s_w, t_w)
        max_diff = (s_w - t_w).abs().max().item()
        status = '✓ 一致' if identical else '✗ 不一致'
        print(f'  Student head[{s_idx}] vs Teacher head[{t_idx}]: {status} (max_diff={max_diff:.8f})')
        if not identical:
            all_identical = False

    if all_identical:
        print('\n  ✓ init_student_from_teacher 已正确生效!')
    else:
        print('\n  ✗ init_student_from_teacher 未完全生效!')

    # 5. 检查 Student head 之间差异 (证明非全部复制同一 head)
    print(f'\n[5] Student head 之间差异:')
    for i in range(3):
        for j in range(i + 1, 3):
            s_i = student.head_series[i].reg_head[0][0].weight
            s_j = student.head_series[j].reg_head[0][0].weight
            diff = (s_i - s_j).abs().max().item()
            print(f'  Student head[{i}] vs head[{j}]: max_diff={diff:.8f}')

    # 6. 模拟 init_weights: 检查是否覆盖
    print(f'\n[6] 模拟 init_weights 覆盖检查:')
    # Student head 是 nn.Module (非 BaseModule), init_weights 不应影响
    # 但手动调用 student._init_weights 看看
    s_head2_before = student.head_series[2].reg_head[0][0].weight.clone()
    t_head5 = teacher.head_series[5].reg_head[0][0].weight
    print(f'  init_weights 前: Student head[2] vs Teacher head[5] identical = {torch.equal(s_head2_before, t_head5)}')

    # mmengine BaseModule.init_weights 会遍历子模块调用 init_weights
    # 但 DiffusionDetHead 是 nn.Module, 不会被调用
    # 这里手动调用 _init_weights (只设 prior_prob, 不影响 conv)
    try:
        student._init_weights(prior_prob=0.01)
        print(f'  student._init_weights() 调用成功 (仅设 prior_prob)')
    except Exception as e:
        print(f'  student._init_weights() 异常: {e}')

    s_head2_after = student.head_series[2].reg_head[0][0].weight.clone()
    print(f'  _init_weights 后: Student head[2] vs Teacher head[5] identical = {torch.equal(s_head2_after, t_head5)}')
    print(f'  _init_weights 前后 Student head[2] 变化: {not torch.equal(s_head2_before, s_head2_after)}')

    # 7. 关键诊断: 检查 checkpoint 中 head_series 的结构
    print(f'\n[7] checkpoint head_series 结构分析:')
    head_series_keys = [k for k in head_state.keys() if k.startswith('head_series.')]
    head_indices = set()
    for k in head_series_keys:
        parts = k.split('.')
        if len(parts) >= 2 and parts[0] == 'head_series':
            try:
                head_indices.add(int(parts[1]))
            except ValueError:
                pass
    print(f'  checkpoint 中 head_series 索引: {sorted(head_indices)}')
    print(f'  Teacher 需要 head_series 索引: 0-5 (H=6)')

    print('\n' + '=' * 70)
    print('诊断完成')
    print('=' * 70)


if __name__ == '__main__':
    main()
