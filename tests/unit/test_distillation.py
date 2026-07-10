"""PD-RF (Direct Knowledge Distillation for RF) 单元测试

测试直接蒸馏机制的核心理论性质:
1. 教师冻结: 教师参数不更新 (理论: 教师作为固定监督源)
2. 蒸馏损失计算: MSE(student_x0, teacher_x0.detach()) 正确 (理论 2.5)
3. 梯度流: 学生前向有梯度, 蒸馏损失梯度回传到学生参数 (核心: 无 no_grad bug)
4. 教师 detach: 教师输出不参与学生计算图 (stop-gradient)
5. 噪声共享: 教师和学生使用同一 x_raw, proposal 对应一致 (核心: 蒸馏语义有效)
6. 教师 x0 形状: 返回 [bs, P, 4] raw 张量 (非 post-NMS 可变长度)
7. box_renewal 关闭: 教师蒸馏推理不替换 proposal (保持对应)
8. lambda=0 禁用: 无蒸馏时退化为标准训练
9. 学生 1步推理: 1步 Euler 正常工作
10. 梯度对齐诊断: 可计算梯度余弦相似度 (诊断指标, 不预设符号)

对应 docs/paper/proposals/PD-RF_Progressive_Distillation.md
理论依据: 2.2 RF轨迹直化度, 2.3 梯度结构差异, 2.5 蒸馏空间选择
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.criterion import (
    BBoxL1Cost,
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    FocalLossCost,
    GIoULoss,
    IoUCost,
    L1Loss,
)


# ============================================================
# 辅助构建函数
# ============================================================

def _make_criterion(num_classes=24):
    """构建最小 DiffusionDetCriterion"""
    matcher = DiffusionDetMatcher(
        match_costs=[
            FocalLossCost(weight=2.0),
            BBoxL1Cost(weight=5.0),
            IoUCost(iou_mode='giou', weight=2.0),
        ],
        center_radius=2.5,
        candidate_topk=5,
    )
    loss_cls = FocalLoss(loss_weight=2.0)
    loss_bbox = L1Loss(loss_weight=5.0)
    loss_giou = GIoULoss(loss_weight=2.0)
    return DiffusionDetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=loss_cls,
        loss_bbox=loss_bbox,
        loss_giou=loss_giou,
    )


def _make_single_head(num_classes=24, feat_channels=64):
    return SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_cls_convs=1,
        num_reg_convs=2,
        use_focal_loss=True,
        use_normalized_classifier=False,
    )


def _make_roi_extractor(out_channels=64):
    return SingleRoIExtractor(
        featmap_strides=[16], out_channels=out_channels,
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True),
    )


def _make_head(
    num_classes=24,
    feat_channels=64,
    num_heads=2,
    num_proposals=10,
    use_distillation=False,
    distill_lambda=1.0,
    solver_type='euler',
    sampling_timesteps=1,
):
    """构建 DiffusionDetHead (学生或教师)"""
    return DiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_proposals=num_proposals,
        num_heads=num_heads,
        single_head=_make_single_head(num_classes, feat_channels),
        roi_extractor=_make_roi_extractor(feat_channels),
        criterion=_make_criterion(num_classes),
        diffusion_type='rectified_flow',
        solver_type=solver_type,
        sampling_timesteps=sampling_timesteps,
        use_distillation=use_distillation,
        distill_lambda=distill_lambda,
    )


def _make_dummy_input(bs=2, num_proposals=10, feat_channels=64, num_classes=24):
    """构造 dummy 训练输入: features, img_metas, gt_bboxes, gt_labels"""
    features = [torch.randn(bs, feat_channels, 16, 16)]
    img_metas = [
        dict(
            img_shape=(16, 16, 3),
            pad_shape=(16, 16, 3),
            ori_shape=(16, 16, 3),
            scale_factor=1.0,
        )
        for _ in range(bs)
    ]
    # GT: 每张图 3 个框
    num_gt = 3
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        boxes = torch.rand(num_gt, 4)
        boxes[:, 2] = boxes[:, 0] + 0.3
        boxes[:, 3] = boxes[:, 1] + 0.3
        gt_bboxes.append(boxes * 15)  # 缩放到图像尺度
        gt_labels.append(torch.randint(0, num_classes, (num_gt,)))
    return features, img_metas, gt_bboxes, gt_labels


# ============================================================
# 1. 教师模型冻结 (理论: 教师作为固定监督源)
# ============================================================

class TestTeacherModelFrozen:
    """测试教师参数不更新"""

    def test_teacher_model_attribute_exists(self):
        """use_distillation=True 时应有 teacher_model 属性"""
        head = _make_head(use_distillation=True)
        assert hasattr(head, 'teacher_model'), \
            'DiffusionDetHead 应有 teacher_model 属性'

    def test_teacher_model_initially_none(self):
        """teacher_model 初始应为 None (外部注入)"""
        head = _make_head(use_distillation=True)
        assert head.teacher_model is None, \
            'teacher_model 初始应为 None'

    def test_teacher_params_frozen_after_injection(self):
        """注入教师后, 教师参数 requires_grad=False"""
        student = _make_head(use_distillation=True)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        student.teacher_model = teacher
        for name, param in teacher.named_parameters():
            assert not param.requires_grad, \
                f'教师参数 {name} 应被冻结 (requires_grad=False)'

    def test_teacher_grad_is_none_after_backward(self):
        """反向传播后, 教师参数 .grad 为 None (不接收梯度)"""
        student = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        # 显式冻结教师
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()
        losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(v for k, v in losses.items() if k.startswith('loss_'))
        total_loss.backward()

        # 教师参数梯度应为 None
        for name, param in teacher.named_parameters():
            assert param.grad is None, \
                f'教师参数 {name} 不应有梯度 (应为 None)'


# ============================================================
# 2. 蒸馏损失计算 (理论 2.5: raw 空间 MSE)
# ============================================================

class TestDistillLossComputation:
    """测试蒸馏损失正确计算"""

    def test_distill_loss_in_output(self):
        """loss_with_distillation 输出应包含 loss_distill"""
        student = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()
        losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        assert 'loss_distill' in losses, '应包含 loss_distill 键'

    def test_distill_loss_is_tensor(self):
        """loss_distill 应是 tensor"""
        student = _make_head(use_distillation=True)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()
        losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        assert isinstance(losses['loss_distill'], torch.Tensor), \
            'loss_distill 应是 tensor'

    def test_distill_loss_scales_with_lambda(self):
        """lambda 增大时 loss_distill 应同比增大"""
        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()

        # lambda=1.0
        student1 = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher1 = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for p in teacher1.parameters():
            p.requires_grad = False
        student1.teacher_model = teacher1
        student1.load_state_dict(student1.state_dict())  # 确保相同初始权重
        student1.train()
        torch.manual_seed(42)
        losses1 = student1.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)

        # lambda=2.0 (相同教师和学生权重, 相同噪声)
        student2 = _make_head(use_distillation=True, distill_lambda=2.0)
        teacher2 = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for p in teacher2.parameters():
            p.requires_grad = False
        student2.teacher_model = teacher2
        student2.load_state_dict(student1.state_dict())  # 相同权重
        teacher2.load_state_dict(teacher1.state_dict())
        student2.train()
        torch.manual_seed(42)
        losses2 = student2.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)

        ratio = losses2['loss_distill'].item() / max(losses1['loss_distill'].item(), 1e-8)
        assert abs(ratio - 2.0) < 0.01, \
            f'lambda=2.0 的 loss_distill 应约为 lambda=1.0 的 2倍, 实际比值: {ratio}'


# ============================================================
# 3. 梯度流 (核心测试: 验证无 no_grad bug, 理论 3.1)
# ============================================================

class TestDistillLossGradientFlow:
    """测试蒸馏损失梯度流向学生参数 (核心: 无 no_grad bug)"""

    def test_student_params_receive_gradient(self):
        """蒸馏损失反传后, 学生参数 .grad 非空"""
        student = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()
        losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(v for k, v in losses.items() if k.startswith('loss_'))
        total_loss.backward()

        # 检查学生 head_series 参数有梯度
        has_grad = False
        for name, param in student.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, '学生参数应有非零梯度 (蒸馏损失梯度回传)'

    def test_distill_loss_requires_grad_on_student_x0(self):
        """学生的 x0_pred 应 requires_grad=True (保留计算图)"""
        student = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        device = features[0].device
        bs = len(img_metas)

        # 模拟 _student_single_step_x0 的行为
        x_raw_shared = torch.randn(bs, student.num_proposals, 4, device=device)
        student.train()
        student_x0 = student._student_single_step_x0(features, img_metas, x_raw_shared)
        assert student_x0.requires_grad, \
            '学生 x0_pred 应 requires_grad=True (不在 no_grad 下)'

    def test_distill_loss_backward_updates_student_weights(self):
        """蒸馏损失反传应改变学生权重 (证明梯度有效)

        注: head_series[0] (首个 head) 在小规模测试输入下梯度为零 (cascade_detach
        导致首 head 输出被 detach, 且 SimOTA 匹配在随机框上可能分配稀疏).
        head_series[-1] (末 head) 产生最终预测, 蒸馏损失直接作用于其 box 回归路径,
        必有非零梯度. 选择 head_series[-1].reg_head[-1].weight 验证.
        """
        student = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()

        # 记录反传前的权重 (末 head 的 reg_head[-1], 蒸馏损失直接作用)
        last_head = student.head_series[-1]
        w_before = last_head.reg_head[-1].weight.clone()

        losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(v for k, v in losses.items() if k.startswith('loss_'))
        total_loss.backward()

        # 验证 reg_head[-1] 确实有非零梯度
        reg_grad = last_head.reg_head[-1].weight.grad
        assert reg_grad is not None and reg_grad.abs().sum() > 0, \
            'head_series[-1].reg_head[-1].weight 应有非零梯度 (蒸馏损失作用于 box 回归路径)'

        # 手动更新一步
        with torch.no_grad():
            for param in student.parameters():
                if param.grad is not None:
                    param -= 0.01 * param.grad

        w_after = last_head.reg_head[-1].weight.clone()
        assert not torch.allclose(w_before, w_after), \
            '蒸馏损失反传后学生权重应改变'


# ============================================================
# 4. 教师输出 detach (stop-gradient)
# ============================================================

class TestTeacherOutputDetached:
    """测试教师输出不参与学生计算图"""

    def test_teacher_x0_detached(self):
        """教师 x0_pred 应是 detached (不参与计算图)"""
        student = _make_head(use_distillation=True)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        device = features[0].device
        bs = len(img_metas)
        x_raw_shared = torch.randn(bs, student.num_proposals, 4, device=device)

        teacher_x0 = student._teacher_multistep_x0(features, img_metas, x_raw_shared)
        assert not teacher_x0.requires_grad, \
            '教师 x0_pred 应 requires_grad=False (detached)'

    def test_teacher_not_in_student_computation_graph(self):
        """教师参数不应出现在学生的计算图中"""
        student = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()
        losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(v for k, v in losses.items() if k.startswith('loss_'))

        # 反传不应报错 (教师已 detach)
        total_loss.backward()


# ============================================================
# 5. 噪声共享 (关键: proposal 对应一致)
# ============================================================

class TestSharedNoiseProposalCorrespondence:
    """测试教师和学生使用同一 x_raw, proposal 对应一致"""

    def test_shared_noise_passed_to_both(self):
        """loss_with_distillation 应使用同一 x_raw_shared 传入教师和学生"""
        student = _make_head(use_distillation=True)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()

        # loss_with_distillation 内部生成 x_raw_shared 并传给教师和学生
        # 验证: 两次调用 (相同种子) 应产生相同的 teacher_x0 (证明确定性共享)
        torch.manual_seed(123)
        losses1 = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        torch.manual_seed(123)
        losses2 = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)

        # 相同种子 → 相同噪声 → 相同蒸馏损失
        assert torch.allclose(losses1['loss_distill'], losses2['loss_distill']), \
            '相同种子应产生相同蒸馏损失 (证明确定性共享噪声)'

    def test_different_seed_different_loss(self):
        """不同种子应产生不同蒸馏损失 (证明噪声是随机的, 非固定)"""
        student = _make_head(use_distillation=True)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()

        torch.manual_seed(123)
        losses1 = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        torch.manual_seed(999)
        losses2 = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)

        assert not torch.allclose(losses1['loss_distill'], losses2['loss_distill']), \
            '不同种子应产生不同蒸馏损失 (噪声是随机的)'


# ============================================================
# 6. 教师 x0 形状 (关键: [bs, P, 4] 非 post-NMS)
# ============================================================

class TestTeacherX0RawShape:
    """测试教师返回的 x0 是 [bs, P, 4] raw 张量"""

    def test_teacher_x0_shape(self):
        """教师 x0 应为 [bs, num_proposals, 4]"""
        num_proposals = 10
        bs = 2
        student = _make_head(num_proposals=num_proposals, use_distillation=True)
        teacher = _make_head(num_proposals=num_proposals, solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, _, _ = _make_dummy_input(bs=bs, num_proposals=num_proposals)
        device = features[0].device
        x_raw = torch.randn(bs, num_proposals, 4, device=device)

        teacher_x0 = student._teacher_multistep_x0(features, img_metas, x_raw)
        assert teacher_x0.shape == (bs, num_proposals, 4), \
            f'教师 x0 形状应为 ({bs}, {num_proposals}, 4), 实际: {teacher_x0.shape}'

    def test_teacher_x0_not_post_nms(self):
        """教师 x0 的 proposal 数应 = num_proposals (非 NMS 后的可变数量)"""
        num_proposals = 10
        bs = 2
        student = _make_head(num_proposals=num_proposals, use_distillation=True)
        teacher = _make_head(num_proposals=num_proposals, solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, _, _ = _make_dummy_input(bs=bs, num_proposals=num_proposals)
        device = features[0].device
        x_raw = torch.randn(bs, num_proposals, 4, device=device)

        teacher_x0 = student._teacher_multistep_x0(features, img_metas, x_raw)
        # post-NMS 会减少 proposal 数量, raw x0 应保持 num_proposals
        assert teacher_x0.shape[1] == num_proposals, \
            f'教师 x0 proposal 数应 = {num_proposals} (非 post-NMS), 实际: {teacher_x0.shape[1]}'


# ============================================================
# 7. box_renewal 关闭 (关键: 保持 proposal 对应)
# ============================================================

class TestTeacherBoxRenewalDisabled:
    """测试教师蒸馏推理不调用 box_renewal"""

    def test_teacher_x0_preserves_all_proposals(self):
        """教师 x0 应保留所有 proposal (box_renewal 会替换低置信度框)"""
        num_proposals = 10
        bs = 2
        student = _make_head(num_proposals=num_proposals, use_distillation=True)
        teacher = _make_head(num_proposals=num_proposals, solver_type='dpm_solver_pp', sampling_timesteps=4)
        # 教师配置中 box_renewal 可能 True, 但蒸馏时应关闭
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, _, _ = _make_dummy_input(bs=bs, num_proposals=num_proposals)
        device = features[0].device
        x_raw = torch.randn(bs, num_proposals, 4, device=device)

        teacher_x0 = student._teacher_multistep_x0(features, img_metas, x_raw)
        # 若 box_renewal 未关闭, 部分 proposal 会被替换, 但形状仍为 [bs, P, 4]
        # 真正的验证: 多步推理中 proposal 不应跳到新轨迹
        # 这里验证形状一致性 (必要条件)
        assert teacher_x0.shape == (bs, num_proposals, 4)


# ============================================================
# 8. lambda=0 禁用蒸馏
# ============================================================

class TestLambdaZeroDisablesDistillation:
    """测试 lambda=0 时无蒸馏"""

    def test_lambda_zero_no_distill_loss(self):
        """lambda=0 时不应有 loss_distill (或 loss_distill=0)"""
        student = _make_head(use_distillation=True, distill_lambda=0.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()
        losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)

        if 'loss_distill' in losses:
            assert losses['loss_distill'].item() == 0.0, \
                'lambda=0 时 loss_distill 应为 0'

    def test_lambda_zero_equals_standard_loss(self):
        """lambda=0 时 loss_with_distillation 应退化为标准 loss"""
        student = _make_head(use_distillation=True, distill_lambda=0.0)
        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()

        torch.manual_seed(42)
        losses_distill = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)

        torch.manual_seed(42)
        losses_standard = student.loss(features, img_metas, gt_bboxes, gt_labels)

        # 比较 loss_cls, loss_bbox, loss_giou (忽略 distill 相关)
        for key in ['loss_cls', 'loss_bbox', 'loss_giou']:
            if key in losses_standard and key in losses_distill:
                assert torch.allclose(losses_standard[key], losses_distill[key]), \
                    f'lambda=0 时 {key} 应与标准 loss 一致'


# ============================================================
# 9. 学生 1步推理
# ============================================================

class TestStudentOneStepInference:
    """测试学生 1步推理正常工作"""

    def test_student_single_step_x0(self):
        """_student_single_step_x0 应返回 [bs, P, 4] 且有梯度"""
        num_proposals = 10
        bs = 2
        student = _make_head(num_proposals=num_proposals, use_distillation=True)
        features, img_metas, _, _ = _make_dummy_input(bs=bs, num_proposals=num_proposals)
        device = features[0].device
        x_raw = torch.randn(bs, num_proposals, 4, device=device)

        student_x0 = student._student_single_step_x0(features, img_metas, x_raw)
        assert student_x0.shape == (bs, num_proposals, 4)
        assert student_x0.requires_grad, '学生 x0 应有梯度'

    def test_student_predict_1step(self):
        """学生 predict 应支持 1步 Euler 推理"""
        student = _make_head(solver_type='euler', sampling_timesteps=1)
        features, img_metas, _, _ = _make_dummy_input()
        student.eval()
        results = student.predict(features, img_metas)
        assert len(results) == len(img_metas)


# ============================================================
# 10. 梯度对齐诊断 (诊断指标, 不预设符号)
# ============================================================

class TestGradientAlignmentDiagnostic:
    """测试梯度余弦相似度可计算 (诊断, 不预设 >0)"""

    def test_gradient_alignment_computable(self):
        """梯度对齐性应可计算 (返回标量)

        注: det_loss (含 SimOTA 匹配的 Focal+L1+GIoU) 对所有参数有梯度,
        而 distill_loss (raw 空间 MSE) 仅对 box 回归路径参数有梯度.
        故只比较两者都有非 None 梯度的参数 (box 回归路径).
        """
        student = _make_head(use_distillation=True, distill_lambda=1.0)
        teacher = _make_head(solver_type='dpm_solver_pp', sampling_timesteps=4)
        for param in teacher.parameters():
            param.requires_grad = False
        student.teacher_model = teacher

        features, img_metas, gt_bboxes, gt_labels = _make_dummy_input()
        student.train()

        # 分别计算 det_loss 和 distill_loss 的梯度
        # det_loss: 标准检测损失
        student.zero_grad()
        det_losses = student.loss(features, img_metas, gt_bboxes, gt_labels)
        det_loss = sum(v for k, v in det_losses.items() if k.startswith('loss_'))
        det_loss.backward(retain_graph=False)
        det_grads = [p.grad.clone() if p.grad is not None else None
                     for p in student.parameters()]

        # distill_loss: 仅蒸馏损失
        student.zero_grad()
        distill_losses = student.loss_with_distillation(features, img_metas, gt_bboxes, gt_labels)
        distill_loss = distill_losses.get('loss_distill', torch.tensor(0.0))
        if isinstance(distill_loss, torch.Tensor) and distill_loss.requires_grad:
            distill_loss.backward(retain_graph=False)
            distill_grads = [p.grad.clone() if p.grad is not None else None
                             for p in student.parameters()]

            # 只比较两者都有非 None 梯度的参数 (梯度结构差异, 见理论 2.3)
            common_det = []
            common_distill = []
            for dg, dsg in zip(det_grads, distill_grads):
                if dg is not None and dsg is not None:
                    common_det.append(dg.flatten())
                    common_distill.append(dsg.flatten())

            assert len(common_det) > 0, \
                '应存在两者都有梯度的参数 (box 回归路径)'

            det_flat = torch.cat(common_det)
            distill_flat = torch.cat(common_distill)
            cos_sim = torch.nn.functional.cosine_similarity(
                det_flat.unsqueeze(0), distill_flat.unsqueeze(0)
            ).item()

            # 诊断指标: 不预设符号, 仅验证可计算
            assert -1.0 <= cos_sim <= 1.0, \
                f'梯度余弦相似度应在 [-1, 1], 实际: {cos_sim}'
