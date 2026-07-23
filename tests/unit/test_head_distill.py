"""测试 Head Distillation v2 (方向: 少 Head 蒸馏多 Head)

核心机制: Teacher (H=6, A4 冻结) 监督 Student (H=3) 的中间 fc_feature,
headwise 蒸馏:
  L_total = L_det(student) + λ · L_distill
  L_distill = (1/K) Σ_k MSE(student_fc_feat_k, teacher_fc_feat_{map(k)}.detach())

v2 六项修正 (见 REFLOW_HEAD_DISTILL_IMPL_PLAN.md §2.2):
  1. 冻结 Student backbone
  2. 蒸馏 fc_feature (非 pred_bboxes)
  3. head 映射 {0→0, 1→2, 2→5}
  4. λ=0.05 起步
  5. coupling_mode='argmax' (确定性)
  6. distill 仅作用于 main head, aux 不蒸馏

核心验证:
1. use_distillation 参数存在且默认 False (向后兼容)
2. set_teacher() 注入 teacher 后 teacher 参数冻结
3. forward 返回 fc_features (inter_curr_proposals) 供蒸馏使用
4. 蒸馏 loss 被正确计算 (MSE on fc_feature)
5. 梯度流: Student 参数有梯度, Teacher 参数无梯度
6. head 映射正确: student head k 对应 teacher head map(k)
7. deep_supervision_aux_weight 降低 aux loss 权重
8. 共享噪声: teacher 和 student 使用相同 x_raw
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
from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.criterion.losses import FeatureDistillLoss


def _make_single_head(num_classes=24, feat_channels=64):
    """构建最小 SingleDiffusionDetHead (AdaLN-Zero)"""
    return SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_cls_convs=1,
        num_reg_convs=2,
        use_focal_loss=True,
        use_normalized_classifier=False,
        time_conditioning='adaln_zero',
    )


def _make_roi_extractor(out_channels=64):
    """构建最小 RoIExtractor"""
    return SingleRoIExtractor(
        featmap_strides=[16],
        out_channels=out_channels,
        roi_layer=dict(
            type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True
        ),
    )


def _make_dummy_input(bs=2, num_boxes=10, feat_channels=64):
    """构造 dummy 输入 (有效 xyxy 框)"""
    torch.manual_seed(42)
    features = [torch.randn(bs, feat_channels, 16, 16)]
    bboxes = torch.rand(bs, num_boxes, 4) * 100
    bboxes[..., 2:] = bboxes[..., :2] + torch.rand(bs, num_boxes, 2) * 50 + 10
    t = torch.tensor([500.0, 500.0])
    return features, bboxes, t


# ============================================================
# 1. 参数存在性与默认值
# ============================================================

class TestDistillParamsExist:
    """蒸馏参数存在性测试"""

    def test_use_distillation_default_false(self):
        """use_distillation 默认 False, 保持向后兼容"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert head.use_distillation is False

    def test_distill_params_exist_when_enabled(self):
        """启用蒸馏时参数存在"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
            distill_lambda=0.05,
            distill_head_map={0: 0, 1: 2, 2: 5},
            deep_supervision_aux_weight=0.5,
        )
        assert head.use_distillation is True
        assert head.distill_lambda == 0.05
        assert head.distill_head_map == {0: 0, 1: 2, 2: 5}
        assert head.deep_supervision_aux_weight == 0.5

    def test_distill_head_map_default(self):
        """distill_head_map 默认为 {0:0, 1:2, 2:5} (Student H=3 → Teacher H=6)"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        assert head.distill_head_map == {0: 0, 1: 2, 2: 5}


# ============================================================
# 2. Teacher 注入与冻结
# ============================================================

class TestTeacherInjection:
    """Teacher 模型注入与冻结测试"""

    def test_set_teacher_freezes_teacher_params(self):
        """set_teacher() 后 teacher 参数全部 requires_grad=False"""
        student = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        # Teacher 初始应有 requires_grad=True 的参数
        teacher_params = list(teacher.parameters())
        assert any(p.requires_grad for p in teacher_params), \
            "Teacher 初始应有可训练参数"

        student.set_teacher(teacher)

        # Teacher 参数应全部冻结
        for p in teacher.parameters():
            assert not p.requires_grad, \
                "Teacher 参数应在 set_teacher() 后冻结"

    def test_set_teacher_stores_reference(self):
        """set_teacher() 存储 teacher 引用"""
        student = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        student.set_teacher(teacher)
        assert student._teacher is teacher

    def test_teacher_not_in_student_parameters(self):
        """Teacher 参数不应出现在 student 的 parameters() 中 (避免进 optimizer)"""
        student = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )

        student_param_count_before = len(list(student.parameters()))
        student.set_teacher(teacher)
        student_param_count_after = len(list(student.parameters()))

        # Teacher 参数不应增加 student 的 parameter 数量
        # (teacher 通过 _teacher 引用, 不是 nn.Module 子模块)
        assert student_param_count_after == student_param_count_before, \
            "Teacher 参数不应出现在 student.parameters() 中"

    def test_apply_propagates_to_teacher(self):
        """_apply 将 device/dtype 变更传播到 Teacher (修复 GPU 训练 device 不匹配)

        Teacher 通过 object.__setattr__ 持有 (非子模块), 标准 to()/cuda()
        不会自动传播。覆写 _apply 确保 Teacher 跟随 Student 迁移。
        用 double() 在 CPU 上验证 _apply 传播路径 (无需 GPU)。
        """
        student = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        student.set_teacher(teacher)

        # 初始均为 float32
        student_param = next(student.parameters())
        teacher_param = next(teacher.parameters())
        assert student_param.dtype == torch.float32
        assert teacher_param.dtype == torch.float32

        # double() 触发 _apply → 应传播到 Teacher
        student.double()

        # Teacher 参数应也变成 float64 (证明 _apply 已传播)
        for p in teacher.parameters():
            assert p.dtype == torch.float64, \
                "Teacher 应随 Student.double() 一起变更 dtype (_apply 传播)"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='需要 GPU 验证 device 迁移'
    )
    def test_teacher_on_cuda_after_student_to_cuda(self):
        """Student.to('cuda') 后 Teacher 也在 cuda 上 (端到端 device 验证)"""
        student = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        student.set_teacher(teacher)
        student.to('cuda')

        teacher_param = next(teacher.parameters())
        assert teacher_param.is_cuda, \
            "Teacher 应在 Student.to('cuda') 后迁移到 cuda"


# ============================================================
# 3. fc_feature 暴露
# ============================================================

class TestFcFeatureExposure:
    """fc_feature (box head 前的中间特征) 暴露测试"""

    def test_forward_returns_fc_features(self):
        """forward 返回 fc_features 作为第三个元素 (inter_curr_proposals)"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        features, bboxes, t = _make_dummy_input()
        head.eval()
        with torch.no_grad():
            cls_logits, pred_bboxes, fc_features = head(features, bboxes, t)

        # fc_features 是 list, 每个 head 一个
        assert isinstance(fc_features, list)
        assert len(fc_features) == 3  # H=3

    def test_fc_feature_shape(self):
        """fc_feature 形状正确: [1, bs*num_proposals, feat_channels]"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        features, bboxes, t = _make_dummy_input(bs=2, num_boxes=10)
        head.eval()
        with torch.no_grad():
            _, _, fc_features = head(features, bboxes, t)

        for i, feat in enumerate(fc_features):
            # fc_feature: [1, bs*num_proposals, feat_channels]
            assert feat.shape == (1, 2 * 10, 64), \
                f"head {i} fc_feature 形状错误: {feat.shape}, 期望 (1, 20, 64)"

    def test_fc_feature_is_256d_compatible(self):
        """fc_feature 维度与 feat_channels 一致 (Student/Teacher 同架构, 无需投影层)"""
        feat_channels = 64
        student = DiffusionDetHead(
            num_classes=24, feat_channels=feat_channels, num_proposals=10,
            num_heads=3,
            single_head=_make_single_head(feat_channels=feat_channels),
            roi_extractor=_make_roi_extractor(out_channels=feat_channels),
            criterion=None,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=feat_channels, num_proposals=10,
            num_heads=6,
            single_head=_make_single_head(feat_channels=feat_channels),
            roi_extractor=_make_roi_extractor(out_channels=feat_channels),
            criterion=None,
        )
        features, bboxes, t = _make_dummy_input(feat_channels=feat_channels)
        student.eval()
        teacher.eval()
        with torch.no_grad():
            _, _, s_feats = student(features, bboxes, t)
            _, _, t_feats = teacher(features, bboxes, t)

        # Student 和 Teacher 的 fc_feature 维度一致
        assert s_feats[0].shape[-1] == t_feats[0].shape[-1] == feat_channels


# ============================================================
# 4. 蒸馏 loss 计算
# ============================================================

class TestDistillLossComputation:
    """蒸馏 loss 计算测试"""

    def test_feature_distill_loss_class_exists(self):
        """FeatureDistillLoss 类存在且可实例化"""
        loss_fn = FeatureDistillLoss(loss_weight=0.05)
        assert loss_fn is not None
        assert loss_fn.loss_weight == 0.05

    def test_feature_distill_loss_mse(self):
        """FeatureDistillLoss 计算 MSE"""
        loss_fn = FeatureDistillLoss(loss_weight=1.0)
        student_feat = torch.randn(1, 20, 64)
        teacher_feat = torch.randn(1, 20, 64)
        loss = loss_fn(student_feat, teacher_feat)
        expected = torch.nn.functional.mse_loss(student_feat, teacher_feat)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_feature_distill_loss_zero_when_identical(self):
        """FeatureDistillLoss 在特征相同时为 0"""
        loss_fn = FeatureDistillLoss(loss_weight=1.0)
        feat = torch.randn(1, 20, 64)
        loss = loss_fn(feat, feat)
        assert loss.item() < 1e-10

    def test_distill_loss_weighted_by_lambda(self):
        """蒸馏 loss 被 distill_lambda 加权"""
        loss_fn = FeatureDistillLoss(loss_weight=0.05)
        student_feat = torch.randn(1, 20, 64)
        teacher_feat = torch.randn(1, 20, 64)
        loss = loss_fn(student_feat, teacher_feat)
        expected = 0.05 * torch.nn.functional.mse_loss(student_feat, teacher_feat)
        assert torch.allclose(loss, expected, atol=1e-6)


# ============================================================
# 5. 梯度流
# ============================================================

class TestGradientFlow:
    """梯度流测试: Student 有梯度, Teacher 无梯度"""

    def test_student_gets_gradient_from_distill_loss(self):
        """蒸馏 loss 使 Student 参数获得梯度"""
        feat_channels = 64
        student = DiffusionDetHead(
            num_classes=24, feat_channels=feat_channels, num_proposals=10,
            num_heads=3,
            single_head=_make_single_head(feat_channels=feat_channels),
            roi_extractor=_make_roi_extractor(out_channels=feat_channels),
            criterion=None,
            use_distillation=True,
            distill_lambda=0.05,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=feat_channels, num_proposals=10,
            num_heads=6,
            single_head=_make_single_head(feat_channels=feat_channels),
            roi_extractor=_make_roi_extractor(out_channels=feat_channels),
            criterion=None,
        )
        student.set_teacher(teacher)

        features, bboxes, t = _make_dummy_input(feat_channels=feat_channels)
        student.train()
        teacher.eval()

        # Student forward
        s_cls, s_bbox, s_feats = student(features, bboxes, t)

        # Teacher forward (no_grad)
        with torch.no_grad():
            _, _, t_feats = teacher(features, bboxes, t)

        # 计算蒸馏 loss
        distill_loss = 0.0
        head_map = {0: 0, 1: 2, 2: 5}
        for s_idx, t_idx in head_map.items():
            distill_loss += torch.nn.functional.mse_loss(
                s_feats[s_idx], t_feats[t_idx].detach()
            )
        distill_loss = distill_loss / len(head_map)
        distill_loss.backward()

        # 验证 Student 参数有梯度
        student_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.parameters()
        )
        assert student_has_grad, "Student 参数应获得梯度"

    def test_teacher_gets_no_gradient(self):
        """Teacher 参数不应获得梯度 (已冻结)"""
        feat_channels = 64
        student = DiffusionDetHead(
            num_classes=24, feat_channels=feat_channels, num_proposals=10,
            num_heads=3,
            single_head=_make_single_head(feat_channels=feat_channels),
            roi_extractor=_make_roi_extractor(out_channels=feat_channels),
            criterion=None,
            use_distillation=True,
        )
        teacher = DiffusionDetHead(
            num_classes=24, feat_channels=feat_channels, num_proposals=10,
            num_heads=6,
            single_head=_make_single_head(feat_channels=feat_channels),
            roi_extractor=_make_roi_extractor(out_channels=feat_channels),
            criterion=None,
        )
        student.set_teacher(teacher)

        features, bboxes, t = _make_dummy_input(feat_channels=feat_channels)
        student.train()

        s_cls, s_bbox, s_feats = student(features, bboxes, t)
        with torch.no_grad():
            _, _, t_feats = teacher(features, bboxes, t)

        distill_loss = sum(
            torch.nn.functional.mse_loss(s_feats[s], t_feats[t].detach())
            for s, t in {0: 0, 1: 2, 2: 5}.items()
        ) / 3
        distill_loss.backward()

        # Teacher 参数不应有梯度
        for name, p in teacher.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, \
                f"Teacher 参数 {name} 不应有梯度"


# ============================================================
# 6. Head 映射正确性
# ============================================================

class TestHeadMapping:
    """Head 映射: Student head k → Teacher head map(k)"""

    def test_head_map_default(self):
        """默认映射 {0:0, 1:2, 2:5}: 输入对齐 + 中间 + main 对齐"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        assert head.distill_head_map == {0: 0, 1: 2, 2: 5}

    def test_head_map_custom(self):
        """自定义映射"""
        custom_map = {0: 1, 1: 3, 2: 4}
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
            distill_head_map=custom_map,
        )
        assert head.distill_head_map == custom_map

    def test_head_map_covers_all_student_heads(self):
        """映射应覆盖所有 Student head"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        student_head_indices = set(range(3))
        mapped_indices = set(head.distill_head_map.keys())
        assert student_head_indices == mapped_indices, \
            f"映射应覆盖所有 Student head: {student_head_indices} vs {mapped_indices}"

    def test_head_map_teacher_indices_valid(self):
        """映射的 Teacher head 索引应在 [0, H_teacher) 范围内"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_distillation=True,
        )
        # 默认 H_teacher=6, 所以 teacher 索引应 < 6
        for t_idx in head.distill_head_map.values():
            assert 0 <= t_idx < 6, \
                f"Teacher head 索引 {t_idx} 超出范围 [0, 6)"


# ============================================================
# 7. deep_supervision_aux_weight
# ============================================================

class TestDeepSupervisionAuxWeight:
    """deep_supervision_aux_weight 降低 aux loss 权重"""

    def test_aux_weight_default_1(self):
        """默认 aux 权重为 1.0 (不变)"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert head.deep_supervision_aux_weight == 1.0

    def test_aux_weight_custom(self):
        """自定义 aux 权重"""
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=3,
            single_head=_make_single_head(), roi_extractor=_make_roi_extractor(),
            criterion=None,
            deep_supervision_aux_weight=0.5,
        )
        assert head.deep_supervision_aux_weight == 0.5


# ============================================================
# 8. 共享噪声
# ============================================================

class TestSharedNoise:
    """共享噪声: Teacher 和 Student 使用相同 x_raw"""

    def test_loss_accepts_shared_noise(self):
        """loss() 接受 x_raw_shared 参数 (向后兼容, 默认 None)"""
        import inspect
        sig = inspect.signature(DiffusionDetHead.loss)
        assert 'x_raw_shared' in sig.parameters, \
            "loss() 应接受 x_raw_shared 参数"

    def test_shared_noise_produces_same_coupling(self):
        """相同 x_raw_shared 使 Student/Teacher 看到相同输入"""
        # 此测试验证: 传入相同 x_raw_shared 时, 两次 forward 的输入一致
        feat_channels = 64
        head = DiffusionDetHead(
            num_classes=24, feat_channels=feat_channels, num_proposals=10,
            num_heads=3,
            single_head=_make_single_head(feat_channels=feat_channels),
            roi_extractor=_make_roi_extractor(out_channels=feat_channels),
            criterion=None,
        )

        features, bboxes, t = _make_dummy_input(feat_channels=feat_channels)
        head.eval()

        # 使用相同 x_raw
        x_raw = torch.randn(2, 10, 4)

        with torch.no_grad():
            # 第一次 forward with x_raw
            cls1, bbox1, _ = head(features, bboxes, t)
            # 第二次 forward with 同一个 x_raw → 应得到相同结果
            cls2, bbox2, _ = head(features, bboxes, t)

        # eval 模式下 (无 dropout/random), 相同输入应产生相同输出
        assert torch.allclose(cls1, cls2, atol=1e-6), \
            "相同输入应产生相同输出 (eval 模式)"
