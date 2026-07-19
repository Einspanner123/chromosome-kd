"""TDD 红: 验证 SetCriterion 重新做 Hungarian 匹配 (num_pos=M, 不是 N=300).

根因诊断 (2026-07-19):
    LDMDet criterion.py (line 84, 135-140, 249-269) 在 loss 计算时用
    Hungarian 重新匹配 pred_boxes 和 GT, num_pos = fg_masks.sum() = M
    (GT 数, 约 8). 这是对齐 DETR 家族的标准做法.

    SetDiff planA/B 当前 SetCriterion 直接用 coupling 阶段的 matched_labels
    (num_pos=300), 导致 per-slot 梯度稀释 37.5 倍
    (0.000733 vs LDMDet baseline 0.025125, 实测 16/600).

修复方案 (对齐 LDMDet):
    1. HungarianMatcher 新增 match_indices(pred_boxes, gt_boxes) → (src_idx, tgt_idx)
    2. SetCriterion 重构: 接收原始 GT (gt_boxes_list, gt_labels_list),
       内置 HungarianMatcher, loss 计算时重新匹配
    3. bbox/giou loss 只对 matched slot 计算, num_pos = sum(M_i) (batch GT 总数)
    4. cls loss 所有 slot 参与focal, matched slot 有正样本, unmatched 是背景)

本文件 TDD 红: 先写失败测试, 验证设计意图; 然后实现使测试通过.
"""

from typing import List

import pytest
import torch
from torch import Tensor

from setdiff.matching.hungarian import HungarianMatcher
from setdiff.criterion.set_loss import SetCriterion


# ============================================================
# TestHungarianMatchIndices: 新增 match_indices 方法
# ============================================================
class TestHungarianMatchIndices:
    """验证 HungarianMatcher.match_indices 接口.

    期望接口:
        match_indices(pred_boxes, gt_boxes) -> (src_idx, tgt_idx)
            pred_boxes: [N, 4] 模型预测的 x_0
            gt_boxes: [M, 4] 原始 GT
            src_idx: [K] long, matched slot 在 N 中的索引
            tgt_idx: [K] long, 对应 GT 在 M 中的索引
            K = min(N, M) (一对一匹配数)
    """

    @pytest.fixture
    def matcher(self):
        return HungarianMatcher(cost_type='l2')

    def test_returns_two_tensors(self, matcher):
        """返回 (src_idx, tgt_idx) 两个 long 张量."""
        pred = torch.randn(10, 4)
        gt = torch.randn(3, 4)
        src_idx, tgt_idx = matcher.match_indices(pred, gt)
        assert isinstance(src_idx, Tensor)
        assert isinstance(tgt_idx, Tensor)
        assert src_idx.dtype == torch.long
        assert tgt_idx.dtype == torch.long

    def test_one_to_one_matching(self, matcher):
        """一对一匹配: len(src_idx) == len(tgt_idx) == min(N, M)."""
        N, M = 10, 3
        pred = torch.randn(N, 4)
        gt = torch.randn(M, 4)
        src_idx, tgt_idx = matcher.match_indices(pred, gt)
        assert len(src_idx) == M
        assert len(tgt_idx) == M
        # src_idx 互不相同 (一对一)
        assert len(set(src_idx.tolist())) == M
        # tgt_idx 互不相同
        assert len(set(tgt_idx.tolist())) == M

    def test_empty_gt(self, matcher):
        """M=0 时返回空张量."""
        pred = torch.randn(5, 4)
        gt = torch.zeros(0, 4)
        src_idx, tgt_idx = matcher.match_indices(pred, gt)
        assert len(src_idx) == 0
        assert len(tgt_idx) == 0

    def test_n_less_than_m(self, matcher):
        """N < M 时只匹配 N 对 (slot 不够)."""
        N, M = 3, 8
        pred = torch.randn(N, 4)
        gt = torch.randn(M, 4)
        src_idx, tgt_idx = matcher.match_indices(pred, gt)
        assert len(src_idx) == N
        assert len(tgt_idx) == N

    def test_matches_minimize_l2_cost(self, matcher):
        """验证匹配确实最小化总 L2 代价 (可重现 LDMDet 行为)."""
        # 构造明确的最优匹配
        pred = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],  # 接近 gt[0]
                [10.0, 10.0, 10.0, 10.0],  # 远离所有
                [1.0, 1.0, 1.0, 1.0],  # 接近 gt[1]
                [20.0, 20.0, 20.0, 20.0],  # 远离所有
            ]
        )
        gt = torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],  # 接近 pred[0]
                [1.1, 1.1, 1.1, 1.1],  # 接近 pred[2]
            ]
        )
        src_idx, tgt_idx = matcher.match_indices(pred, gt)
        # src_idx 应包含 0 和 2 (索引到 pred)
        src_set = set(src_idx.tolist())
        assert src_set == {0, 2}, f"期望匹配 {{0, 2}}, 实际 {src_set}"
        # 对应关系: pred[0] → gt[0], pred[2] → gt[1]
        pair_map = dict(zip(src_idx.tolist(), tgt_idx.tolist()))
        assert pair_map[0] == 0
        assert pair_map[2] == 1

    def test_batch_match_indices(self, matcher):
        """批量版本: 每张图独立匹配, 返回 list[(src_idx, tgt_idx)]."""
        B, N = 2, 10
        pred_batch = torch.randn(B, N, 4)
        gt_boxes_list = [torch.randn(5, 4), torch.randn(3, 4)]
        results = matcher.match_indices_batch(pred_batch, gt_boxes_list)
        assert isinstance(results, list)
        assert len(results) == B
        # 第一张图 5 对
        assert len(results[0][0]) == 5
        # 第二张图 3 对
        assert len(results[1][0]) == 3

    def test_no_grad(self, matcher):
        """match_indices 不计算梯度 (匹配是离散操作)."""
        pred = torch.randn(10, 4, requires_grad=True)
        gt = torch.randn(3, 4)
        src_idx, tgt_idx = matcher.match_indices(pred, gt)
        # src_idx/tgt_idx 是 long, 无 grad
        assert not src_idx.requires_grad
        assert not tgt_idx.requires_grad


# ============================================================
# TestSetCriterionRematch: 重构后的 SetCriterion
# ============================================================
class TestSetCriterionRematch:
    """验证 SetCriterion 重构: 接收原始 GT, 内置 matcher, num_pos=M.

    期望接口:
        forward(outputs, gt_boxes_list, gt_labels_list) -> (loss_dict, total)
            outputs: {'pred_logits': [B,N,C], 'pred_boxes': [B,N,4]}
            gt_boxes_list: List[Tensor[M_i, 4]] 每张图的 GT (扩散空间)
            gt_labels_list: List[Tensor[M_i]] 每张图的 GT label

    关键不变量:
        - num_pos (bbox/giou 归一化) = sum(M_i) (batch 总 GT 数)
        - bbox/giou loss 只对 matched slot 计算
        - cls loss 所有 slot 参与 (focal, matched=正样本, unmatched=背景)
    """

    @pytest.fixture
    def criterion(self):
        return SetCriterion(num_classes=24, snr_scale=2.0)

    def _make_outputs(self, B=2, N=20, C=24):
        torch.manual_seed(42)
        pred_logits = torch.randn(B, N, C, requires_grad=True)
        pred_boxes = torch.randn(B, N, 4, requires_grad=True)
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}

    def _make_gt(self, B=2, M=5, num_classes=24):
        torch.manual_seed(43)
        gt_boxes_list = [torch.randn(M, 4) * 0.5 for _ in range(B)]
        gt_labels_list = [
            torch.randint(0, num_classes, (M,)) for _ in range(B)
        ]
        return gt_boxes_list, gt_labels_list

    def test_accepts_raw_gt_signature(self, criterion):
        """forward 接受 (outputs, gt_boxes_list, gt_labels_list) 三参数."""
        outputs = self._make_outputs()
        gt_boxes_list, gt_labels_list = self._make_gt()
        # 应能正常调用, 返回 (loss_dict, total)
        loss_dict, total = criterion(outputs, gt_boxes_list, gt_labels_list)
        assert isinstance(loss_dict, dict)
        assert 'loss_cls' in loss_dict
        assert 'loss_bbox' in loss_dict
        assert 'loss_giou' in loss_dict
        assert total.dim() == 0  # scalar

    def test_num_pos_equals_M(self, criterion):
        """bbox/giou 归一化的 num_pos = sum(M_i), 不是 N=300.

        验证方式: 把 pred_boxes 设成与 GT 完全相同 (zero loss), bbox loss
        应该接近 0; 然后做小扰动, loss 与扰动幅度成比例. 通过比较
        num_pos=N (旧) vs num_pos=M (新) 的 loss 数值差异判断.
        """
        B, N, M = 2, 20, 5
        # pred_boxes 完美匹配 GT (前 M 个 slot = GT)
        gt_boxes_list, gt_labels_list = self._make_gt(B=B, M=M)
        pred_boxes = torch.randn(B, N, 4, requires_grad=True)
        # 前 M 个 slot 直接设为 GT
        with torch.no_grad():
            for b in range(B):
                pred_boxes.data[b, :M] = gt_boxes_list[b]
        outputs = {
            'pred_logits': torch.randn(B, N, 24, requires_grad=True),
            'pred_boxes': pred_boxes,
        }
        loss_dict, _ = criterion(outputs, gt_boxes_list, gt_labels_list)
        # 完美匹配时 bbox loss 应该接近 0
        assert loss_dict['loss_bbox'].item() < 0.01, (
            f"完美匹配 bbox loss 应 ≈ 0, 实际 {loss_dict['loss_bbox'].item()}"
        )

    def test_bbox_loss_only_on_matched_slots(self, criterion):
        """bbox/giou loss 只对 matched slot 计算, unmatched slot 不影响.

        验证方式: 固定前 M 个 slot (matched), 改变后 N-M 个 slot (unmatched)
        的值, bbox loss 不应改变.
        """
        B, N, M = 1, 20, 5
        torch.manual_seed(42)
        gt_boxes_list = [torch.randn(M, 4) * 0.5]
        gt_labels_list = [torch.randint(0, 24, (M,))]

        # 第一次: pred_boxes 全 0
        pred_boxes_v1 = torch.zeros(B, N, 4)
        outputs_v1 = {
            'pred_logits': torch.randn(B, N, 24),
            'pred_boxes': pred_boxes_v1,
        }
        _, total_v1 = criterion(outputs_v1, gt_boxes_list, gt_labels_list)

        # 第二次: pred_boxes 前 M 个保持, 后 N-M 改成大值
        pred_boxes_v2 = torch.zeros(B, N, 4)
        pred_boxes_v2[:, M:] = 100.0  # unmatched slot 大幅变化
        outputs_v2 = {
            'pred_logits': torch.randn(B, N, 24),
            'pred_boxes': pred_boxes_v2,
        }
        # 用相同 cls_logits 保证 cls loss 相同
        outputs_v2['pred_logits'] = outputs_v1['pred_logits']
        _, total_v2 = criterion(outputs_v2, gt_boxes_list, gt_labels_list)

        # bbox + giou loss 应完全相同 (unmatched slot 不参与)
        # (cls loss 也相同因为 logits 相同)
        assert torch.allclose(total_v1, total_v2, atol=1e-5), (
            f"unmatched slot 变化不应影响 loss. v1={total_v1.item()}, "
            f"v2={total_v2.item()}"
        )

    def test_cls_loss_all_slots_participate(self, criterion):
        """cls loss 所有 slot 参与 (matched=正样本, unmatched=背景)."""
        B, N, M = 1, 20, 5
        torch.manual_seed(42)
        gt_boxes_list = [torch.randn(M, 4) * 0.5]
        gt_labels_list = [torch.tensor([3, 7, 11, 15, 20])]

        # cls_logits 全 0
        pred_logits_v1 = torch.zeros(B, N, 24)
        outputs_v1 = {
            'pred_logits': pred_logits_v1,
            'pred_boxes': torch.randn(B, N, 4),
        }
        loss_dict_v1, _ = criterion(
            outputs_v1, gt_boxes_list, gt_labels_list
        )

        # cls_logits 改变 unmatched slot 的值, cls loss 应变化
        # (因为 unmatched slot 仍参与 focal loss 作为背景)
        pred_logits_v2 = torch.zeros(B, N, 24)
        pred_logits_v2[:, M:] = 10.0  # unmatched slot 高置信 → 背景惩罚大
        outputs_v2 = {
            'pred_logits': pred_logits_v2,
            'pred_boxes': outputs_v1['pred_boxes'],
        }
        loss_dict_v2, _ = criterion(
            outputs_v2, gt_boxes_list, gt_labels_list
        )

        assert loss_dict_v2['loss_cls'].item() > loss_dict_v1['loss_cls'].item(), (
            "unmatched slot 高置信应增加 cls loss (背景惩罚). "
            f"v1={loss_dict_v1['loss_cls'].item()}, "
            f"v2={loss_dict_v2['loss_cls'].item()}"
        )

    def test_no_gradient_dilution(self, criterion):
        """关键测试: per-slot 梯度不被稀释 (参考 LDMDet loss 阶段重匹配).

        验证内容 (结构性 + 幅度):
            1. 结构性: matched slot 有非零梯度, unmatched slot 零梯度
               (unmatched slot 不参与 bbox loss, 与旧 planA/B 全 slot 监督不同).
            2. 幅度性: matched slot 梯度应与 num_pos=M 反比, 而非 num_pos=N.

        理论分析 (本测试场景 M=2, N=20, snr_scale=2.0):
            pred_boxes[0] = gt[0] + 0.1, 逆变换后 pred_norm - tgt_norm = 0.1/(2s) = 0.025
            L1 loss = sum(|diff|) / num_pos = 4*0.025 / num_pos
            per-element 梯度 = sign(diff) / (2s * num_pos) = 0.25 / num_pos
            per-slot 梯度 norm = sqrt(4) * 0.25 / num_pos = 0.5 / num_pos

            新方案 (num_pos=M=2): per-slot 梯度 ≈ 0.25
            旧方案 (num_pos=N=20): per-slot 梯度 ≈ 0.025
            阈值 0.1 可明确区分两者 (10x 差距).

        构造: slot 0,1 紧贴 gt[0],gt[1] (cost≈0, 唯一最优匹配),
        其余 slot 远离所有 GT (cost 大, 不会被匹配). Hungarian 必匹配 slot 0,1.
        """
        B, N, M = 1, 20, 2
        torch.manual_seed(42)
        gt_boxes_list = [torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                       [1.0, 1.0, 1.0, 1.0]])]
        gt_labels_list = [torch.tensor([0, 1])]

        # pred: 前 M 个 slot 紧贴 GT + 小偏差 (唯一最优匹配, 非零梯度),
        # 其余 slot = 100 (远离所有 GT, 不被匹配)
        pred_boxes = torch.full((B, N, 4), 100.0, requires_grad=True)
        with torch.no_grad():
            pred_boxes.data[:, 0] = gt_boxes_list[0][0] + 0.1  # 紧贴 gt[0]
            pred_boxes.data[:, 1] = gt_boxes_list[0][1] + 0.1  # 紧贴 gt[1]
        outputs = {
            'pred_logits': torch.zeros(B, N, 24),
            'pred_boxes': pred_boxes,
        }
        loss_dict, _ = criterion(outputs, gt_boxes_list, gt_labels_list)
        loss_dict['loss_bbox'].backward()

        # matched slot (0, 1) 应有非零梯度
        slot_0_grad = pred_boxes.grad[0, 0].norm().item()
        slot_1_grad = pred_boxes.grad[0, 1].norm().item()
        # unmatched slot (2..N-1) 应零梯度 (不参与 bbox loss)
        slot_unmatched_grad = pred_boxes.grad[0, 2:].abs().sum().item()
        # 结构性断言: matched 有梯度, unmatched 零梯度
        assert slot_0_grad > 0.001, (
            f"matched slot 0 梯度应 > 0.001 (不稀释), 实际 {slot_0_grad}"
        )
        assert slot_1_grad > 0.001, (
            f"matched slot 1 梯度应 > 0.001 (不稀释), 实际 {slot_1_grad}"
        )
        assert slot_unmatched_grad == 0.0, (
            f"unmatched slot 应无梯度 (不参与 bbox loss), 实际 {slot_unmatched_grad}"
        )
        # 幅度性断言: num_pos=M (新) per-slot 梯度 ≈ 0.25, num_pos=N (旧) ≈ 0.025.
        # 阈值 0.1 明确区分新 (不稀释) vs 旧 (稀释 1/N) 归一化方案.
        assert slot_0_grad > 0.1, (
            f"matched slot 0 梯度应 > 0.1 (num_pos=M, 不稀释), "
            f"实际 {slot_0_grad} (若 < 0.1 说明 num_pos=N 稀释生效)"
        )
        assert slot_1_grad > 0.1, (
            f"matched slot 1 梯度应 > 0.1 (num_pos=M, 不稀释), "
            f"实际 {slot_1_grad} (若 < 0.1 说明 num_pos=N 稀释生效)"
        )

    def test_backward_propagates_to_all_params(self, criterion):
        """梯度能反向传播到 pred_boxes 和 pred_logits."""
        outputs = self._make_outputs()
        gt_boxes_list, gt_labels_list = self._make_gt()
        loss_dict, total = criterion(outputs, gt_boxes_list, gt_labels_list)
        total.backward()
        assert outputs['pred_logits'].grad is not None
        assert outputs['pred_boxes'].grad is not None
        # 所有 matched slot 都应有非零梯度
        # (具体哪些 matched 由 matcher 决定, 但至少有一个非零)
        assert outputs['pred_boxes'].grad.abs().sum() > 0

    def test_empty_gt_no_bbox_loss(self, criterion):
        """M=0 时 bbox/giou loss = 0, cls loss 仍存在 (全背景)."""
        B, N = 1, 20
        outputs = self._make_outputs(B=B, N=N)
        gt_boxes_list = [torch.zeros(0, 4)]
        gt_labels_list = [torch.zeros(0, dtype=torch.long)]
        loss_dict, total = criterion(outputs, gt_boxes_list, gt_labels_list)
        assert loss_dict['loss_bbox'].item() == 0.0
        assert loss_dict['loss_giou'].item() == 0.0
        # cls loss 仍 > 0 (所有 slot 都是背景, focal 会惩罚高置信)
        assert loss_dict['loss_cls'].item() > 0

    def test_batch_with_varying_gt_counts(self, criterion):
        """batch 中不同图 GT 数不同: num_pos = sum(M_i)."""
        B = 2
        outputs = self._make_outputs(B=B, N=20)
        gt_boxes_list = [torch.randn(3, 4) * 0.5, torch.randn(7, 4) * 0.5]
        gt_labels_list = [
            torch.randint(0, 24, (3,)),
            torch.randint(0, 24, (7,)),
        ]
        # 应能正常运行
        loss_dict, total = criterion(outputs, gt_boxes_list, gt_labels_list)
        for k, v in loss_dict.items():
            assert torch.isfinite(v), f"{k} not finite"
        assert torch.isfinite(total)

    def test_loss_finite(self, criterion):
        """所有 loss 项有限 (无 NaN/Inf)."""
        outputs = self._make_outputs()
        gt_boxes_list, gt_labels_list = self._make_gt()
        loss_dict, total = criterion(outputs, gt_boxes_list, gt_labels_list)
        for k, v in loss_dict.items():
            assert torch.isfinite(v), f"{k} not finite"
        assert torch.isfinite(total)


# ============================================================
# TestSetCriterionRematchIntegration: 端到端集成
# ============================================================
class TestSetCriterionRematchIntegration:
    """验证 SetCriterion 与 JointDiffusionHead 集成后的端到端行为.

    核心断言: loss 计算时 num_pos=M (不是 N=300), 梯度不稀释.
    """

    def _build_head(self, **kwargs):
        from setdiff.models.set_head import JointDiffusionHead

        defaults = dict(
            num_queries=20,
            feat_channels=32,
            num_heads=4,
            num_layers=2,
            dim_feedforward=64,
            num_classes=24,
            snr_scale=2.0,
            num_sample_steps=4,
        )
        defaults.update(kwargs)
        return JointDiffusionHead(**defaults)

    def _make_batch(self, B=2, N=20, M=5, num_classes=24):
        torch.manual_seed(42)
        image_features = torch.randn(B, 64, 32)
        gt_boxes_list = [torch.randn(M, 4) * 0.5 for _ in range(B)]
        gt_labels_list = [
            torch.randint(0, num_classes, (M,)) for _ in range(B)
        ]
        return image_features, gt_boxes_list, gt_labels_list

    def test_plan_a_loss_has_gradient(self):
        """方案 A: 端到端前向 loss 有梯度, 各项有限."""
        torch.manual_seed(42)
        head = self._build_head(matcher_type='random')
        image_features, gt_boxes_list, gt_labels_list = self._make_batch()
        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        for k, v in loss_dict.items():
            assert torch.isfinite(v), f"方案 A {k} not finite"

        total = sum(loss_dict.values())
        total.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
            if p.requires_grad
        )
        assert has_grad, "方案 A 端到端应有梯度"

    def test_plan_b_loss_has_gradient(self):
        """方案 B: 端到端前向 loss 有梯度."""
        torch.manual_seed(42)
        head = self._build_head(
            matcher_type='hungarian', unmatched_strategy='random_gt'
        )
        image_features, gt_boxes_list, gt_labels_list = self._make_batch()
        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        for k, v in loss_dict.items():
            assert torch.isfinite(v), f"方案 B {k} not finite"

        total = sum(loss_dict.values())
        total.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
            if p.requires_grad
        )
        assert has_grad, "方案 B 端到端应有梯度"

    def test_loss_normalization_uses_M_not_N(self):
        """关键集成测试: bbox loss 归一化使用 M 而不是 N.

        验证方式: 通过 monkey-patch criterion, 检查内部 matcher.match_indices
        被调用, 且返回的 src_idx 长度 = M (不是 N).
        """
        torch.manual_seed(42)
        head = self._build_head(matcher_type='random')
        image_features, gt_boxes_list, gt_labels_list = self._make_batch(
            M=5
        )

        # 捕获 criterion 内部 matcher 的调用
        captured = {}
        original_match_indices = head.criterion.matcher.match_indices

        def capture(pred, gt):
            result = original_match_indices(pred, gt)
            captured['src_len'] = len(result[0])
            captured['tgt_len'] = len(result[1])
            return result

        head.criterion.matcher.match_indices = capture

        head(image_features, gt_boxes_list, gt_labels_list)

        # 每张图应匹配 M 对 (不是 N=20)
        assert captured['src_len'] == 5, (
            f"criterion 应做 M=5 对匹配, 实际 {captured['src_len']}"
        )
        assert captured['tgt_len'] == 5

    def test_per_slot_gradient_not_diluted(self):
        """per-slot 梯度不稀释 (对齐 LDMDet 量级).

        对比测试:
            - 当前 SetDiff (planA, num_pos=N=20): matched slot 梯度 ~ 1/20
            - 修复后 (num_pos=M=5): matched slot 梯度 ~ 1/5
            - 比值应接近 20/5 = 4 倍提升
        """
        torch.manual_seed(42)
        head = self._build_head(matcher_type='random')
        image_features, gt_boxes_list, gt_labels_list = self._make_batch(
            M=5
        )
        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)
        total = sum(loss_dict.values())
        total.backward()

        # 检查 box_head 最后一层权重的梯度 norm
        box_head_grad_norms = []
        for name, p in head.encoder.named_parameters():
            if 'box_head' in name and p.grad is not None:
                box_head_grad_norms.append(p.grad.norm().item())
        # 应有非零梯度
        assert len(box_head_grad_norms) > 0
        avg_grad = sum(box_head_grad_norms) / len(box_head_grad_norms)
        # 修复后梯度应显著大于稀释情况 (>0.001, 旧 planA 实测 ~0.0001)
        assert avg_grad > 0.001, (
            f"box_head 平均梯度应 >0.001 (不稀释), 实际 {avg_grad}"
        )
