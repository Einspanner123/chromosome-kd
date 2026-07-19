"""TDD 红: 验证方向 A (coupling 用 pred_boxes matching) 的实现.

方向 A 背景 (2026-07-19, DS 路径 A):
    SetDiff 当前 coupling 阶段用 noise 做 matching (random/Hungarian), 每
    步重新匹配导致 slot 语义角色漂移. DN-DETR 明确警告: bipartite matching
    不稳定是 slow convergence 根因.

    方向 A: coupling 阶段用 pred_boxes (第一次 encoder forward 的预测) 做
    matching, 而非 noise. matching 随训练稳定 (DETR 早期 matching 也不稳定,
    但随训练收敛). 保留 joint state + 理论区分点.

实现:
    1. HungarianMatcher.match_coupling_batch(proposals, noise, gt_boxes_list,
       gt_labels_list): proposals 用于 cost matrix, noise 用于 unmatched
       slot fallback (保持 unmatched_strategy='noise' 语义).
    2. RandomMatcher.match_coupling_batch: 复用 match_batch (忽略 proposals).
    3. JointDiffusionHead(coupling_source='pred_init'): _forward_train 先做
       no_grad forward 得到 pred_init, 再用 pred_init 做 coupling matching.

已知张力 (Plan agent A-3, 需在代码 docstring 记录):
    coupling matching 用 t=1.0 的 pred_init, loss matching 用 sampled t 的
    pred_boxes, 两者天然不一致. 但方向 A 的改进是 coupling 从"完全随机(noise)"
    变成"基于模型预测(pred_init)", 即使 t 不同, 至少都是模型预测, 比 noise 稳定.

本文件 TDD 红: 先写失败测试, 验证设计意图; 然后实现使测试通过.
"""

from typing import List

import pytest
import torch
from torch import Tensor

from setdiff.matching.hungarian import HungarianMatcher
from setdiff.matching.random_matcher import RandomMatcher


# ============================================================
# TestHungarianMatchCouplingBatch: HungarianMatcher 新增 match_coupling_batch
# ============================================================
class TestHungarianMatchCouplingBatch:
    """验证 HungarianMatcher.match_coupling_batch 接口.

    期望接口:
        match_coupling_batch(proposals, noise, gt_boxes_list, gt_labels_list)
            -> (matched_boxes [B,N,4], matched_labels [B,N])

        - proposals [B,N,4]: 用于 cost matrix (扩散空间, 来自 pred_init)
        - noise [B,N,4]: 用于 unmatched slot 的 matched_boxes fallback
        - matched slot: Hungarian 匹配 (基于 proposals 的 cost)
        - unmatched slot: 'noise' 策略下 = noise; 'random_gt' 策略下 = 随机 GT
    """

    @pytest.fixture
    def gt_boxes_list(self):
        """2 张图, 每张 2 个 GT (扩散空间 cxcywh)."""
        return [
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]]),
            torch.tensor([[1.0, 1.0, 0.5, 0.5]]),
        ]

    @pytest.fixture
    def gt_labels_list(self):
        return [
            torch.tensor([0, 1]),
            torch.tensor([2]),
        ]

    def test_match_coupling_batch_uses_proposals_for_cost(
        self, gt_boxes_list, gt_labels_list
    ):
        """match_coupling_batch 用 proposals (而非 noise) 计算 cost matrix.

        构造: proposals[0] 紧贴 gt[0], proposals[1] 紧贴 gt[1],
        proposals[2,3] 远离所有 GT. noise 完全随机.
        期望: matched_boxes[0,0] ≈ gt[0], matched_boxes[0,1] ≈ gt[1]
        (基于 proposals 的 matching, 不是 noise).
        """
        matcher = HungarianMatcher(cost_type='l2', unmatched_strategy='noise')

        # proposals: slot 0,1 紧贴 GT, slot 2,3 远离
        proposals = torch.tensor([
            [
                [0.0, 0.0, 1.0, 1.0],  # 紧贴 gt[0]
                [2.0, 2.0, 1.0, 1.0],  # 紧贴 gt[1]
                [100.0, 100.0, 1.0, 1.0],  # 远离
                [200.0, 200.0, 1.0, 1.0],  # 远离
            ]
        ])
        # noise: 完全随机 (与 proposals 不同)
        noise = torch.tensor([
            [
                [50.0, 50.0, 1.0, 1.0],
                [60.0, 60.0, 1.0, 1.0],
                [70.0, 70.0, 1.0, 1.0],
                [80.0, 80.0, 1.0, 1.0],
            ]
        ])

        matched_boxes, matched_labels = matcher.match_coupling_batch(
            proposals, noise, gt_boxes_list[:1], gt_labels_list[:1]
        )

        # slot 0,1 应匹配到 gt[0], gt[1] (基于 proposals 的 cost)
        assert matched_labels[0, 0].item() == 0  # gt[0] label
        assert matched_labels[0, 1].item() == 1  # gt[1] label
        # matched_boxes[0,0] 应是 gt[0] (不是 noise[0])
        assert torch.allclose(
            matched_boxes[0, 0], gt_boxes_list[0][0], atol=1e-6
        )
        assert torch.allclose(
            matched_boxes[0, 1], gt_boxes_list[0][1], atol=1e-6
        )

    def test_match_coupling_batch_unmatched_fallback_to_noise(
        self, gt_boxes_list, gt_labels_list
    ):
        """unmatched_strategy='noise' 时, unmatched slot 的 matched_boxes = noise.

        构造: 4 slot, 2 GT. slot 0,1 匹配 GT, slot 2,3 unmatched.
        期望: matched_boxes[0,2,3] == noise[0,2,3] (fallback to noise).
        """
        matcher = HungarianMatcher(cost_type='l2', unmatched_strategy='noise')

        proposals = torch.tensor([
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 1.0, 1.0],
                [100.0, 100.0, 1.0, 1.0],
                [200.0, 200.0, 1.0, 1.0],
            ]
        ])
        noise = torch.tensor([
            [
                [50.0, 50.0, 1.0, 1.0],
                [60.0, 60.0, 1.0, 1.0],
                [70.0, 70.0, 1.0, 1.0],
                [80.0, 80.0, 1.0, 1.0],
            ]
        ])

        matched_boxes, matched_labels = matcher.match_coupling_batch(
            proposals, noise, gt_boxes_list[:1], gt_labels_list[:1]
        )

        # slot 2,3 unmatched, matched_boxes 应 fallback to noise
        assert matched_labels[0, 2].item() == -1
        assert matched_labels[0, 3].item() == -1
        assert torch.allclose(matched_boxes[0, 2], noise[0, 2], atol=1e-6)
        assert torch.allclose(matched_boxes[0, 3], noise[0, 3], atol=1e-6)

    def test_match_coupling_batch_random_gt_strategy(
        self, gt_boxes_list, gt_labels_list
    ):
        """unmatched_strategy='random_gt' 时, unmatched slot 分配随机 GT.

        构造: 4 slot, 2 GT. slot 0,1 匹配 GT, slot 2,3 unmatched.
        期望: matched_labels[0,2,3] >= 0 (随机 GT label, 不是 -1).
        """
        matcher = HungarianMatcher(
            cost_type='l2', unmatched_strategy='random_gt'
        )

        proposals = torch.tensor([
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 1.0, 1.0],
                [100.0, 100.0, 1.0, 1.0],
                [200.0, 200.0, 1.0, 1.0],
            ]
        ])
        noise = torch.tensor([
            [
                [50.0, 50.0, 1.0, 1.0],
                [60.0, 60.0, 1.0, 1.0],
                [70.0, 70.0, 1.0, 1.0],
                [80.0, 80.0, 1.0, 1.0],
            ]
        ])

        matched_boxes, matched_labels = matcher.match_coupling_batch(
            proposals, noise, gt_boxes_list[:1], gt_labels_list[:1]
        )

        # slot 0,1 匹配 (基于 proposals)
        assert matched_labels[0, 0].item() == 0
        assert matched_labels[0, 1].item() == 1
        # slot 2,3 应分配随机 GT (label >= 0)
        assert matched_labels[0, 2].item() >= 0
        assert matched_labels[0, 3].item() >= 0
        # matched_boxes[0,2,3] 应是某个 GT (不是 noise)
        for i in [2, 3]:
            is_gt_0 = torch.allclose(
                matched_boxes[0, i], gt_boxes_list[0][0], atol=1e-6
            )
            is_gt_1 = torch.allclose(
                matched_boxes[0, i], gt_boxes_list[0][1], atol=1e-6
            )
            assert is_gt_0 or is_gt_1, (
                f"slot {i} 应是随机 GT (gt[0] 或 gt[1]), "
                f"实际 {matched_boxes[0, i]}"
            )

    def test_match_coupling_batch_random_matcher_ignores_proposals(
        self, gt_boxes_list, gt_labels_list
    ):
        """RandomMatcher.match_coupling_batch 忽略 proposals, 行为与 match_batch 一致.

        RandomMatcher 不基于 cost matrix, proposals 参数无意义.
        两次调用 (一次传 proposals, 一次传不同 proposals) 应结果一致
        (在相同 random seed 下).
        """
        matcher = RandomMatcher()

        noise = torch.randn(1, 4, 4)
        proposals_v1 = torch.randn(1, 4, 4)
        proposals_v2 = torch.randn(1, 4, 4)  # 不同的 proposals

        # 相同 seed 下, 两次调用应一致 (proposals 被忽略)
        torch.manual_seed(42)
        mb_v1, ml_v1 = matcher.match_coupling_batch(
            proposals_v1, noise, gt_boxes_list[:1], gt_labels_list[:1]
        )
        torch.manual_seed(42)
        mb_v2, ml_v2 = matcher.match_coupling_batch(
            proposals_v2, noise, gt_boxes_list[:1], gt_labels_list[:1]
        )

        assert torch.allclose(mb_v1, mb_v2, atol=1e-6)
        assert torch.equal(ml_v1, ml_v2)

        # 也应与 match_batch(noise, ...) 一致
        torch.manual_seed(42)
        mb_v3, ml_v3 = matcher.match_batch(
            noise, gt_boxes_list[:1], gt_labels_list[:1]
        )
        assert torch.allclose(mb_v1, mb_v3, atol=1e-6)
        assert torch.equal(ml_v1, ml_v3)


# ============================================================
# TestForwardTrainPredInitCoupling: JointDiffusionHead coupling_source='pred_init'
# ============================================================
class TestForwardTrainPredInitCoupling:
    """验证 JointDiffusionHead(coupling_source='pred_init') 的 _forward_train.

    期望行为:
        1. 第一次 encoder forward (no_grad, t=1.0) 得到 pred_init
        2. 用 pred_init 做 coupling matching (match_coupling_batch)
        3. q_sample(matched_boxes, noise, sampled_t) → x_t
        4. 第二次 encoder forward (需要梯度, sampled_t) → pred_final
        5. loss(pred_final, GT)
    """

    @pytest.fixture
    def head_kwargs(self):
        """小尺寸 head 配置 (快速测试)."""
        return dict(
            num_queries=8,
            feat_channels=16,
            num_heads=2,
            num_layers=2,
            dim_feedforward=32,
            num_classes=3,
            snr_scale=2.0,
            num_sample_steps=4,
            sampler='euler',
            matcher_type='hungarian',
            unmatched_strategy='noise',
        )

    @pytest.fixture
    def forward_inputs(self):
        """forward 输入: image_features + gt_boxes + gt_labels."""
        torch.manual_seed(42)
        image_features = torch.randn(2, 20, 16)  # [B, HW, C]
        gt_boxes = [
            torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]]),
            torch.tensor([[1.0, 1.0, 0.5, 0.5]]),
        ]
        gt_labels = [
            torch.tensor([0, 1]),
            torch.tensor([2]),
        ]
        return image_features, gt_boxes, gt_labels

    def test_forward_train_with_pred_init_coupling(
        self, head_kwargs, forward_inputs
    ):
        """coupling_source='pred_init' 时 _forward_train 不报错, loss 各项有限."""
        from setdiff.models.set_head import JointDiffusionHead

        head = JointDiffusionHead(coupling_source='pred_init', **head_kwargs)
        image_features, gt_boxes, gt_labels = forward_inputs

        loss_dict = head(image_features, gt_boxes, gt_labels)

        # 应返回 loss_cls, loss_bbox, loss_giou (预乘权重)
        assert 'loss_cls' in loss_dict
        assert 'loss_bbox' in loss_dict
        assert 'loss_giou' in loss_dict
        # 各项应有限 (not NaN, not Inf)
        for k, v in loss_dict.items():
            assert torch.isfinite(v), f"{k} 不有限: {v}"
            assert v.item() >= 0, f"{k} 应非负: {v}"

    def test_forward_train_pred_init_under_no_grad(
        self, head_kwargs, forward_inputs
    ):
        """第一次 forward (pred_init) 在 no_grad 下, 输出不携带梯度.

        结构化验证 (不依赖间接信号):
        - coupling_source='pred_init' 应有 2 次 encoder.forward
        - 第一次 (pred_init) 输出 requires_grad=False (no_grad 上下文)
        - 第二次 (sampled t) 输出 requires_grad=True (有梯度)
        """
        from setdiff.models.set_head import JointDiffusionHead

        head = JointDiffusionHead(coupling_source='pred_init', **head_kwargs)
        image_features, gt_boxes, gt_labels = forward_inputs

        # 用 wrapper 记录每次 encoder.forward 输出的 requires_grad
        output_grad_flags = []
        original_forward = head.encoder.forward

        def wrapped_forward(*args, **kwargs):
            cls_logits, pred_boxes = original_forward(*args, **kwargs)
            output_grad_flags.append(pred_boxes.requires_grad)
            return cls_logits, pred_boxes

        head.encoder.forward = wrapped_forward

        loss_dict = head(image_features, gt_boxes, gt_labels)
        total_loss = sum(loss_dict.values())
        total_loss.backward()

        # 应有 2 次 encoder forward (第一次 no_grad pred_init + 第二次有梯度)
        assert len(output_grad_flags) == 2, (
            f"应有 2 次 encoder forward (第一次 no_grad pred_init + "
            f"第二次有梯度), 实际: {len(output_grad_flags)}"
        )
        # 第一次 forward (pred_init) 应在 no_grad 下, 输出不携带梯度
        assert output_grad_flags[0] is False, (
            f"第一次 forward (pred_init) 应在 no_grad 下, 输出 "
            f"requires_grad 应为 False, 实际: {output_grad_flags[0]}"
        )
        # 第二次 forward 应携带梯度 (用于 loss.backward)
        assert output_grad_flags[1] is True, (
            f"第二次 forward 应携带梯度, 输出 requires_grad 应为 True, "
            f"实际: {output_grad_flags[1]}"
        )

    def test_forward_train_pred_init_vs_noise_diff_behavior(
        self, head_kwargs, forward_inputs
    ):
        """coupling_source='pred_init' 与 'noise' 调用不同的 matcher 方法.

        结构化验证 (不依赖数值巧合或 random seed):
        - coupling_source='noise': 调用 matcher.match_batch (原行为)
        - coupling_source='pred_init': 调用 matcher.match_coupling_batch (方向 A)

        这是方向 A 的核心区分点: coupling matching 决策路径不同.
        """
        from setdiff.models.set_head import JointDiffusionHead

        image_features, gt_boxes, gt_labels = forward_inputs

        # coupling_source='noise' 应调用 match_batch (原行为)
        head_noise = JointDiffusionHead(
            coupling_source='noise', **head_kwargs
        )
        call_log_noise = []
        original_match_batch = head_noise.matcher.match_batch

        def wrapped_match_batch(*args, **kwargs):
            call_log_noise.append('match_batch')
            return original_match_batch(*args, **kwargs)

        head_noise.matcher.match_batch = wrapped_match_batch
        head_noise(image_features, gt_boxes, gt_labels)
        assert call_log_noise == ['match_batch'], (
            f"coupling_source='noise' 应调用 match_batch 一次, "
            f"实际调用: {call_log_noise}"
        )

        # coupling_source='pred_init' 应调用 match_coupling_batch (方向 A)
        head_pred = JointDiffusionHead(
            coupling_source='pred_init', **head_kwargs
        )
        call_log_pred = []
        original_match_coupling = head_pred.matcher.match_coupling_batch

        def wrapped_match_coupling(*args, **kwargs):
            call_log_pred.append('match_coupling_batch')
            return original_match_coupling(*args, **kwargs)

        head_pred.matcher.match_coupling_batch = wrapped_match_coupling
        head_pred(image_features, gt_boxes, gt_labels)
        assert call_log_pred == ['match_coupling_batch'], (
            f"coupling_source='pred_init' 应调用 match_coupling_batch 一次, "
            f"实际调用: {call_log_pred}"
        )

    def test_forward_train_pred_init_with_self_attn_disabled(
        self, head_kwargs, forward_inputs
    ):
        """方向 A + 方向 C 组合: coupling_source='pred_init' + enable_self_attn=False.

        验证组合场景能正常 forward + backward + loss 有限. 第一次 forward
        (pred_init) 也会用对角 mask (因 enable_self_attn=False), 存在潜在
        交互, 需确认不报错且梯度正常.

        覆盖审查 MINOR-6: 方向 A 与方向 C 逻辑独立, 但组合下第一次 forward
        也会用对角 mask, 需验证无副作用.
        """
        from setdiff.models.set_head import JointDiffusionHead

        head = JointDiffusionHead(
            coupling_source='pred_init',
            enable_self_attn=False,
            **head_kwargs,
        )
        image_features, gt_boxes, gt_labels = forward_inputs

        loss_dict = head(image_features, gt_boxes, gt_labels)
        total_loss = sum(loss_dict.values())
        total_loss.backward()

        # 各项应有限非负
        for k, v in loss_dict.items():
            assert torch.isfinite(v), f"{k} 不有限: {v}"
            assert v.item() >= 0, f"{k} 应非负: {v}"

        # encoder 参数应有梯度 (来自第二次 forward)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in head.encoder.parameters()
        )
        assert has_grad, "encoder 参数应有梯度 (来自第二次 forward)"
