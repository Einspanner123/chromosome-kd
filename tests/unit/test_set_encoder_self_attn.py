"""TDD 红: 验证 SetEncoder 的 enable_self_attn 参数 (方向 C: per-proposal 退化).

方向 C 背景 (2026-07-19):
    SetDiff 训练 3 epoch 后 val mAP 仍为 0 (planA/planB 均为 0.000),
    梯度归一化修复后梯度幅度已恢复正常 (grad_norm=40-170), 但 mAP 不动.
    论文调研发现核心矛盾: joint state + self-attention 耦合 + 每步重匹配
    导致 slot 语义角色漂移. DN-DETR 明确警告 bipartite matching 不稳定是
    slow convergence 根因.

方向 C (诊断性消融 + 保底方案):
    禁用 self-attention (用对角 mask 阻断 slot 间交互), 每个 slot 独立处理,
    退化为 per-proposal 范式 (对齐 DiffusionDet). 若方向 C 成功 (mAP > 0),
    说明 self-attention 是 mAP=0 根因; 若失败, 说明根因在别处.

实现选择 (Plan agent C-Q1 验证):
    对角 mask ≠ 严格跳过 self-attn 层 (保留 value/out 投影, per-slot
    线性变换), 但满足 "slot 间无信息流动" 的核心需求. 跳过层方案需重写
    nn.TransformerDecoderLayer, 改动量大, mask 方案更简洁.

本文件 TDD 红: 先写失败测试, 验证设计意图; 然后实现使测试通过.
"""

import pytest
import torch

from setdiff.core.set_encoder import SetEncoder


# ============================================================
# TestEnableSelfAttn: SetEncoder 新增 enable_self_attn 参数
# ============================================================
class TestEnableSelfAttn:
    """验证 SetEncoder 的 enable_self_attn 参数.

    期望接口:
        SetEncoder(..., enable_self_attn: bool = True)
            - True (默认): 原行为, self-attention 允许 slot 间交互
            - False: 用对角 mask 阻断 slot 间交互, 每个 slot 只 attend to 自己
    """

    @pytest.fixture
    def encoder_kwargs(self):
        """小尺寸 encoder 配置 (快速测试)."""
        return dict(
            num_queries=4,
            feat_channels=16,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            num_classes=3,
        )

    @pytest.fixture
    def inputs(self):
        """固定输入: x_t [1, 4, 4], t_emb [1, 16], image_features [1, 10, 16]."""
        torch.manual_seed(42)
        x_t = torch.randn(1, 4, 4)
        t_emb = torch.randn(1, 16)
        image_features = torch.randn(1, 10, 16)
        return x_t, t_emb, image_features

    def test_enable_self_attn_false_blocks_inter_slot(
        self, encoder_kwargs, inputs
    ):
        """enable_self_attn=False 时, 改变 slot 0 不影响 slot 1-3 输出.

        核心验证: 对角 mask 阻断 slot 间信息流动.
        - 改变 slot 0 的 x_t → slot 0 的 pred_boxes 变 (正常, per-slot 处理)
        - 改变 slot 0 的 x_t → slot 1-3 的 pred_boxes 不变 (阻断 inter-slot)
        """
        x_t, t_emb, image_features = inputs
        encoder = SetEncoder(enable_self_attn=False, **encoder_kwargs)
        encoder.eval()  # 关闭 dropout, 确保两次 forward 唯一区别是输入

        with torch.no_grad():
            _, pred_v1 = encoder(x_t, t_emb, image_features)

            # 大幅改变 slot 0 的输入
            x_t_modified = x_t.clone()
            x_t_modified[0, 0] += 10.0
            _, pred_v2 = encoder(x_t_modified, t_emb, image_features)

        # slot 0 输出应变 (per-slot 处理仍保留 value/out 投影)
        slot_0_diff = (pred_v2[0, 0] - pred_v1[0, 0]).abs().max().item()
        assert slot_0_diff > 1e-6, (
            f"slot 0 输出应变 (输入改变), 实际 diff={slot_0_diff}"
        )

        # slot 1-3 输出不应变 (inter-slot 阻断)
        other_slots_diff = (
            (pred_v2[0, 1:] - pred_v1[0, 1:]).abs().max().item()
        )
        assert other_slots_diff < 1e-6, (
            f"slot 1-3 输出不应变 (enable_self_attn=False 阻断 inter-slot), "
            f"实际 diff={other_slots_diff}"
        )

    def test_enable_self_attn_true_inter_slot(
        self, encoder_kwargs, inputs
    ):
        """enable_self_attn=True (默认) 时, 改变 slot 0 影响 slot 1-3 输出.

        对照组: 确认默认行为允许 slot 间交互, 与方向 C 形成对比.
        """
        x_t, t_emb, image_features = inputs
        encoder = SetEncoder(enable_self_attn=True, **encoder_kwargs)
        encoder.eval()

        with torch.no_grad():
            _, pred_v1 = encoder(x_t, t_emb, image_features)

            x_t_modified = x_t.clone()
            x_t_modified[0, 0] += 10.0
            _, pred_v2 = encoder(x_t_modified, t_emb, image_features)

        # slot 1-3 输出应变 (self-attention 允许 inter-slot 交互)
        other_slots_diff = (
            (pred_v2[0, 1:] - pred_v1[0, 1:]).abs().max().item()
        )
        assert other_slots_diff > 1e-4, (
            f"slot 1-3 输出应变 (enable_self_attn=True 允许 inter-slot), "
            f"实际 diff={other_slots_diff}"
        )

    def test_enable_self_attn_false_active_in_train_mode(
        self, encoder_kwargs, inputs
    ):
        """enable_self_attn=False 时, 对角 mask 在 train 模式下也生效.

        核心验证: 对角 mask 不依赖 training mode (与 matched_mask 不同,
        matched_mask 仅在 self.training=True 时生效). train 模式下
        enable_self_attn=False 仍应阻断 inter-slot.
        """
        x_t, t_emb, image_features = inputs
        encoder = SetEncoder(enable_self_attn=False, **encoder_kwargs)

        # train 模式 (但用 no_grad + manual_seed 控制 dropout)
        encoder.train()
        torch.manual_seed(123)
        with torch.no_grad():
            _, pred_train = encoder(x_t, t_emb, image_features)

        # 改变 slot 0, 验证 slot 1-3 不变 (对角 mask 在 train 模式下也生效)
        x_t_modified = x_t.clone()
        x_t_modified[0, 0] += 10.0
        torch.manual_seed(123)
        with torch.no_grad():
            _, pred_train_modified = encoder(x_t_modified, t_emb, image_features)

        other_slots_diff = (
            (pred_train_modified[0, 1:] - pred_train[0, 1:]).abs().max().item()
        )
        assert other_slots_diff < 1e-6, (
            f"train 模式下 slot 1-3 仍应不受 slot 0 影响, "
            f"实际 diff={other_slots_diff}"
        )

    def test_enable_self_attn_false_no_matched_mask_conflict(
        self, encoder_kwargs, inputs
    ):
        """enable_self_attn=False + matched_mask 同时传入时不报错, 行为一致.

        互斥处理 (Plan agent C-Q2): 对角 mask 优先于 matched_mask.
        对角 mask 是 matched_mask 的超集 (阻断所有 inter-slot, 包括
        unmatched→matched 路径), 所以 matched_mask 在 enable_self_attn=False
        时完全冗余, 应被忽略.
        """
        x_t, t_emb, image_features = inputs
        encoder = SetEncoder(enable_self_attn=False, **encoder_kwargs)
        encoder.eval()

        # matched_mask: slot 0,1 为 matched, slot 2,3 为 unmatched
        matched_mask = torch.tensor([[True, True, False, False]])

        with torch.no_grad():
            # 只传 enable_self_attn=False (通过 encoder 构造), 不传 matched_mask
            _, pred_no_mask = encoder(x_t, t_emb, image_features)
            # 同时传 matched_mask (应被忽略)
            _, pred_with_mask = encoder(
                x_t, t_emb, image_features, matched_mask=matched_mask
            )

        # 两者应一致 (matched_mask 被对角 mask 覆盖, 无副作用)
        diff = (pred_with_mask - pred_no_mask).abs().max().item()
        assert diff < 1e-6, (
            f"enable_self_attn=False 时 matched_mask 应被忽略, "
            f"实际 diff={diff}"
        )

    def test_enable_self_attn_default_true(self, encoder_kwargs, inputs):
        """enable_self_attn 默认值为 True (向后兼容)."""
        x_t, t_emb, image_features = inputs
        # 不传 enable_self_attn, 应默认为 True
        encoder = SetEncoder(**encoder_kwargs)
        assert encoder.enable_self_attn is True
