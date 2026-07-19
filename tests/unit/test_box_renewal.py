"""Box Renewal 最小验证实验测试

验证 SetDiff mAP=0 的核心假设: unmatched slot 在推理时发散导致 OOD.
修复: 推理时每步 Euler 后, 低置信 slot 重置为 randn (回到训练分布),
保留高置信 slot 继续迭代.

TDD 红绿重构: 先写测试 (RED), 再实现 (GREEN).
"""

import torch

from setdiff.models.set_head import JointDiffusionHead


def _make_head(
    num_queries: int = 20,
    feat_channels: int = 64,
    snr_scale: float = 2.0,
    num_sample_steps: int = 4,
    box_renewal: bool = True,
    score_thr: float = 0.3,
    min_keep: int = 5,
) -> JointDiffusionHead:
    """构建测试用 head (小尺寸加速)."""
    return JointDiffusionHead(
        num_queries=num_queries,
        feat_channels=feat_channels,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=24,
        snr_scale=snr_scale,
        num_sample_steps=num_sample_steps,
        sampler='euler',
        box_renewal=box_renewal,
        score_thr=score_thr,
        min_keep=min_keep,
    )


# ============================================================
# 修复 1: JointDiffusionHead 接受 box_renewal 参数
# ============================================================


class TestBoxRenewalConfig:
    """JointDiffusionHead 应接受 box_renewal 配置参数."""

    def test_head_accepts_box_renewal_params(self):
        """head 应接受 box_renewal, score_thr, min_keep 参数."""
        head = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
            snr_scale=2.0,
            box_renewal=True,
            score_thr=0.3,
            min_keep=5,
        )
        assert hasattr(head, 'box_renewal')
        assert head.box_renewal is True
        assert head.score_thr == 0.3
        assert head.min_keep == 5

    def test_head_default_box_renewal_enabled(self):
        """默认应启用 box_renewal (验证实验需要).

        对齐 LDMDet: box_renewal 是标准推理组件, 默认 True.
        """
        head = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
        )
        # 默认启用 (验证实验), 可通过配置关闭做对比
        assert head.box_renewal is True
        # 默认参数对齐 LDMDet
        assert head.score_thr > 0.0
        assert head.min_keep > 0


# ============================================================
# 修复 2: _apply_box_renewal 行为测试
# ============================================================


class TestApplyBoxRenewal:
    """_apply_box_renewal: 低置信 slot 重置为噪声, 高置信 slot 保留."""

    def test_low_confidence_slots_replaced_by_noise(self):
        """置信度 < score_thr 的 slot 应被重置为噪声 (值改变)."""
        head = _make_head(
            num_queries=10, score_thr=0.5, min_keep=2
        )
        head.eval()

        # 构造 x_t 和 cls_logits
        B, N = 2, 10
        x_t = torch.zeros(B, N, 4)  # 全 0, 重置后应变为非 0
        # slot 0,1 高置信 (>0.5), 其余低置信
        cls_logits = torch.full((B, N, 24), -10.0)  # sigmoid(-10) ≈ 0
        cls_logits[:, 0, :] = 10.0  # sigmoid(10) ≈ 1
        cls_logits[:, 1, :] = 10.0

        x_t_new = head._apply_box_renewal(x_t, cls_logits)

        # slot 0,1 (高置信) 应保持 0
        assert torch.allclose(x_t_new[:, 0, :], torch.zeros(B, 4))
        assert torch.allclose(x_t_new[:, 1, :], torch.zeros(B, 4))
        # slot 2-9 (低置信) 应被重置为噪声 (非 0)
        assert not torch.allclose(x_t_new[:, 2, :], torch.zeros(B, 4))

    def test_high_confidence_slots_preserved(self):
        """置信度 >= score_thr 的 slot 应保持原值不变."""
        head = _make_head(
            num_queries=10, score_thr=0.3, min_keep=2
        )
        head.eval()

        B, N = 1, 10
        x_t = torch.randn(B, N, 4)  # 随机初始值
        # 所有 slot 高置信
        cls_logits = torch.full((B, N, 24), 10.0)  # sigmoid(10) ≈ 1

        x_t_new = head._apply_box_renewal(x_t, cls_logits)

        # 所有 slot 应保持原值
        assert torch.allclose(x_t_new, x_t)

    def test_min_keep_enforced(self):
        """当高置信 slot < min_keep 时, 应 topk 补足到 min_keep 个."""
        head = _make_head(
            num_queries=10, score_thr=0.9, min_keep=5
        )
        head.eval()

        B, N = 1, 10
        x_t = torch.randn(B, N, 4)
        # 所有 slot 低置信 (sigmoid(-10) ≈ 0 < 0.9)
        cls_logits = torch.full((B, N, 24), -10.0)
        # 但 slot 3 稍高一点 (topk 应保留它)
        cls_logits[:, 3, :] = -1.0  # sigmoid(-1) ≈ 0.27

        x_t_new = head._apply_box_renewal(x_t, cls_logits)

        # min_keep=5, 所以至少 5 个 slot 保留原值, 5 个被重置
        # slot 3 (最高置信) 应被保留
        assert torch.allclose(x_t_new[:, 3, :], x_t[:, 3, :])
        # 至少 5 个 slot 保留 (min_keep)
        preserved = sum(
            torch.allclose(x_t_new[:, i, :], x_t[:, i, :])
            for i in range(N)
        )
        assert preserved >= 5

    def test_all_slots_low_confidence_min_keep_applied(self):
        """所有 slot 都低置信时, min_keep 保证至少保留 min_keep 个."""
        head = _make_head(
            num_queries=10, score_thr=0.99, min_keep=3
        )
        head.eval()

        B, N = 1, 10
        x_t = torch.randn(B, N, 4)
        cls_logits = torch.full((B, N, 24), -10.0)  # 全低置信

        x_t_new = head._apply_box_renewal(x_t, cls_logits)

        # 至少 3 个 slot 保留 (min_keep=3)
        preserved = sum(
            torch.allclose(x_t_new[:, i, :], x_t[:, i, :])
            for i in range(N)
        )
        assert preserved >= 3
        # 其余 slot 被重置为噪声
        renewed = N - preserved
        assert renewed > 0

    def test_renewed_slots_are_random_noise(self):
        """被重置的 slot 应是 N(0,1) 噪声 (统计验证)."""
        head = _make_head(
            num_queries=100, score_thr=0.99, min_keep=1
        )
        head.eval()

        B, N = 1, 100
        x_t = torch.zeros(B, N, 4)  # 全 0
        cls_logits = torch.full((B, N, 24), -10.0)  # 全低置信

        torch.manual_seed(42)
        x_t_new = head._apply_box_renewal(x_t, cls_logits)

        # 至少 99 个 slot 被重置为噪声
        renewed = x_t_new[0, 1:, :]  # 排除 topk 保留的 slot 0
        # 噪声应非 0 且分布近似 N(0,1)
        assert renewed.abs().mean() > 0.1  # 非 0
        assert renewed.std() > 0.1  # 有方差, 不是常数

    def test_batch_independence(self):
        """不同 batch 的 renewal 应独立 (基于各自置信度)."""
        head = _make_head(
            num_queries=10, score_thr=0.5, min_keep=1
        )
        head.eval()

        B, N = 2, 10
        x_t = torch.zeros(B, N, 4)
        # batch 0: slot 0 高置信; batch 1: slot 5 高置信
        cls_logits = torch.full((B, N, 24), -10.0)
        cls_logits[0, 0, :] = 10.0
        cls_logits[1, 5, :] = 10.0

        torch.manual_seed(42)
        x_t_new = head._apply_box_renewal(x_t, cls_logits)

        # batch 0: slot 0 保留 (0), slot 1 重置 (非0)
        assert torch.allclose(x_t_new[0, 0, :], torch.zeros(4))
        assert not torch.allclose(x_t_new[0, 1, :], torch.zeros(4))
        # batch 1: slot 5 保留 (0), slot 0 重置 (非0)
        assert torch.allclose(x_t_new[1, 5, :], torch.zeros(4))
        assert not torch.allclose(x_t_new[1, 0, :], torch.zeros(4))


# ============================================================
# 修复 3: predict() 集成 box_renewal
# ============================================================


class TestPredictWithBoxRenewal:
    """predict() 应在 Euler 迭代中应用 box_renewal."""

    def test_predict_returns_valid_output_with_renewal(self):
        """启用 box_renewal 时 predict 应正常返回."""
        head = _make_head(
            num_queries=10, num_sample_steps=4, box_renewal=True
        )
        head.eval()

        B = 2
        image_features = torch.randn(B, 50, 64)
        with torch.no_grad():
            outputs = head.predict(image_features)

        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        assert outputs['pred_logits'].shape == (B, 10, 24)
        assert outputs['pred_boxes'].shape == (B, 10, 4)
        assert torch.isfinite(outputs['pred_logits']).all()
        assert torch.isfinite(outputs['pred_boxes']).all()

    def test_predict_without_renewal_still_works(self):
        """关闭 box_renewal 时 predict 应兼容旧逻辑."""
        head = _make_head(
            num_queries=10, num_sample_steps=4, box_renewal=False
        )
        head.eval()

        B = 1
        image_features = torch.randn(B, 50, 64)
        with torch.no_grad():
            outputs = head.predict(image_features)

        assert outputs['pred_logits'].shape == (B, 10, 24)
        assert outputs['pred_boxes'].shape == (B, 10, 4)

    def test_box_renewal_not_applied_on_last_step(self):
        """最后一步 Euler 后不应应用 renewal (之后不再迭代)."""
        # 通过 spy 验证: 最后一步后 _apply_box_renewal 不被调用
        head = _make_head(
            num_queries=10, num_sample_steps=3, box_renewal=True
        )
        head.eval()

        call_count = 0
        original_method = head._apply_box_renewal

        def counting_spy(x_t, cls_logits):
            nonlocal call_count
            call_count += 1
            return original_method(x_t, cls_logits)

        head._apply_box_renewal = counting_spy

        B = 1
        image_features = torch.randn(B, 50, 64)
        with torch.no_grad():
            head.predict(image_features)

        # num_sample_steps=3, renewal 应在 step 0, 1 后调用 (非最后一步)
        # 最后一步 (step 2) 后不调用
        assert call_count == 2, (
            f'Expected 2 renewal calls (3 steps - 1 last), got {call_count}'
        )

    def test_box_renewal_disabled_no_calls(self):
        """box_renewal=False 时 _apply_box_renewal 不应被调用."""
        head = _make_head(
            num_queries=10, num_sample_steps=4, box_renewal=False
        )
        head.eval()

        call_count = 0
        original_method = head._apply_box_renewal

        def counting_spy(x_t, cls_logits):
            nonlocal call_count
            call_count += 1
            return original_method(x_t, cls_logits)

        head._apply_box_renewal = counting_spy

        B = 1
        image_features = torch.randn(B, 50, 64)
        with torch.no_grad():
            head.predict(image_features)

        assert call_count == 0


# ============================================================
# 修复 4: box_renewal 保持确定性 (seed 可复现)
# ============================================================


class TestBoxRenewalDeterminism:
    """box_renewal 的噪声生成应支持 seed 可复现 (test.py --seed 42)."""

    def test_same_seed_same_output(self):
        """相同 seed 下 predict 输出应一致."""
        head = _make_head(
            num_queries=10, num_sample_steps=4, box_renewal=True
        )
        head.eval()

        B = 1
        image_features = torch.randn(B, 50, 64)

        torch.manual_seed(42)
        with torch.no_grad():
            out1 = head.predict(image_features)

        torch.manual_seed(42)
        with torch.no_grad():
            out2 = head.predict(image_features)

        assert torch.allclose(out1['pred_boxes'], out2['pred_boxes'])
        assert torch.allclose(out1['pred_logits'], out2['pred_logits'])
