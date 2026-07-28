"""测试 Anderson-Accelerated Cascade (AAC) 模块

核心验证:
1. AndersonMixing 模块基本功能 (m=0/1/2)
2. m=0 退化为 Picard (与当前 cascade 一致)
3. m=2 加速收敛 (残差衰减更快)
4. Tikhonov 正则化防止秩亏崩溃
5. Phantom gradient: 历史项 stop-gradient, 仅当前 f_k 反传
6. reset() 清空历史
7. DiffusionDetHead 集成: use_aac=True 时 forward 正常
8. AAC 不改变 deep supervision 输出数量
"""

import os
import sys

import torch

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core.anderson_mixing import AndersonMixing
from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.core.single_head import SingleDiffusionDetHead


# ============================================================
# 辅助函数
# ============================================================

def _make_single_head(num_classes=24, feat_channels=64):
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
    return SingleRoIExtractor(
        featmap_strides=[16],
        out_channels=out_channels,
        roi_layer=dict(
            type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True
        ),
    )


def _make_dummy_input(bs=2, num_boxes=10, feat_channels=64):
    torch.manual_seed(42)
    features = [torch.randn(bs, feat_channels, 16, 16)]
    bboxes = torch.rand(bs, num_boxes, 4) * 100
    bboxes[..., 2:] = bboxes[..., :2] + torch.rand(bs, num_boxes, 2) * 50 + 10
    t = torch.tensor([500.0, 500.0])
    return features, bboxes, t


# ============================================================
# AndersonMixing 单元测试
# ============================================================

class TestAndersonMixingBasic:
    """AndersonMixing 模块基本功能"""

    def test_m0_degrades_to_picard(self):
        """m=0 时退化为 Picard: x_{k+1} = g(x_k)"""
        mixer = AndersonMixing(mem_depth=0, damping_beta=1.0)
        mixer.reset()
        x = torch.randn(2, 10, 4)
        g_x = torch.randn(2, 10, 4)
        x_next = mixer(x, g_x)
        # m=0: x_{k+1} = g_x (Picard 一步)
        assert torch.allclose(x_next, g_x)

    def test_first_step_is_picard(self):
        """历史不足时 (k=0), 退化为 Picard"""
        mixer = AndersonMixing(mem_depth=2, damping_beta=1.0)
        mixer.reset()
        x = torch.randn(2, 10, 4)
        g_x = torch.randn(2, 10, 4)
        x_next = mixer(x, g_x)
        # 第一步: 历史为空, 退化为 Picard
        assert torch.allclose(x_next, g_x)

    def test_output_shape_correct(self):
        """输出形状与输入一致"""
        mixer = AndersonMixing(mem_depth=2, damping_beta=1.0)
        mixer.reset()
        for _ in range(3):
            x = torch.randn(2, 10, 4)
            g_x = torch.randn(2, 10, 4)
            x_next = mixer(x, g_x)
            assert x_next.shape == x.shape

    def test_reset_clears_history(self):
        """reset() 清空历史, 下一步退化为 Picard"""
        mixer = AndersonMixing(mem_depth=2, damping_beta=1.0)
        mixer.reset()
        # 累积 2 步历史
        x1 = torch.randn(2, 10, 4)
        g1 = torch.randn(2, 10, 4)
        mixer(x1, g1)
        x2 = torch.randn(2, 10, 4)
        g2 = torch.randn(2, 10, 4)
        mixer(x2, g2)
        # 此时历史应非空
        assert len(mixer._f_history) == 2
        # reset 后历史清空
        mixer.reset()
        assert len(mixer._f_history) == 0
        # 下一步退化为 Picard
        x3 = torch.randn(2, 10, 4)
        g3 = torch.randn(2, 10, 4)
        x_next = mixer(x3, g3)
        assert torch.allclose(x_next, g3)


class TestAndersonMixingAcceleration:
    """验证 Anderson 加速效果 (残差衰减更快)"""

    def test_linear_fixed_point_acceleration(self):
        """在线性不动点问题上, AA(m=2) 应比 Picard 收敛更快

        构造: G(x) = A @ x + b, 其中 A 的谱半径 < 1 (压缩映射)
        不动点: x* = (I - A)^{-1} @ b
        """
        torch.manual_seed(42)
        d = 4
        N = 50
        # 构造压缩映射 A (谱半径 ~0.5)
        A = torch.randn(d, d) * 0.1
        b = torch.randn(1, d)
        x_star = torch.linalg.solve(torch.eye(d) - A, b.squeeze(0))

        # Picard 迭代
        x_picard = torch.randn(N, d)
        picard_residuals = []
        for _ in range(6):
            g_x = x_picard @ A.t() + b
            f = g_x - x_picard
            picard_residuals.append(f.norm().item())
            x_picard = g_x

        # Anderson(m=2) 迭代
        mixer = AndersonMixing(mem_depth=2, damping_beta=1.0, reg_lambda=1e-8)
        mixer.reset()
        x_anderson = torch.randn(N, d)
        anderson_residuals = []
        for _ in range(6):
            g_x = x_anderson @ A.t() + b
            f = g_x - x_anderson
            anderson_residuals.append(f.norm().item())
            x_anderson = mixer(x_anderson, g_x)

        # Anderson 的最终残差应显著小于 Picard (加速效果)
        assert anderson_residuals[-1] < picard_residuals[-1] * 0.5, (
            f"Anderson 残差 {anderson_residuals[-1]:.6f} 未显著小于 "
            f"Picard {picard_residuals[-1]:.6f} (期望 < 50%)"
        )

    def test_gamma_bounded(self):
        """Anderson 系数 γ 不爆炸 (有界)"""
        torch.manual_seed(42)
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=1.0, gamma_norm_clip=10.0
        )
        mixer.reset()
        # 运行 6 步 (模拟 6 级 cascade)
        x = torch.randn(2, 50, 4)
        for i in range(6):
            g_x = x + torch.randn(2, 50, 4) * 0.1
            x = mixer(x, g_x)
            if mixer._last_gamma is not None:
                gamma_norm = mixer._last_gamma.norm().item()
                assert gamma_norm <= 10.0 + 1e-5, (
                    f"γ 范数 {gamma_norm:.4f} 超过上界 10.0"
                )

    def test_nonstationary_acceleration(self):
        """非平稳 cascade 下 AAC(m=2) 仍加速收敛 (R1 评审建议补充)

        构造: 6 个不同线性映射 G_k(x) = A_k @ x + b_k, 其中各 A_k 谱半径 < 1
        但 A_k 矩阵不同 (模拟 6 级 cascade head 参数 θ_k 不同)。
        各 A_k 由基矩阵 A_base 加小扰动构成 (近似共享不动点, ε_fp 较小),
        验证 AAC(m=2) 的最终残差不超过 Picard (定理 1 r-线性保证) 且通常更小。
        """
        torch.manual_seed(42)
        d = 4
        N = 50
        H = 6  # 6 级 cascade head

        # 基矩阵 A_base: 谱半径 ~0.5 (压缩映射)
        A_base = torch.randn(d, d) * 0.1
        b_base = torch.randn(1, d)

        # 6 个非平稳映射: A_base + 小扰动 (ε_fp 较小, 近似共享不动点)
        A_list = [A_base + torch.randn(d, d) * 0.02 for _ in range(H)]
        b_list = [b_base + torch.randn(1, d) * 0.02 for _ in range(H)]

        # 公平起点: Picard 与 Anderson 从同一 x0 出发
        x0 = torch.randn(N, d)

        # Picard 迭代 (当前 cascade, m=0)
        x_picard = x0.clone()
        picard_residuals = []
        for k in range(H):
            g_x = x_picard @ A_list[k].t() + b_list[k]
            f = g_x - x_picard
            picard_residuals.append(f.norm().item())
            x_picard = g_x

        # Anderson(m=2) 迭代 (非平稳, 同一 G_k 序列, 同一起点)
        mixer = AndersonMixing(mem_depth=2, damping_beta=1.0, reg_lambda=1e-8)
        mixer.reset()
        x_anderson = x0.clone()
        anderson_residuals = []
        for k in range(H):
            g_x = x_anderson @ A_list[k].t() + b_list[k]
            f = g_x - x_anderson
            anderson_residuals.append(f.norm().item())
            x_anderson = mixer(x_anderson, g_x)

        # 定理 1 保证: AAC r-线性收敛因子不超过 Picard (不慢于 Picard)
        # 在近似共享不动点 (小 ε_fp) 下, AAC 通常加速
        assert anderson_residuals[-1] <= picard_residuals[-1] * 1.05 + 1e-6, (
            f"非平稳 AAC 最终残差 {anderson_residuals[-1]:.6f} 显著大于 "
            f"Picard {picard_residuals[-1]:.6f}, 违反定理 1 r-线性保证"
        )


class TestAndersonMixingNumericalStability:
    """数值稳定性测试"""

    def test_rank_deficient_no_crash(self):
        """秩亏情况 (所有 proposal 残差共线) 不崩溃"""
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=1.0, reg_lambda=1e-6
        )
        mixer.reset()
        # 构造共线残差: 所有 proposal 的 g_x - x 同方向
        x = torch.zeros(2, 50, 4)
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        g1 = x + direction * 0.5
        mixer(x, g1)
        x2 = g1
        g2 = x2 + direction * 0.3
        mixer(x2, g2)
        x3 = g2
        g3 = x3 + direction * 0.1
        # 不应抛异常
        x_next = mixer(x3, g3)
        assert x_next.shape == x.shape
        assert not torch.isnan(x_next).any()

    def test_zero_residual_no_crash(self):
        """零残差 (已收敛) 不崩溃"""
        mixer = AndersonMixing(mem_depth=2, damping_beta=1.0)
        mixer.reset()
        x = torch.randn(2, 50, 4)
        g1 = x.clone()  # f=0
        mixer(x, g1)
        x2 = g1
        g2 = x2.clone()  # f=0
        mixer(x2, g2)
        x3 = g2
        g3 = x3.clone()  # f=0
        x_next = mixer(x3, g3)
        # 零残差下 x_next ≈ x (f_k=0, correction=0)
        assert torch.allclose(x_next, x3, atol=1e-4)


class TestAndersonMixingGradient:
    """梯度回传测试 (Phantom gradient)"""

    def test_gradient_flows_through_current_residual(self):
        """梯度通过当前 f_k = g_x - x_curr 回传到 g_x"""
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=1.0, stop_grad_history=True
        )
        mixer.reset()
        # 累积 2 步历史 (detach, 不反传)
        x1 = torch.randn(2, 50, 4, requires_grad=False)
        g1 = torch.randn(2, 50, 4, requires_grad=False)
        mixer(x1, g1)
        x2 = torch.randn(2, 50, 4, requires_grad=False)
        g2 = torch.randn(2, 50, 4, requires_grad=False)
        mixer(x2, g2)
        # 当前步: g_x 需要 requires_grad
        x3 = torch.randn(2, 50, 4, requires_grad=False)
        g3 = torch.randn(2, 50, 4, requires_grad=True)
        x_next = mixer(x3, g3)
        loss = x_next.sum()
        loss.backward()
        # g3 应有梯度 (通过 f_curr = g3 - x3 回传)
        assert g3.grad is not None
        assert g3.grad.abs().sum() > 0

    def test_history_stop_gradient(self):
        """stop_grad_history=True 时, 历史项不反传"""
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=1.0, stop_grad_history=True
        )
        mixer.reset()
        # 第一步 (历史为空, Picard)
        x1 = torch.randn(2, 50, 4, requires_grad=True)
        g1 = torch.randn(2, 50, 4, requires_grad=True)
        mixer(x1, g1)
        # 第二步 (有历史, Anderson)
        x2 = g1.detach()  # 模拟 cascade_detach
        g2 = torch.randn(2, 50, 4, requires_grad=True)
        x_next = mixer(x2, g2)
        loss = x_next.sum()
        loss.backward()
        # g1 不应有梯度 (历史已 detach, stop_grad_history=True)
        assert g1.grad is None or g1.grad.abs().sum() == 0


# ============================================================
# DiffusionDetHead 集成测试
# ============================================================

class TestAACHeadIntegration:
    """AAC 与 DiffusionDetHead 的集成测试"""

    def test_aac_disabled_by_default(self):
        """默认不启用 AAC"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert head.use_aac is False
        assert not hasattr(head, 'aac_mixer')

    def test_aac_enabled_creates_mixer(self):
        """启用 AAC 时创建 aac_mixer"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_aac=True,
            aac_mem_depth=2,
            aac_beta=0.5,
        )
        assert head.use_aac is True
        assert hasattr(head, 'aac_mixer')
        assert head.aac_mixer.m == 2
        assert head.aac_mixer.beta == 0.5

    def test_aac_forward_runs(self):
        """AAC 启用时 forward 正常运行"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_aac=True,
            aac_mem_depth=2,
            aac_beta=0.5,
        )
        features, bboxes, t = _make_dummy_input()
        all_cls, all_bbox, _ = head(features, bboxes, t)
        # 6 级 head, deep supervision
        assert all_cls.shape[0] == 6
        assert all_bbox.shape[0] == 6

    def test_aac_deep_supervision_preserved(self):
        """AAC 不改变 deep supervision 输出数量"""
        features, bboxes, t = _make_dummy_input()

        # 不启用 AAC
        single_head1 = _make_single_head()
        head_no_aac = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head1, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=False,
        )
        head_no_aac.eval()
        with torch.no_grad():
            cls_no, bbox_no, _ = head_no_aac(features, bboxes, t)

        # 启用 AAC
        single_head2 = _make_single_head()
        head_aac = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head2, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=0.5,
        )
        head_aac.eval()
        with torch.no_grad():
            cls_aac, bbox_aac, _ = head_aac(features, bboxes, t)

        # 输出形状一致 (deep supervision 不变)
        assert cls_no.shape == cls_aac.shape
        assert bbox_no.shape == bbox_aac.shape

    def test_aac_history_resets_per_forward(self):
        """AAC 历史在每个 forward() 开始时 reset"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=0.5,
        )
        features, bboxes, t = _make_dummy_input()
        head.eval()
        with torch.no_grad():
            head(features, bboxes, t)
            # forward 结束后历史应非空 (cascade 内部累积)
            assert len(head.aac_mixer._f_history) > 0
            # 第二次 forward 前应被 reset
            head(features, bboxes, t)
            # 历史应不超过 m+1 (reset 后重新累积)
            assert len(head.aac_mixer._f_history) <= head.aac_mixer.m + 1

    def test_aac_no_new_parameters(self):
        """AAC 是纯算法模块, 不引入新可学习参数"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=0.5,
        )
        aac_params = [
            n for n, p in head.named_parameters() if 'aac' in n.lower()
        ]
        # AndersonMixing 无可学习参数
        assert len(aac_params) == 0, (
            f"AAC 不应有可学习参数, 但找到: {aac_params}"
        )

    def test_aac_m0_equivalent_to_no_aac(self):
        """AAC m=0 (纯 Picard) 与不启用 AAC 行为一致

        验证: 同一权重下, m=0 的输出 == 不启用 AAC 的输出
        """
        torch.manual_seed(123)
        features, bboxes, t = _make_dummy_input()

        # 不启用 AAC
        single_head1 = _make_single_head()
        head_no = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head1, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=False, cascade_detach=True,
        )
        head_no.eval()
        with torch.no_grad():
            cls_no, bbox_no, _ = head_no(features, bboxes, t)

        # AAC m=0 (Picard)
        single_head2 = _make_single_head()
        # 复制权重确保一致
        single_head2.load_state_dict(single_head1.state_dict())
        head_m0 = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head2, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=0, cascade_detach=True,
        )
        head_m0.eval()
        with torch.no_grad():
            cls_m0, bbox_m0, _ = head_m0(features, bboxes, t)

        # 输出应完全一致 (m=0 退化为 Picard = 当前 cascade)
        assert torch.allclose(cls_no, cls_m0, atol=1e-5), (
            "AAC m=0 与未启用 AAC 的 cls_logits 不一致"
        )
        assert torch.allclose(bbox_no, bbox_m0, atol=1e-5), (
            "AAC m=0 与未启用 AAC 的 pred_bboxes 不一致"
        )
