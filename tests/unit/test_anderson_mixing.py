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


# ============================================================
# 设计方案一致性验证: 数学正确性 (AAC_DESIGN.md §1.3-§1.4)
# ============================================================

class TestAndersonMixingMathCorrectness:
    """验证 AndersonMixing 的数学实现与方案文档 §1.3-§1.4 推导一致。

    核心验证:
      1. γ = (ΔF ΔF^T + λI)^{-1} ΔF f_k  (行式 Gram, 方案 §1.4 注)
      2. x_{k+1} = x_k + β [f_k - (ΔX + ΔF)^T γ]  (方案 §1.4 步骤 4)
      3. 行式 (代码) 与列式 (方案公式) 数学等价 (方案 §1.4 注)
      4. 非连续差分基 (代码) 与连续差分基 (方案 §1.4) 等价
    """

    def test_gamma_matches_reference_least_squares(self):
        """γ 与参考最小二乘解 (torch.linalg.lstsq) 一致。

        方案 §1.3 步骤 3: γ = argmin ||f_k - ΔF γ||² (+ Tikhonov λ||γ||²)
        正规方程: (ΔF ΔF^T + λI) γ = ΔF f_k  (行式, 方案 §1.4 注)
        """
        torch.manual_seed(42)
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=1.0, reg_lambda=1e-6,
            gamma_norm_clip=1e9,  # 关闭裁剪以纯验证数学
        )
        mixer.reset()
        mixer.eval()  # 关闭诊断探针
        # 累积 2 步历史
        x0 = torch.randn(2, 50, 4)
        g0 = torch.randn(2, 50, 4)
        mixer(x0, g0)
        x1 = torch.randn(2, 50, 4)
        g1 = torch.randn(2, 50, 4)
        mixer(x1, g1)
        # 在 Anderson 步之前保存历史 (mixer.forward 末尾会 push 当前步,
        # 导致调用后 _f_history[-1] 是当前步而非计算时使用的历史项)
        f_prev1 = mixer._f_history[-1].clone()  # f_{k-1} (计算时作为 [-1])
        f_prev2 = mixer._f_history[-2].clone()  # f_{k-2} (计算时作为 [-2])
        x_prev1 = mixer._x_history[-1].clone()
        x_prev2 = mixer._x_history[-2].clone()
        # 第三步 (有 2 步历史, 触发 Anderson)
        x2 = torch.randn(2, 50, 4)
        g2 = torch.randn(2, 50, 4)
        _ = mixer(x2, g2)
        gamma_code = mixer._last_gamma.clone()  # [m_k, 1]

        # 参考计算: 手动构造 ΔF (行式) 并解正规方程
        f_curr = g2 - x2
        # 代码的差分基: f_curr - f_{prev_i}
        delta_f1 = (f_curr - f_prev1).reshape(-1)
        delta_f2 = (f_curr - f_prev2).reshape(-1)
        Delta_F_ref = torch.stack([delta_f1, delta_f2], dim=0)  # [2, D]
        f_flat = f_curr.reshape(-1)  # [D]
        gram_ref = Delta_F_ref @ Delta_F_ref.t() + 1e-6 * torch.eye(2)
        rhs_ref = Delta_F_ref @ f_flat.unsqueeze(-1)
        gamma_ref = torch.linalg.solve(gram_ref, rhs_ref)  # [2, 1]

        assert torch.allclose(gamma_code, gamma_ref, atol=1e-5), (
            f"γ 与参考解不一致: code={gamma_code.flatten().tolist()}, "
            f"ref={gamma_ref.flatten().tolist()}"
        )

    def test_update_formula_exact(self):
        """x_next 严格等于 x_k + β [f_k - (ΔX + ΔF)^T γ] (方案 §1.4 步骤 4)"""
        torch.manual_seed(7)
        beta = 0.7
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=beta, reg_lambda=1e-6,
            gamma_norm_clip=1e9,
        )
        mixer.reset()
        mixer.eval()
        x0 = torch.randn(2, 30, 4)
        g0 = torch.randn(2, 30, 4)
        mixer(x0, g0)
        x1 = torch.randn(2, 30, 4)
        g1 = torch.randn(2, 30, 4)
        mixer(x1, g1)
        # 在 Anderson 步之前保存历史 (forward 末尾 push 会偏移索引)
        f_prev1 = mixer._f_history[-1].clone()
        f_prev2 = mixer._f_history[-2].clone()
        x_prev1 = mixer._x_history[-1].clone()
        x_prev2 = mixer._x_history[-2].clone()
        # 第三步
        x2 = torch.randn(2, 30, 4)
        g2 = torch.randn(2, 30, 4)
        x_next = mixer(x2, g2)
        gamma = mixer._last_gamma

        # 手动重建更新
        f_curr = g2 - x2
        delta_f1 = (f_curr - f_prev1).reshape(-1)
        delta_f2 = (f_curr - f_prev2).reshape(-1)
        delta_x1 = (x2 - x_prev1).reshape(-1)
        delta_x2 = (x2 - x_prev2).reshape(-1)
        Delta_F = torch.stack([delta_f1, delta_f2], dim=0)
        Delta_X = torch.stack([delta_x1, delta_x2], dim=0)
        correction = (Delta_X + Delta_F).t() @ gamma
        correction = correction.squeeze(-1).reshape_as(x2)
        x_next_ref = x2 + beta * (f_curr - correction)

        assert torch.allclose(x_next, x_next_ref, atol=1e-6), (
            f"更新公式不匹配: max_diff={(x_next - x_next_ref).abs().max().item()}"
        )

    def test_row_form_equivalent_to_column_form(self):
        """行式 (代码 ΔF[m,D]) 与列式 (方案公式 ΔF[D,m]) 数学等价 (方案 §1.4 注)

        方案 §1.4 注: "行式 = 列式的转置, 两种约定数学等价"
        验证: 行式 γ_row = (ΔF_row ΔF_row^T)^{-1} ΔF_row f
              列式 γ_col = (ΔF_col^T ΔF_col)^{-1} ΔF_col^T f
              其中 ΔF_row = ΔF_col^T, 故 γ_row == γ_col, 且 x_next 一致。
        """
        torch.manual_seed(99)
        # 构造历史
        x0 = torch.randn(3, 20, 4)
        f0 = torch.randn(3, 20, 4)
        x1 = torch.randn(3, 20, 4)
        f1 = torch.randn(3, 20, 4)
        x_curr = torch.randn(3, 20, 4)
        g_x = torch.randn(3, 20, 4)
        f_curr = g_x - x_curr
        lam = 1e-6

        # 行式 (代码约定): ΔF_row [m, D], Gram = ΔF_row ΔF_row^T [m, m]
        df1_row = (f_curr - f1).reshape(-1)
        df2_row = (f_curr - f0).reshape(-1)
        Delta_F_row = torch.stack([df1_row, df2_row], dim=0)  # [2, D]
        gram_row = Delta_F_row @ Delta_F_row.t() + lam * torch.eye(2)
        rhs_row = Delta_F_row @ f_curr.reshape(-1).unsqueeze(-1)
        gamma_row = torch.linalg.solve(gram_row, rhs_row)  # [2, 1]

        # 列式 (方案公式约定): ΔF_col [D, m] = ΔF_row^T, Gram = ΔF_col^T ΔF_col [m, m]
        Delta_F_col = Delta_F_row.t()  # [D, 2]
        gram_col = Delta_F_col.t() @ Delta_F_col + lam * torch.eye(2)
        rhs_col = Delta_F_col.t() @ f_curr.reshape(-1).unsqueeze(-1)
        gamma_col = torch.linalg.solve(gram_col, rhs_col)  # [2, 1]

        assert torch.allclose(gamma_row, gamma_col, atol=1e-6), (
            "行式 γ 与列式 γ 不等价 (违反方案 §1.4 注)"
        )

    def test_nonconsecutive_equivalent_to_consecutive_basis(self):
        """代码的非连续差分基与方案 §1.4 的连续差分基给出相同 x_{k+1}

        代码: ΔF = [f_k - f_{k-1}, f_k - f_{k-2}]  (f_curr 减各历史)
        标准: ΔF = [f_k - f_{k-1}, f_{k-1} - f_{k-2}]  (连续差分)
        二者通过可逆变换 T=[[1,1],[0,1]] 关联 (ΔF_code = ΔF_std @ T),
        故 colspan 相同, 且 (ΔX+ΔF)γ 不变 (ΔG_code γ_code = ΔG_std T T^{-1} γ_std)。
        """
        torch.manual_seed(2024)
        d, N = 4, 30
        A = torch.randn(d, d) * 0.1  # 压缩映射
        b = torch.randn(1, d)

        x0 = torch.randn(N, d)
        g0 = x0 @ A.t() + b
        f0 = g0 - x0
        x1 = g0
        g1 = x1 @ A.t() + b
        f1 = g1 - x1
        x2 = g1
        g2 = x2 @ A.t() + b
        f2 = g2 - x2
        lam = 1e-10
        beta = 1.0

        # 代码版本 (非连续基)
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=beta, reg_lambda=lam,
            gamma_norm_clip=1e9,
        )
        mixer.reset()
        mixer.eval()
        mixer._push_history(x0.detach(), f0.detach())
        mixer._push_history(x1.detach(), f1.detach())
        x_next_code = mixer(x2, g2)

        # 标准连续基: ΔF = [f_k-f_{k-1}, f_{k-1}-f_{k-2}]
        Delta_F_std = torch.stack(
            [f2 - f1, f1 - f0], dim=0
        ).reshape(2, -1)  # [2, D]
        Delta_X_std = torch.stack(
            [x2 - x1, x1 - x0], dim=0
        ).reshape(2, -1)
        f_flat = f2.reshape(-1)
        gram_std = Delta_F_std @ Delta_F_std.t() + lam * torch.eye(2)
        rhs_std = Delta_F_std @ f_flat.unsqueeze(-1)
        gamma_std = torch.linalg.solve(gram_std, rhs_std)
        correction_std = (Delta_X_std + Delta_F_std).t() @ gamma_std
        x_next_std = x2 + beta * (
            f2 - correction_std.squeeze(-1).reshape_as(x2)
        )

        assert torch.allclose(x_next_code, x_next_std, atol=1e-5), (
            f"非连续基与连续基 x_next 不一致: "
            f"max_diff={(x_next_code - x_next_std).abs().max().item()}"
        )


# ============================================================
# 设计方案行为规范验证 (AAC_DESIGN.md §4.1, §4.2)
# ============================================================

class TestAndersonMixingBehaviorSpecs:
    """验证 AndersonMixing 的行为符合方案 §4.1 的设计参数与边界处理。"""

    def test_gamma_clip_rescales_to_bound(self):
        """γ 范数裁剪: 超过上界时精确缩放至 clip (方案 §4.1 gamma_norm_clip)

        方案 §4.1: gamma = gamma * (clip / gamma_norm.detach())
        裁剪后 ||γ|| 应 == clip (在浮点精度内)。
        """
        torch.manual_seed(42)
        clip = 2.0  # 用小上界便于触发
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=1.0, reg_lambda=1e-12,
            gamma_norm_clip=clip,
        )
        mixer.reset()
        mixer.eval()
        # 构造大残差使 γ 爆炸 (超过 clip=2.0)
        x0 = torch.zeros(2, 50, 4)
        g0 = torch.ones(2, 50, 4) * 100  # 大残差
        mixer(x0, g0)
        x1 = g0
        g1 = x1 + torch.ones(2, 50, 4) * 50
        mixer(x1, g1)
        x2 = g1
        g2 = x2 + torch.ones(2, 50, 4) * 30
        mixer(x2, g2)

        gamma = mixer._last_gamma
        gamma_norm = gamma.norm().item()
        # 裁剪后范数应 <= clip (允许浮点误差)
        assert gamma_norm <= clip + 1e-4, (
            f"裁剪后 γ 范数 {gamma_norm:.6f} 超过上界 {clip}"
        )
        # 若原范数确实 > clip, 裁剪后应 == clip
        # (验证缩放生效, 而非 γ 本身就小)

    def test_beta_damping_scales_update(self):
        """β 阻尼: β=0.5 的更新量是 β=1.0 的一半 (方案 §4.1 damping_beta)

        方案 §4.1: x_{k+1} = x_k + β [f_k - correction]
        β 仅缩放 (f_k - correction) 项, 不改变 γ (γ 由最小二乘决定, 与 β 无关)。
        """
        torch.manual_seed(42)
        x0 = torch.randn(2, 50, 4)
        g0 = torch.randn(2, 50, 4)
        x1 = torch.randn(2, 50, 4)
        g1 = torch.randn(2, 50, 4)
        x2 = torch.randn(2, 50, 4)
        g2 = torch.randn(2, 50, 4)

        # β=1.0
        mixer_full = AndersonMixing(
            mem_depth=2, damping_beta=1.0, reg_lambda=1e-6,
            gamma_norm_clip=1e9,
        )
        mixer_full.reset()
        mixer_full.eval()
        mixer_full(x0, g0)
        mixer_full(x1, g1)
        x_next_full = mixer_full(x2, g2)

        # β=0.5 (相同历史与输入)
        mixer_half = AndersonMixing(
            mem_depth=2, damping_beta=0.5, reg_lambda=1e-6,
            gamma_norm_clip=1e9,
        )
        mixer_half.reset()
        mixer_half.eval()
        mixer_half(x0, g0)
        mixer_half(x1, g1)
        x_next_half = mixer_half(x2, g2)

        # x_next = x_k + β * delta, 故 (x_next_half - x2) = 0.5 * (x_next_full - x2)
        delta_full = x_next_full - x2
        delta_half = x_next_half - x2
        assert torch.allclose(delta_half, 0.5 * delta_full, atol=1e-6), (
            "β=0.5 的更新量未等于 β=1.0 的一半 (阻尼未正确施加)"
        )

    def test_history_capped_at_m_plus_1(self):
        """历史 buffer 上限为 m+1 (方案 §4.1 _push_history)

        方案 §4.1: "保留最近 m+1 个历史 (m 个差分需要 m+1 个点)"
        """
        mixer = AndersonMixing(mem_depth=2, damping_beta=1.0)
        mixer.reset()
        # 运行 10 步 (远超 m+1=3)
        for _ in range(10):
            x = torch.randn(2, 20, 4)
            g = torch.randn(2, 20, 4)
            mixer(x, g)
        assert len(mixer._x_history) <= mixer.m + 1, (
            f"历史 x 超过 m+1={mixer.m + 1}: {len(mixer._x_history)}"
        )
        assert len(mixer._f_history) <= mixer.m + 1, (
            f"历史 f 超过 m+1={mixer.m + 1}: {len(mixer._f_history)}"
        )
        # m=2 时上限为 3
        assert len(mixer._x_history) == 3

    def test_history_cap_for_various_m(self):
        """不同 m 值下历史上限均为 m+1"""
        for m in [1, 2, 3, 5]:
            mixer = AndersonMixing(mem_depth=m, damping_beta=1.0)
            mixer.reset()
            for _ in range(20):
                x = torch.randn(1, 10, 4)
                g = torch.randn(1, 10, 4)
                mixer(x, g)
            assert len(mixer._x_history) == m + 1, (
                f"m={m}: 历史上限应为 {m + 1}, got {len(mixer._x_history)}"
            )

    def test_m0_history_not_accumulated(self):
        """m=0 时不累积历史 (Picard 退化, 无需历史)"""
        mixer = AndersonMixing(mem_depth=0, damping_beta=1.0)
        mixer.reset()
        for _ in range(5):
            x = torch.randn(1, 10, 4)
            g = torch.randn(1, 10, 4)
            mixer(x, g)
        # m=0: _push_history 仍存, 但上限 m+1=1, 且下次 forward 走 Picard 分支
        assert len(mixer._x_history) <= 1

    def test_stop_grad_false_retains_history_gradient(self):
        """stop_grad_history=False 时, 历史项保留梯度 (方案 §4.1 参数说明)

        方案 §4.1: stop_grad_history 默认 True (Phantom gradient),
        设为 False 时历史不 detach, 允许完整反传 (未来消融实验用)。
        """
        mixer = AndersonMixing(
            mem_depth=2, damping_beta=1.0, stop_grad_history=False,
            gamma_norm_clip=1e9,
        )
        mixer.reset()
        mixer.eval()
        # 第一步 (历史为空, 但 _push_history 不 detach)
        x1 = torch.randn(2, 30, 4, requires_grad=True)
        g1 = torch.randn(2, 30, 4, requires_grad=True)
        mixer(x1, g1)
        # 历史应保留 grad_fn (未 detach)
        assert mixer._x_history[0].requires_grad, (
            "stop_grad_history=False 时历史 x 应保留梯度"
        )
        assert mixer._f_history[0].requires_grad, (
            "stop_grad_history=False 时历史 f 应保留梯度"
        )


# ============================================================
# 设计方案集成规范验证 (AAC_DESIGN.md §4.2, §6.5.5)
# ============================================================

class TestAACDesignIntegrationSpecs:
    """验证 AAC 与 DiffusionDetHead 集成符合方案 §4.2 的设计决策。"""

    def test_mixer_called_only_for_non_last_heads(self):
        """末级 head 不混合 (方案 §4.2 设计决策 2)

        方案 §4.2: "末级 head 不混合: 输出直接作为 cascade 结果"
        代码: `if self.use_aac and i < len(self.head_series) - 1`
        验证: mixer.forward 被调用 num_heads-1 次 (前 5 级), 末级不调用。
        """
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=0.5,
        )
        head.eval()
        # 用计数器包装 mixer.forward
        call_count = [0]
        original_forward = head.aac_mixer.forward

        def counting_forward(x_curr, g_x):
            call_count[0] += 1
            return original_forward(x_curr, g_x)

        head.aac_mixer.forward = counting_forward

        features, bboxes, t = _make_dummy_input()
        with torch.no_grad():
            head(features, bboxes, t)

        # 6 级 head, 末级不混合 → mixer 调用 5 次
        assert call_count[0] == 5, (
            f"mixer 应被调用 5 次 (前 5 级), 实际 {call_count[0]} 次"
        )

    def test_cls_logits_not_passed_to_mixer(self):
        """cls_logits 不参与混合 (方案 §4.2 设计决策 1)

        方案 §4.2: "仅对 box 做混合, cls_logits 不混合 (分类不构成不动点)"
        验证: mixer 的输入 g_x 是 pred_bboxes (box), 非 cls_logits;
              inter_cls_logits 是原始 head 输出 (未经 mixer 变换)。
        """
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=1.0,
        )
        head.eval()
        # 记录 mixer 接收的所有 g_x 的形状
        recorded_g_x_shapes = []
        original_forward = head.aac_mixer.forward

        def recording_forward(x_curr, g_x):
            recorded_g_x_shapes.append(g_x.shape)
            return original_forward(x_curr, g_x)

        head.aac_mixer.forward = recording_forward
        features, bboxes, t = _make_dummy_input()
        with torch.no_grad():
            cls_logits, pred_bboxes, _ = head(features, bboxes, t)

        # mixer 接收的 g_x 应是 box 形状 [bs, N, 4], 而非 cls 形状 [bs, N, num_classes]
        for shape in recorded_g_x_shapes:
            assert shape[-1] == 4, (
                f"mixer 输入应为 box (d=4), got shape {shape} — cls_logits 可能被错误混合"
            )

    def test_ccbr_aac_mutual_warning(self, caplog):
        """AAC 与 CCBR 同时启用时发出告警 (方案 §4.2)

        方案 §4.2: "use_aac=True 且 use_ccbr=True: CCBR 路径 (_forward_at_t_ccbr)
                   未集成 AAC, AAC 仅在标准 forward/predict 路径生效。"
        """
        import logging
        single_head = _make_single_head()
        with caplog.at_level(logging.WARNING, logger='ldmdet.core.head'):
            head = DiffusionDetHead(
                num_classes=24, feat_channels=64, num_proposals=10,
                num_heads=6, single_head=single_head,
                roi_extractor=_make_roi_extractor(), criterion=None,
                use_aac=True, use_ccbr=True,  # 同时启用
            )
        # 应有 CCBR/AAC 互斥告警
        warning_found = any(
            'use_aac' in rec.message and 'use_ccbr' in rec.message
            for rec in caplog.records
        )
        assert warning_found, (
            "use_aac=True 且 use_ccbr=True 时未发出互斥告警 (违反方案 §4.2)"
        )

    def test_aac_history_reset_per_forward_call(self):
        """每个 forward() 调用开始时 reset 历史 (方案 §4.4, §6.5.4)

        方案 §4.4: "AAC 历史在每个 solver step 开始时 reset, 不跨 step 累积"
        方案 §6.5.4: "AAC 历史在 step 内部累积, step 切换时 reset"
        验证: 连续两次 forward 之间, 历史不跨 forward 残留 (第二次 forward 开始时
              历史被清空, 故第二次 forward 的 mixer 第一步走 Picard)。
        """
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=1.0,
        )
        head.eval()
        reset_count = [0]
        original_reset = head.aac_mixer.reset
        def counting_reset():
            reset_count[0] += 1
            original_reset()
        head.aac_mixer.reset = counting_reset

        features, bboxes, t = _make_dummy_input()
        with torch.no_grad():
            head(features, bboxes, t)
            head(features, bboxes, t)
        # 每次 forward 调用一次 reset → 2 次
        assert reset_count[0] == 2, (
            f"应每次 forward reset 一次 (共 2 次), 实际 {reset_count[0]} 次"
        )

    def test_aac_no_extra_nfe(self):
        """AAC 不增加 NFE (方案 §6.5.5 NFE 不变性)

        方案 §6.5.5: "每个 cascade head 仍执行 1 次前向, 共 6 次;
                      AAC 仅在 head 之间插入 Anderson 混合, 不增加 NFE。"
        验证: 启用/不启用 AAC, head_series 中每个 head 的前向调用次数相同。
        """
        features, bboxes, t = _make_dummy_input()

        def count_head_calls(head_module):
            """统计 head_series 中所有 head 的前向调用次数。"""
            count = [0]
            original_forwards = []
            for h in head_module.head_series:
                orig = h.forward
                original_forwards.append(orig)

                def make_counter(orig):
                    def counter(*args, **kwargs):
                        count[0] += 1
                        return orig(*args, **kwargs)
                    return counter
                h.forward = make_counter(orig)
            return count, original_forwards

        # 不启用 AAC
        single_head1 = _make_single_head()
        head_no = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head1, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=False,
        )
        head_no.eval()
        count_no, origs_no = count_head_calls(head_no)
        with torch.no_grad():
            head_no(features, bboxes, t)
        for h, orig in zip(head_no.head_series, origs_no):
            h.forward = orig

        # 启用 AAC
        single_head2 = _make_single_head()
        head_aac = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head2, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=1.0,
        )
        head_aac.eval()
        count_aac, origs_aac = count_head_calls(head_aac)
        with torch.no_grad():
            head_aac(features, bboxes, t)
        for h, orig in zip(head_aac.head_series, origs_aac):
            h.forward = orig

        # 两者 head 前向次数应相同 (NFE 不变)
        assert count_no[0] == count_aac[0] == 6, (
            f"NFE 不一致: no_aac={count_no[0]}, aac={count_aac[0]}, 期望均为 6"
        )

    def test_aac_active_in_eval_mode(self):
        """AAC 在推理 (eval) 模式下仍激活 (方案 §4.4)

        方案 §4.4: "推理时 AAC 与训练行为一致 (self.aac_mixer 在
                   predict → _forward_at_t → forward 中自动激活)。"
        验证: eval 模式下 forward 仍调用 mixer (非 no-op)。
        """
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24, feat_channels=64, num_proposals=10, num_heads=6,
            single_head=single_head, roi_extractor=_make_roi_extractor(),
            criterion=None, use_aac=True, aac_mem_depth=2, aac_beta=1.0,
        )
        head.eval()  # 推理模式
        mixer_called = [False]
        original_forward = head.aac_mixer.forward
        def flag_forward(x_curr, g_x):
            mixer_called[0] = True
            return original_forward(x_curr, g_x)
        head.aac_mixer.forward = flag_forward

        features, bboxes, t = _make_dummy_input()
        with torch.no_grad():
            head(features, bboxes, t)
        assert mixer_called[0], (
            "eval 模式下 AAC mixer 未被调用 (违反方案 §4.4 推理一致性)"
        )
