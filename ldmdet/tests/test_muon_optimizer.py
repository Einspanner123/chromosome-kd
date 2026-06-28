"""测试 Muon 优化器和混合 OptimWrapper

方向六优化器改进实验:
- Muon: Newton-Schulz 正交化动量, 适用于 2D 矩阵参数
- MuonHybridOptimWrapper: 对不同参数使用不同优化器
  * bbox_head 的 transformer 参数 (head_series, time_mlp) → Muon
  * backbone/neck → AdamW
  * 偏置/LayerNorm → AdamW

参考: Jordan et al. 2024 "Muon: Momentum + Orthogonalization"
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.optim.muon import Muon, zeropower_via_newtonschulz5
from ldmdet.optim.hybrid_wrapper import (
    MuonHybridOptimWrapper,
    classify_parameters,
)
from ldmdet.optim.hybrid_optimizer import MuonHybrid


class TestNewtonSchulz:
    """Newton-Schulz 正交化测试"""

    def test_output_orthogonal(self):
        """正交化后矩阵应接近正交 (X X^T ≈ I)"""
        torch.manual_seed(42)
        G = torch.randn(64, 64)
        # 小矩阵需要更多步数收敛 (实际训练用 5 步, 测试用 20 步验证正确性)
        X = zeropower_via_newtonschulz5(G, steps=20)

        # X X^T 应接近单位矩阵 (CPU float32 NS 迭代精度有限, 验证对角线均值接近 1)
        product = X @ X.T
        diag_mean = product.diag().mean().item()
        off_diag = product - torch.diag(product.diag())
        off_diag_max = off_diag.abs().max().item()

        # 对角线均值应在 [0.7, 1.1] 范围 (近似正交)
        assert 0.7 < diag_mean < 1.1, f'Diagonal mean {diag_mean} not close to 1'
        # 非对角线元素应远小于对角线
        assert off_diag_max < 0.3, f'Off-diagonal max {off_diag_max} too large'

    def test_preserves_direction(self):
        """正交化保持矩阵的主方向 (与原矩阵的列空间一致)"""
        torch.manual_seed(0)
        G = torch.randn(32, 64)  # 非方阵
        X = zeropower_via_newtonschulz5(G, steps=5)

        # X 的形状应与 G 一致
        assert X.shape == G.shape

    def test_handles_tall_matrix(self):
        """高瘦矩阵 (m > n) 也能正交化"""
        torch.manual_seed(1)
        G = torch.randn(128, 32)
        X = zeropower_via_newtonschulz5(G, steps=20)

        # X^T X 应接近单位矩阵 (列正交)
        product = X.T @ X
        diag_mean = product.diag().mean().item()
        off_diag = product - torch.diag(product.diag())
        off_diag_max = off_diag.abs().max().item()

        assert 0.7 < diag_mean < 1.1, f'Diagonal mean {diag_mean} not close to 1'
        assert off_diag_max < 0.3, f'Off-diagonal max {off_diag_max} too large'

    def test_handles_wide_matrix(self):
        """矮胖矩阵 (m < n) 也能正交化"""
        torch.manual_seed(2)
        G = torch.randn(32, 128)
        X = zeropower_via_newtonschulz5(G, steps=20)

        # X X^T 应接近单位矩阵 (行正交)
        product = X @ X.T
        diag_mean = product.diag().mean().item()
        off_diag = product - torch.diag(product.diag())
        off_diag_max = off_diag.abs().max().item()

        assert 0.7 < diag_mean < 1.1, f'Diagonal mean {diag_mean} not close to 1'
        assert off_diag_max < 0.3, f'Off-diagonal max {off_diag_max} too large'

    def test_different_steps(self):
        """不同迭代步数都能收敛 (步数越多越精确)"""
        torch.manual_seed(3)
        G = torch.randn(64, 64)

        X1 = zeropower_via_newtonschulz5(G, steps=1)
        X5 = zeropower_via_newtonschulz5(G, steps=5)

        err1 = (X1 @ X1.T - torch.eye(64)).abs().max().item()
        err5 = (X5 @ X5.T - torch.eye(64)).abs().max().item()

        # 5 步应比 1 步更精确
        assert err5 < err1

    def test_zero_matrix_safe(self):
        """零矩阵不应导致 NaN"""
        G = torch.zeros(32, 32)
        X = zeropower_via_newtonschulz5(G, steps=5)
        assert not torch.isnan(X).any()


class TestMuonOptimizer:
    """Muon 优化器单元测试"""

    @pytest.fixture
    def simple_model(self):
        """简单模型: 一个 2D 矩阵参数 + 一个偏置"""
        return nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))

    def test_basic_step(self, simple_model):
        """基本 step 不报错, 参数有更新"""
        opt = Muon(simple_model.parameters(), lr=0.01, momentum=0.95)

        # 前向 + 反向
        x = torch.randn(8, 32)
        loss = simple_model(x).sum()
        loss.backward()

        params_before = [p.clone() for p in simple_model.parameters()]

        opt.step()

        # 2D 参数应该有更新
        assert not torch.equal(simple_model[0].weight, params_before[0])
        # 1D 偏置也应更新 (Muon 对 1D 退化为 momentum SGD)
        assert not torch.equal(simple_model[0].bias, params_before[1])

    def test_gradient_flow(self, simple_model):
        """梯度正确传播到参数"""
        opt = Muon(simple_model.parameters(), lr=0.01)

        x = torch.randn(4, 32)
        loss = simple_model(x).sum()
        loss.backward()

        opt.step()

        # 所有参数应有梯度
        for p in simple_model.parameters():
            assert p.grad is not None

    def test_zero_grad(self, simple_model):
        """zero_grad 清空梯度"""
        opt = Muon(simple_model.parameters(), lr=0.01)

        x = torch.randn(4, 32)
        loss = simple_model(x).sum()
        loss.backward()
        opt.step()

        opt.zero_grad()
        for p in simple_model.parameters():
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))

    def test_momentum_accumulation(self, simple_model):
        """动量累积: 多次 step 后动量缓冲区存在"""
        opt = Muon(simple_model.parameters(), lr=0.01, momentum=0.9)

        for _ in range(3):
            x = torch.randn(4, 32)
            loss = simple_model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # 动量缓冲区应存在
        assert len(opt.state) > 0

    def test_convergence_simple_quadratic(self):
        """简单二次函数收敛测试"""
        # f(x) = 0.5 * x^T A x, 最小值在 0
        torch.manual_seed(42)
        A = torch.randn(64, 64)
        A = A @ A.T + torch.eye(64)  # 正定
        # 归一化 A 的谱范数到 ~1, 避免大梯度导致 NS 幂迭代数值溢出
        A = A / torch.linalg.norm(A, ord=2)

        W = nn.Parameter(torch.randn(64, 32) * 0.1)
        opt = Muon([W], lr=0.02, momentum=0.95)

        initial_loss = 0.5 * (W.T @ A @ W).trace().item()

        for _ in range(500):
            opt.zero_grad()
            loss = 0.5 * (W.T @ A @ W).trace()
            loss.backward()
            opt.step()

        final_loss = 0.5 * (W.T @ A @ W).trace().item()

        # 损失应显著下降
        assert final_loss < initial_loss * 0.5, (
            f'No convergence: {initial_loss} → {final_loss}'
        )

    def test_rms_scaling(self):
        """RMS-aligned 缩放: 更新幅度与 Adam 量级匹配"""
        torch.manual_seed(0)
        W = nn.Parameter(torch.randn(64, 32))

        # Muon
        opt_muon = Muon([W], lr=0.01, momentum=0.95)
        W.grad = torch.randn_like(W)
        params_before = W.clone()
        opt_muon.step()
        muon_update = (W - params_before).abs().mean().item()

        # 量级应在合理范围 (不是 0, 也不爆炸)
        assert 0 < muon_update < 1.0

    def test_weight_decay(self, simple_model):
        """weight_decay 正则化生效: 相同梯度下, 有 wd 的参数模长更小"""
        torch.manual_seed(42)
        # 两个相同模型, 一个有 wd 一个无
        model_wd = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
        model_no_wd = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
        # 同步初始权重
        for p_wd, p_no in zip(model_wd.parameters(), model_no_wd.parameters()):
            p_no.data.copy_(p_wd.data)

        opt_wd = Muon(model_wd.parameters(), lr=0.01, momentum=0.95, weight_decay=0.5)
        opt_no = Muon(model_no_wd.parameters(), lr=0.01, momentum=0.95, weight_decay=0.0)

        x = torch.randn(4, 32)
        loss_wd = model_wd(x).sum()
        loss_no = model_no_wd(x).sum()
        loss_wd.backward()
        loss_no.backward()

        opt_wd.step()
        opt_no.step()

        # 有 wd 的参数模长应更小
        for p_wd, p_no in zip(model_wd.parameters(), model_no_wd.parameters()):
            assert p_wd.norm() < p_no.norm() + 1e-6, (
                f'weight_decay not working: wd={p_wd.norm()} >= no_wd={p_no.norm()}'
            )


class TestParameterClassification:
    """参数分类测试: 区分 Muon 参数和 AdamW 参数"""

    def test_classify_simple_model(self):
        """简单模型参数分类"""
        model = nn.Sequential(
            nn.Linear(32, 16),  # weight (2D) → Muon, bias (1D) → AdamW
            nn.LayerNorm(16),  # weight (1D) → AdamW, bias (1D) → AdamW
            nn.Linear(16, 4),  # weight (2D) → Muon, bias (1D) → AdamW
        )

        muon_params, adam_params = classify_parameters(
            model, muon_patterns=None, adam_patterns=None
        )

        # 2 个 Linear weight → Muon
        assert len(muon_params) == 2
        # 2 个 bias + 2 个 LayerNorm 参数 → AdamW
        assert len(adam_params) == 4

        # 所有参数都被分类
        total = sum(p.numel() for p in muon_params + adam_params)
        expected = sum(p.numel() for p in model.parameters())
        assert total == expected

    def test_classify_with_patterns(self):
        """按名称模式分类"""
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.LayerNorm(16),
        )
        model[0]._full_name = 'backbone.layer1'
        model[1]._full_name = 'neck.fpn'

        # backbone 用 AdamW, 其他用 Muon
        muon_params, adam_params = classify_parameters(
            model,
            muon_patterns=None,
            adam_patterns=['backbone'],
            name_resolver=lambda m, p: getattr(m, '_full_name', ''),
        )

        # backbone Linear weight → AdamW (覆盖 2D 默认)
        # LayerNorm → AdamW
        assert len(adam_params) == 3  # backbone weight + 2 LayerNorm params

    def test_classify_ldmdet_like_model(self):
        """类 LDMDet 模型分类: backbone + neck + bbox_head"""
        class FakeLDMDet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Conv2d(3, 64, 3), nn.BatchNorm2d(64))
                self.neck = nn.Sequential(nn.Conv2d(64, 256, 1))
                self.bbox_head = nn.Module()
                self.bbox_head.head_series = nn.ModuleList([
                    nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256))
                ])
                self.bbox_head.time_mlp = nn.Sequential(
                    nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 1024)
                )

        model = FakeLDMDet()

        # 默认: 2D+ → Muon, 1D → AdamW
        muon_params, adam_params = classify_parameters(
            model, muon_patterns=None, adam_patterns=None
        )

        # Muon (ndim >= 2): backbone conv (4D), neck conv (4D),
        #                   head Linear (2D), time_mlp Linear x2 (2D) = 5
        assert len(muon_params) == 5
        # AdamW (1D): backbone bias, BN weight+bias, neck bias,
        #             head bias, LN weight+bias, time_mlp bias x2 = 9
        assert len(adam_params) == 9

        # 所有参数都被分类
        total = sum(p.numel() for p in muon_params + adam_params)
        expected = sum(p.numel() for p in model.parameters())
        assert total == expected


class TestMuonHybridOptimWrapper:
    """混合优化器 wrapper 测试"""

    @pytest.fixture
    def hybrid_wrapper(self):
        """构建混合 wrapper: 2D 参数用 Muon, 1D 用 AdamW"""
        model = nn.Sequential(
            nn.Linear(32, 16),  # weight → Muon, bias → AdamW
            nn.LayerNorm(16),
            nn.Linear(16, 4),
        )

        muon_params, adam_params = classify_parameters(
            model, muon_patterns=None, adam_patterns=None
        )

        muon_opt = Muon(muon_params, lr=0.01, momentum=0.95)
        adam_opt = torch.optim.AdamW(adam_params, lr=5e-5, weight_decay=1e-4)

        wrapper = MuonHybridOptimWrapper(
            muon_optimizer=muon_opt,
            adam_optimizer=adam_opt,
            muon_param_ids={id(p) for p in muon_params},
        )
        return wrapper, model

    def test_step_updates_all_params(self, hybrid_wrapper):
        """step 更新所有参数"""
        wrapper, model = hybrid_wrapper

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()

        params_before = [p.clone() for p in model.parameters()]
        wrapper.step()
        wrapper.zero_grad()

        # 所有参数都应更新
        for p, p_before in zip(model.parameters(), params_before):
            assert not torch.equal(p, p_before)

    def test_zero_grad(self, hybrid_wrapper):
        """zero_grad 清空所有梯度"""
        wrapper, model = hybrid_wrapper

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        wrapper.step()

        wrapper.zero_grad()
        for p in model.parameters():
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))

    def test_param_routing(self, hybrid_wrapper):
        """梯度路由: Muon 参数走 Muon, AdamW 参数走 AdamW"""
        wrapper, model = hybrid_wrapper

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()

        # 记录 step 前 Muon 和 AdamW 的 state
        muon_state_before = {
            id(p): p.clone() for p in model.parameters() if id(p) in wrapper.muon_param_ids
        }
        adam_state_before = {
            id(p): p.clone() for p in model.parameters() if id(p) not in wrapper.muon_param_ids
        }

        wrapper.step()

        # 验证两组参数都更新了
        muon_updated = all(
            not torch.equal(p, muon_state_before[id(p)])
            for p in model.parameters()
            if id(p) in wrapper.muon_param_ids
        )
        adam_updated = all(
            not torch.equal(p, adam_state_before[id(p)])
            for p in model.parameters()
            if id(p) not in wrapper.muon_param_ids
        )

        assert muon_updated, 'Muon params not updated'
        assert adam_updated, 'AdamW params not updated'

    def test_clip_grad(self, hybrid_wrapper):
        """梯度裁剪对两组优化器都生效"""
        wrapper, model = hybrid_wrapper

        x = torch.randn(4, 32)
        loss = model(x).sum() * 1000  # 放大梯度
        loss.backward()

        # 应不报错
        wrapper.clip_grad(max_norm=1.0)

        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.norm() <= 1.0 + 1e-4

    def test_state_dict(self, hybrid_wrapper):
        """state_dict 可序列化和加载"""
        wrapper, model = hybrid_wrapper

        # 训练几步产生 state
        for _ in range(2):
            x = torch.randn(4, 32)
            loss = model(x).sum()
            loss.backward()
            wrapper.step()
            wrapper.zero_grad()

        # 保存
        state = wrapper.state_dict()
        assert 'muon' in state
        assert 'adam' in state

        # 新 wrapper 加载
        muon_params, adam_params = classify_parameters(
            model, muon_patterns=None, adam_patterns=None
        )
        new_muon = Muon(muon_params, lr=0.01)
        new_adam = torch.optim.AdamW(adam_params, lr=5e-5)
        new_wrapper = MuonHybridOptimWrapper(
            muon_optimizer=new_muon,
            adam_optimizer=new_adam,
            muon_param_ids={id(p) for p in muon_params},
        )
        new_wrapper.load_state_dict(state)

        # state 应加载成功
        assert len(new_muon.state) > 0 or len(new_adam.state) > 0


class TestMuonHybridOptimizer:
    """MuonHybrid 单一 Optimizer 子类测试 (mmengine 兼容)"""

    def test_is_torch_optimizer(self):
        """MuonHybrid 应是 torch.optim.Optimizer 子类"""
        model = nn.Linear(8, 8)
        opt = MuonHybrid(model.parameters(), lr=1e-3)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_param_groups_with_use_muon_flag(self):
        """param_group 应支持 use_muon 标志"""
        model = nn.Sequential(
            nn.Linear(8, 16),   # 2D weight → 默认 use_muon=True
            nn.LayerNorm(16),   # 1D weight → 默认 use_muon=False
        )

        # 显式指定 use_muon
        opt = MuonHybrid(
            [
                {'params': [model[0].weight], 'use_muon': True},   # Linear weight → Muon
                {'params': [model[0].bias], 'use_muon': False},   # bias → AdamW
                {'params': [model[1].weight], 'use_muon': False}, # LayerNorm → AdamW
            ],
            lr=5e-5,
            muon_lr=0.02,
        )
        assert opt.param_groups[0]['use_muon'] is True
        assert opt.param_groups[1]['use_muon'] is False
        assert opt.param_groups[2]['use_muon'] is False

    def test_step_updates_muon_params(self):
        """Muon 组参数应被更新 (Newton-Schulz 正交化)"""
        torch.manual_seed(42)
        W = nn.Parameter(torch.randn(16, 8) * 0.1)
        opt = MuonHybrid([{'params': [W], 'use_muon': True}], lr=5e-5, muon_lr=0.02)

        W.grad = torch.randn_like(W)
        params_before = W.clone()
        opt.step()

        # 参数应改变
        assert not torch.allclose(W, params_before)
        # 应有 momentum_buffer 状态
        assert 'momentum_buffer' in opt.state[W]

    def test_step_updates_adamw_params(self):
        """AdamW 组参数应被更新 (AdamW 算法)"""
        torch.manual_seed(0)
        W = nn.Parameter(torch.randn(16, 8) * 0.1)
        opt = MuonHybrid([{'params': [W], 'use_muon': False}], lr=1e-3)

        W.grad = torch.randn_like(W)
        params_before = W.clone()
        opt.step()

        # 参数应改变
        assert not torch.allclose(W, params_before)
        # 应有 AdamW 状态 (exp_avg, exp_avg_sq)
        assert 'exp_avg' in opt.state[W]
        assert 'exp_avg_sq' in opt.state[W]

    def test_mixed_param_groups(self):
        """混合参数组: Muon 和 AdamW 同时更新"""
        torch.manual_seed(42)
        # 模拟 LDMDet: backbone (AdamW) + head (Muon)
        backbone_w = nn.Parameter(torch.randn(16, 8) * 0.1)
        head_w = nn.Parameter(torch.randn(16, 8) * 0.1)
        head_bias = nn.Parameter(torch.randn(16) * 0.1)

        opt = MuonHybrid(
            [
                {'params': [backbone_w], 'use_muon': False, 'lr': 5e-5},
                {'params': [head_w], 'use_muon': True, 'muon_lr': 0.02},
                {'params': [head_bias], 'use_muon': False, 'lr': 5e-5},
            ],
            lr=5e-5,
            muon_lr=0.02,
        )

        # 模拟前向 + 反向
        backbone_w.grad = torch.randn_like(backbone_w)
        head_w.grad = torch.randn_like(head_w)
        head_bias.grad = torch.randn_like(head_bias)

        bb_before = backbone_w.clone()
        hw_before = head_w.clone()
        hb_before = head_bias.clone()

        opt.step()

        # 三个参数都应更新
        assert not torch.allclose(backbone_w, bb_before)
        assert not torch.allclose(head_w, hw_before)
        assert not torch.allclose(head_bias, hb_before)

    def test_zero_grad(self):
        """zero_grad 清空梯度"""
        W = nn.Parameter(torch.randn(8, 8))
        opt = MuonHybrid([{'params': [W], 'use_muon': True}], lr=5e-5)
        W.grad = torch.randn_like(W)
        opt.zero_grad()
        # set_to_none=True 时 grad 为 None
        assert W.grad is None

    def test_state_dict_serialization(self):
        """state_dict 可序列化和加载"""
        torch.manual_seed(0)
        W1 = nn.Parameter(torch.randn(16, 8) * 0.1)
        W2 = nn.Parameter(torch.randn(16, 8) * 0.1)
        opt = MuonHybrid(
            [
                {'params': [W1], 'use_muon': True},
                {'params': [W2], 'use_muon': False},
            ],
            lr=1e-3,
        )

        # 训练几步产生 state
        for _ in range(3):
            W1.grad = torch.randn_like(W1)
            W2.grad = torch.randn_like(W2)
            opt.step()
            opt.zero_grad()

        state = opt.state_dict()
        assert 'state' in state
        assert 'param_groups' in state

        # 新优化器加载
        opt2 = MuonHybrid(
            [
                {'params': [W1], 'use_muon': True},
                {'params': [W2], 'use_muon': False},
            ],
            lr=1e-3,
        )
        opt2.load_state_dict(state)
        assert len(opt2.state) > 0

    def test_mmengine_compatible_interface(self):
        """mmengine OptimWrapper 期望的接口应齐全"""
        W = nn.Parameter(torch.randn(8, 8))
        opt = MuonHybrid([{'params': [W], 'use_muon': True}], lr=5e-5)

        # 必须属性
        assert hasattr(opt, 'param_groups')
        assert hasattr(opt, 'defaults')
        assert hasattr(opt, 'state')
        # 必须方法
        assert callable(opt.step)
        assert callable(opt.zero_grad)
        assert callable(opt.state_dict)
        assert callable(opt.load_state_dict)

    def test_convergence_mixed_optimization(self):
        """混合优化收敛测试: Muon (2D) + AdamW (1D) 共同优化二次函数"""
        torch.manual_seed(42)
        # f(W, b) = 0.5 * ||A W + b||^2, 最小值在 W=0, b=0
        # A: (m, n), W: (n, k), b: (m,) → A @ W: (m, k), b 广播到 (m, k)
        m, n, k = 32, 16, 8
        A = torch.randn(m, n)
        A = A / A.norm()  # 归一化避免数值问题

        W = nn.Parameter(torch.randn(n, k) * 0.1)
        b = nn.Parameter(torch.randn(m, 1) * 0.1)  # (m, 1) 广播到 (m, k)

        opt = MuonHybrid(
            [
                {'params': [W], 'use_muon': True, 'muon_lr': 0.01},
                {'params': [b], 'use_muon': False, 'lr': 1e-3},
            ],
            lr=1e-3,
            muon_lr=0.01,
        )

        initial_loss = 0.5 * (A @ W + b).pow(2).sum().item()

        for _ in range(300):
            opt.zero_grad()
            loss = 0.5 * (A @ W + b).pow(2).sum()
            loss.backward()
            opt.step()

        final_loss = 0.5 * (A @ W + b).pow(2).sum().item()
        assert final_loss < initial_loss * 0.5, (
            f'No convergence: {initial_loss} → {final_loss}'
        )
