"""公共诊断工具测试 — 统计计算、数值健康度、Hook 基础行为.

测试文件划分原则:
- test_diagnostics_common.py: 公共统计工具 + TrainingDiagnosticsHook 基础行为
- test_diagnostics_coupling.py: 方向一耦合诊断 (可独立清理)
- test_diagnostics_count.py: 方向二计数诊断 (可独立清理)
- test_diagnostics_snr.py: 方向三 SNR 诊断 (可独立清理)
"""

import torch
import torch.nn as nn
import pytest

from ldmdet.diagnostics.stats import (
    compute_weight_stats,
    compute_grad_stats,
    compute_activation_stats,
    check_numerical_health,
    ModuleGroup,
    group_modules_by_prefix,
)


# ──────────────────────────────────────────────
# compute_weight_stats: 权重统计
# ──────────────────────────────────────────────

class TestWeightStats:
    """测试权重统计计算."""

    def test_basic_stats(self):
        """基本统计: mean, std, norm, numel."""
        layer = nn.Linear(4, 3)
        with torch.no_grad():
            layer.weight.fill_(2.0)
            layer.bias.fill_(1.0)

        stats = compute_weight_stats(layer)

        # weight: [3, 4] 全 2.0, bias: [3] 全 1.0
        # 合并: 12 个 2.0 + 3 个 1.0 = 15 个值
        # mean = (12*2 + 3*1) / 15 = 27/15 = 1.8
        # norm = sqrt(12*4 + 3*1) = sqrt(51) ≈ 7.141
        assert stats['mean'] == pytest.approx(1.8, abs=1e-4)
        assert stats['norm'] == pytest.approx(51 ** 0.5, abs=1e-4)
        assert stats['numel'] == 15
        assert stats['has_nan'] is False
        assert stats['has_inf'] is False

    def test_dead_neurons(self):
        """死神经元检测: 某输出维度权重全零."""
        layer = nn.Linear(4, 3)
        with torch.no_grad():
            layer.weight[0].fill_(0.0)  # 第 0 个输出维度死神经元
            layer.bias[0].fill_(0.0)
            layer.weight[1:].fill_(1.0)
            layer.bias[1:].fill_(0.5)

        stats = compute_weight_stats(layer)
        # 3 个输出维度, 1 个死 → dead_ratio = 1/3
        assert stats['dead_ratio'] == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_nan_detection(self):
        """NaN 检测."""
        layer = nn.Linear(4, 2)
        with torch.no_grad():
            layer.weight[0, 0] = float('nan')

        stats = compute_weight_stats(layer)
        assert stats['has_nan'] is True

    def test_inf_detection(self):
        """Inf 检测."""
        layer = nn.Linear(4, 2)
        with torch.no_grad():
            layer.weight[0, 0] = float('inf')

        stats = compute_weight_stats(layer)
        assert stats['has_inf'] is True


# ──────────────────────────────────────────────
# compute_grad_stats: 梯度统计
# ──────────────────────────────────────────────

class TestGradStats:
    """测试梯度统计计算."""

    def test_basic_stats(self):
        """基本梯度统计."""
        layer = nn.Linear(4, 2)
        # 手动设置梯度
        layer.weight.grad = torch.ones_like(layer.weight) * 0.5
        layer.bias.grad = torch.ones_like(layer.bias) * 0.5

        stats = compute_grad_stats(layer)

        assert stats['mean'] == pytest.approx(0.5, abs=1e-4)
        assert stats['norm'] == pytest.approx((8 * 0.25 + 2 * 0.25) ** 0.5, abs=1e-4)
        assert stats['numel'] == 10
        assert stats['has_nan'] is False

    def test_zero_grad_ratio(self):
        """零梯度比例."""
        layer = nn.Linear(4, 2)
        layer.weight.grad = torch.zeros(2, 4)
        layer.weight.grad[0, 0] = 1.0  # 只有 1 个非零
        layer.bias.grad = torch.zeros(2)

        stats = compute_grad_stats(layer)
        # 10 个值, 9 个零 → zero_ratio = 0.9
        assert stats['zero_ratio'] == pytest.approx(0.9, abs=1e-4)

    def test_no_grad(self):
        """无梯度时应返回 zero_ratio=1.0, norm=0."""
        layer = nn.Linear(4, 2)
        # 不设置 grad
        stats = compute_grad_stats(layer)
        assert stats['norm'] == 0.0
        assert stats['zero_ratio'] == 1.0

    def test_grad_to_weight_ratio(self):
        """梯度/权重比 (学习效率指标)."""
        layer = nn.Linear(2, 1)
        with torch.no_grad():
            layer.weight.fill_(2.0)
        layer.weight.grad = torch.ones_like(layer.weight) * 0.1

        stats = compute_grad_stats(layer, layer.weight.data)
        # grad_mean=0.1, weight_mean=2.0 → ratio=0.05
        assert stats['grad_to_weight_ratio'] == pytest.approx(0.05, abs=1e-4)


# ──────────────────────────────────────────────
# compute_activation_stats: 激活统计
# ──────────────────────────────────────────────

class TestActivationStats:
    """测试激活统计计算."""

    def test_basic_stats(self):
        """基本激活统计."""
        activation = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stats = compute_activation_stats(activation)

        assert stats['mean'] == pytest.approx(3.5, abs=1e-4)
        assert stats['numel'] == 6
        assert stats['has_nan'] is False

    def test_dead_activation_ratio(self):
        """死激活比例 (值为 0 的元素)."""
        activation = torch.tensor([[0.0, 0.0, 1.0], [0.0, 2.0, 3.0]])
        stats = compute_activation_stats(activation)
        # 6 个值, 3 个零 → dead_ratio = 0.5
        assert stats['dead_ratio'] == pytest.approx(0.5, abs=1e-4)

    def test_saturation(self):
        """饱和度 (|x| > threshold 的比例)."""
        activation = torch.tensor([0.0, 0.5, 1.0, 10.0])
        stats = compute_activation_stats(activation, saturation_threshold=5.0)
        # 4 个值, 1 个 > 5.0 → saturation = 0.25
        assert stats['saturation'] == pytest.approx(0.25, abs=1e-4)


# ──────────────────────────────────────────────
# check_numerical_health: 数值健康度
# ──────────────────────────────────────────────

class TestNumericalHealth:
    """测试数值健康度检查."""

    def test_healthy_tensor(self):
        """健康张量."""
        x = torch.randn(10, 10)
        health = check_numerical_health(x)
        assert health['has_nan'] is False
        assert health['has_inf'] is False
        assert health['is_healthy'] is True

    def test_nan_tensor(self):
        """含 NaN 张量."""
        x = torch.randn(10, 10)
        x[0, 0] = float('nan')
        health = check_numerical_health(x)
        assert health['has_nan'] is True
        assert health['is_healthy'] is False

    def test_inf_tensor(self):
        """含 Inf 张量."""
        x = torch.randn(10, 10)
        x[0, 0] = float('inf')
        health = check_numerical_health(x)
        assert health['has_inf'] is True
        assert health['is_healthy'] is False

    def test_extreme_values(self):
        """极端值检测 (非 NaN/Inf 但过大)."""
        x = torch.randn(10, 10)
        x[0, 0] = 1e10  # 极端大值
        health = check_numerical_health(x, extreme_threshold=1e6)
        assert health['has_extreme'] is True
        assert health['is_healthy'] is False


# ──────────────────────────────────────────────
# group_modules_by_prefix: 模块分组
# ──────────────────────────────────────────────

class TestModuleGrouping:
    """测试模块分组工具."""

    def test_group_by_prefix(self):
        """按前缀分组模块."""
        model = nn.Sequential(
            nn.Linear(10, 10),  # 0
            nn.Sequential(
                nn.Linear(10, 10),  # 1.0
                nn.Linear(10, 10),  # 1.1
            ),
            nn.Linear(10, 10),  # 2
        )

        groups = group_modules_by_prefix(model, prefixes={'0': 'backbone', '1': 'head', '2': 'head'})

        assert 'backbone' in groups
        assert 'head' in groups
        assert len(groups['backbone'].modules) == 1
        # head 组: 1, 1.0, 1.1, 2 (前缀 '1' 匹配 '1' 和 '1.0', '1.1'; '2' 匹配 '2')
        assert len(groups['head'].modules) >= 3

    def test_module_group_dataclass(self):
        """ModuleGroup 数据结构."""
        group = ModuleGroup(name='head', modules=[nn.Linear(10, 10)])
        assert group.name == 'head'
        assert len(group.modules) == 1
