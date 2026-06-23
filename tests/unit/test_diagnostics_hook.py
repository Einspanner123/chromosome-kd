"""TrainingDiagnosticsHook 测试 — 基础监控行为.

测试内容:
- Hook 注册和初始化
- 权重/梯度统计采集 (按模块组聚合)
- 激活统计采集 (前向 hook)
- 数值健康度检查
- SwanLab 上传 (mock)
- 采样频率控制
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch

from ldmdet.diagnostics.hooks import TrainingDiagnosticsHook
from ldmdet.diagnostics.stats import ModuleGroup


# ──────────────────────────────────────────────
# 测试辅助
# ──────────────────────────────────────────────

class _MockRunner:
    """模拟 mmengine Runner 的最小接口."""

    def __init__(self, model, iter=0, epoch=0, rank=0):
        self.model = model
        self.iter = iter
        self.epoch = epoch
        self.rank = rank
        self.logger = MagicMock()


class _SimpleModel(nn.Module):
    """简单模型用于测试."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        self.head = nn.Sequential(nn.Linear(10, 5), nn.ReLU())

    def forward(self, x):
        return self.head(self.backbone(x))


# ──────────────────────────────────────────────
# Hook 初始化
# ──────────────────────────────────────────────

class TestHookInit:
    """测试 Hook 初始化."""

    def test_default_config(self):
        """默认配置: 标准频率, 启用所有监控."""
        hook = TrainingDiagnosticsHook()
        assert hook.weight_grad_interval == 100
        assert hook.activation_interval == 500
        assert hook.log_weights is True
        assert hook.log_grads is True
        assert hook.log_activations is True
        assert hook.log_numerical_health is True

    def test_custom_config(self):
        """自定义配置."""
        hook = TrainingDiagnosticsHook(
            weight_grad_interval=50,
            activation_interval=200,
            log_activations=False,
        )
        assert hook.weight_grad_interval == 50
        assert hook.activation_interval == 200
        assert hook.log_activations is False

    def test_module_prefixes(self):
        """模块前缀配置."""
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone', 'head.': 'head'},
        )
        assert 'backbone.' in hook.module_prefixes
        assert hook.module_prefixes['backbone.'] == 'backbone'


# ──────────────────────────────────────────────
# 权重/梯度统计采集
# ──────────────────────────────────────────────

class TestWeightGradCollection:
    """测试权重和梯度统计采集."""

    def test_collect_weight_stats_by_group(self):
        """按模块组采集权重统计."""
        model = _SimpleModel()
        runner = _MockRunner(model=model, iter=100)
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone', 'head.': 'head'},
            log_activations=False,  # 简化测试
        )

        with patch('ldmdet.diagnostics.hooks.swanlab_log') as mock_log:
            hook.after_train_iter(runner, batch_idx=100)

        # 应记录 backbone 和 head 两组的权重统计
        assert mock_log.called
        logged_data = mock_log.call_args[0][0]
        assert any('weights/backbone' in k for k in logged_data)
        assert any('weights/head' in k for k in logged_data)

    def test_collect_grad_stats(self):
        """采集梯度统计."""
        model = _SimpleModel()
        # 模拟反向传播
        x = torch.randn(2, 10)
        y = model(x)
        y.sum().backward()

        runner = _MockRunner(model=model, iter=100)
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone', 'head.': 'head'},
            log_activations=False,
        )

        with patch('ldmdet.diagnostics.hooks.swanlab_log') as mock_log:
            hook.after_train_iter(runner, batch_idx=100)

        logged_data = mock_log.call_args[0][0]
        assert any('grads/backbone' in k for k in logged_data)
        assert any('grads/head' in k for k in logged_data)

    def test_interval_control(self):
        """采样频率控制: 非采样点不记录."""
        model = _SimpleModel()
        runner = _MockRunner(model=model, iter=50)
        hook = TrainingDiagnosticsHook(
            weight_grad_interval=100,
            module_prefixes={'backbone.': 'backbone'},
            log_activations=False,
        )

        with patch('ldmdet.diagnostics.hooks.swanlab_log') as mock_log:
            hook.after_train_iter(runner, batch_idx=50)  # iter=50, 不是 100 的倍数

        assert not mock_log.called


# ──────────────────────────────────────────────
# 数值健康度
# ──────────────────────────────────────────────

class TestNumericalHealthCheck:
    """测试数值健康度检查."""

    def test_detect_nan_in_weights(self):
        """检测权重中的 NaN."""
        model = _SimpleModel()
        with torch.no_grad():
            model.backbone[0].weight[0, 0] = float('nan')

        runner = _MockRunner(model=model, iter=100)
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone'},
            log_activations=False,
        )

        with patch('ldmdet.diagnostics.hooks.swanlab_log') as mock_log:
            hook.after_train_iter(runner, batch_idx=100)

        logged_data = mock_log.call_args[0][0]
        # 应记录 has_nan=True
        assert any('nan' in k and v for k, v in logged_data.items())


# ──────────────────────────────────────────────
# 激活统计采集
# ──────────────────────────────────────────────

class TestActivationCollection:
    """测试激活统计采集."""

    def test_register_forward_hooks(self):
        """before_run 注册前向 hook."""
        model = _SimpleModel()
        runner = _MockRunner(model=model)
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone', 'head.': 'head'},
            activation_layers=['backbone.0', 'head.0'],
        )

        hook.before_run(runner)

        # 应注册了前向 hook
        assert len(hook._activation_hooks) > 0
        # 清理
        hook._remove_hooks()

    def test_collect_activation_stats(self):
        """采集激活统计."""
        model = _SimpleModel()
        runner = _MockRunner(model=model, iter=500)
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone'},
            activation_layers=['backbone.0'],
            weight_grad_interval=10000,  # 避免权重统计干扰
        )

        hook.before_run(runner)

        # 前向传播触发 hook
        x = torch.randn(2, 10)
        _ = model(x)

        with patch('ldmdet.diagnostics.hooks.swanlab_log') as mock_log:
            hook.after_train_iter(runner, batch_idx=500)

        logged_data = mock_log.call_args[0][0]
        assert any('acts/backbone' in k for k in logged_data)

        hook._remove_hooks()


# ──────────────────────────────────────────────
# SwanLab 上传
# ──────────────────────────────────────────────

class TestSwanLabUpload:
    """测试 SwanLab 上传."""

    def test_log_scalars(self):
        """标量指标上传."""
        model = _SimpleModel()
        runner = _MockRunner(model=model, iter=100)
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone'},
            log_activations=False,
        )

        with patch('ldmdet.diagnostics.hooks.swanlab_log') as mock_log:
            hook.after_train_iter(runner, batch_idx=100)

        # 验证调用参数
        mock_log.assert_called_once()
        args = mock_log.call_args
        assert args[0][1] == 100  # step=iter


# ──────────────────────────────────────────────
# 训练动态
# ──────────────────────────────────────────────

class TestTrainingDynamics:
    """测试训练动态监控 (损失分解)."""

    def test_log_loss_breakdown(self):
        """损失分解上传."""
        model = _SimpleModel()
        runner = _MockRunner(model=model, iter=1)
        hook = TrainingDiagnosticsHook(
            module_prefixes={'backbone.': 'backbone'},
            log_activations=False,
            weight_grad_interval=10000,  # 避免权重统计干扰
        )

        # 模拟 outputs (含损失)
        outputs = {
            'loss_cls': torch.tensor(0.5),
            'loss_bbox': torch.tensor(0.3),
            'loss_giou': torch.tensor(0.2),
        }

        with patch('ldmdet.diagnostics.hooks.swanlab_log') as mock_log:
            hook.after_train_iter(runner, batch_idx=1, outputs=outputs)

        logged_data = mock_log.call_args[0][0]
        assert 'dynamics/loss_cls' in logged_data
        assert 'dynamics/loss_bbox' in logged_data
        assert 'dynamics/loss_giou' in logged_data
        assert 'dynamics/loss_total' in logged_data
