"""测试 ldmdet.diagnostics.instrumentation — 运行时探针 (Probe)

覆盖:
- _compute_histogram / _compute_tensor_stats 工具函数
- Probe 生命周期 (enable/disable, enabled/inference_enabled)
- 频率控制 (train_interval, _should_collect)
- 训练缓冲区 (record_scalar/tensor_stats/detail)
- 推理缓冲区 (record_inference_*, on_inference_begin/end)
- 梯度 hook (register_grad_hooks, _make_grad_hook, backward 触发)
- 缓冲区隔离 (train vs inference)
- 未启用时 no-op 行为
"""

import torch
import torch.nn as nn
import pytest

from ldmdet.diagnostics.instrumentation import (
    probe,
    _Probe,
    _compute_histogram,
    _compute_tensor_stats,
    _HIST_BINS,
    _HIST_MAX_ELEMENTS,
)


# ============================================================
# 共享 fixtures
# ============================================================

@pytest.fixture(autouse=True)
def reset_probe():
    """每个测试前后重置 Probe 单例状态, 避免测试间污染."""
    probe.disable()
    yield
    probe.disable()


class _MockBBoxHead(nn.Module):
    """模拟 DiffusionDetHead 的子模块结构 (供梯度 hook 测试)."""

    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        self.head_series = nn.ModuleList([
            nn.Linear(8, 8) for _ in range(3)
        ])

    def forward(self, x):
        for h in self.head_series:
            x = h(x)
        x = self.time_mlp(x)
        return x


class _MockModel(nn.Module):
    """模拟完整检测器结构 (backbone/neck/bbox_head)."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        self.neck = nn.Linear(8, 8)
        self.bbox_head = _MockBBoxHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.bbox_head(x)
        return x


# ============================================================
# _compute_tensor_stats
# ============================================================

class TestComputeTensorStats:
    def test_normal_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = _compute_tensor_stats(t)
        assert stats['mean'] == pytest.approx(2.5)
        assert stats['min'] == 1.0
        assert stats['max'] == 4.0
        assert stats['norm'] == pytest.approx(torch.tensor([1.0, 2.0, 3.0, 4.0]).norm(2).item())
        assert stats['std'] == pytest.approx(t.std(unbiased=True).item())

    def test_single_element(self):
        """单元素张量 std 应为 0 (避免 numel=1 时 .std() 报错)."""
        t = torch.tensor([5.0])
        stats = _compute_tensor_stats(t)
        assert stats['mean'] == 5.0
        assert stats['std'] == 0.0
        assert stats['min'] == 5.0
        assert stats['max'] == 5.0
        assert stats['norm'] == 5.0

    def test_empty_tensor(self):
        """空张量返回全 0 统计."""
        t = torch.tensor([])
        stats = _compute_tensor_stats(t)
        assert stats == {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'norm': 0.0}

    def test_multi_dim_tensor(self):
        """多维张量应展平后统计."""
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        stats = _compute_tensor_stats(t)
        assert stats['mean'] == pytest.approx(2.5)
        assert stats['min'] == 0.0
        assert stats['max'] == 5.0

    def test_detach_safety(self):
        """输入带梯度的张量不应报错."""
        t = torch.randn(4, requires_grad=True)
        stats = _compute_tensor_stats(t)
        assert 'mean' in stats

    def test_returns_floats(self):
        """所有值应为 Python float (避免 tensor 泄漏)."""
        t = torch.randn(10)
        stats = _compute_tensor_stats(t)
        for v in stats.values():
            assert isinstance(v, float)


# ============================================================
# _compute_histogram
# ============================================================

class TestComputeHistogram:
    def test_normal_tensor(self):
        t = torch.randn(1000)
        hist = _compute_histogram(t, bins=_HIST_BINS)
        assert hist is not None
        assert len(hist['hist']) == _HIST_BINS
        assert len(hist['bin_edges']) == _HIST_BINS + 1
        assert hist['min'] == pytest.approx(t.min().item())
        assert hist['max'] == pytest.approx(t.max().item())
        assert hist['mean'] == pytest.approx(t.mean().item())
        # 归一化: 直方图总和接近 1
        assert sum(hist['hist']) == pytest.approx(1.0, abs=1e-5)

    def test_empty_tensor(self):
        """空张量返回 None."""
        t = torch.tensor([])
        assert _compute_histogram(t) is None

    def test_constant_tensor(self):
        """常数张量 (min==max) 不应报错."""
        t = torch.full((100,), 3.14)
        hist = _compute_histogram(t, bins=8)
        assert hist is not None
        assert len(hist['hist']) == 8
        # 第一个桶应为 1.0, 其余为 0
        assert hist['hist'][0] == 1.0
        assert all(h == 0.0 for h in hist['hist'][1:])
        assert hist['min'] == pytest.approx(3.14)
        assert hist['max'] == pytest.approx(3.14)

    def test_bin_count_custom(self):
        t = torch.randn(100)
        hist = _compute_histogram(t, bins=16)
        assert hist is not None
        assert len(hist['hist']) == 16
        assert len(hist['bin_edges']) == 17

    def test_large_tensor_downsampling(self):
        """超过 _HIST_MAX_ELEMENTS 的张量应下采样但不报错."""
        n = _HIST_MAX_ELEMENTS + 1000
        t = torch.randn(n)
        hist = _compute_histogram(t)
        assert hist is not None
        assert len(hist['hist']) == _HIST_BINS
        # 下采样后 min/max 可能略窄于原张量, 但应在合理范围
        assert hist['mean'] == pytest.approx(t.mean().item(), abs=0.1)

    def test_single_element(self):
        """单元素张量 (常数) 不应报错."""
        t = torch.tensor([7.0])
        hist = _compute_histogram(t)
        assert hist is not None
        assert hist['hist'][0] == 1.0


# ============================================================
# Probe 生命周期
# ============================================================

class TestProbeLifecycle:
    def test_initial_state_disabled(self):
        """新创建的 Probe 默认未启用."""
        p = _Probe()
        assert p.enabled is False
        assert p.inference_enabled is False

    def test_enable_default(self):
        p = _Probe()
        p.enable()
        assert p.enabled is True
        assert p.inference_enabled is True

    def test_enable_custom_args(self):
        p = _Probe()
        p.enable(train_interval=50, inference_enabled=False)
        assert p.enabled is True
        assert p.inference_enabled is False

    def test_disable(self):
        p = _Probe()
        p.enable()
        assert p.enabled is True
        p.disable()
        assert p.enabled is False
        assert p.inference_enabled is False

    def test_inference_enabled_requires_enabled(self):
        """inference_enabled 应同时依赖 _enabled 和 _inference_enabled."""
        p = _Probe()
        p.enable(inference_enabled=True)
        assert p.inference_enabled is True
        # disable 后即使 _inference_enabled=True, 也不应启用
        p.disable()
        assert p.inference_enabled is False

    def test_singleton_is_shared(self):
        """probe 全局单例应与直接 import 的实例一致."""
        from ldmdet.diagnostics.instrumentation import probe as probe2
        assert probe is probe2


# ============================================================
# 频率控制 (训练)
# ============================================================

class TestTrainFrequencyControl:
    def test_collect_at_interval_boundary(self):
        """step % interval == 0 时应采集."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_scalar('test/key', 1.0)
        # 缓冲区应有数据 (通过 flush 间接验证)
        probe._should_collect = True  # on_train_iter_begin(0) 已设置
        # 验证 record 后缓冲区非空
        assert 'test/key' in probe._scalars['train']

    def test_no_collect_off_interval(self):
        """step % interval != 0 时不应采集."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(50)
        probe.record_scalar('test/key', 1.0)
        assert 'test/key' not in probe._scalars['train']

    def test_collect_custom_interval(self):
        probe.enable(train_interval=50)
        probe.on_train_iter_begin(50)
        probe.record_scalar('test/key', 1.0)
        assert 'test/key' in probe._scalars['train']

    def test_on_train_iter_end_flushes(self):
        """on_train_iter_end 应清空训练缓冲区."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_scalar('test/key', 1.0)
        assert len(probe._scalars['train']) > 0
        probe.on_train_iter_end(0)
        assert len(probe._scalars['train']) == 0

    def test_on_train_iter_end_skips_when_not_collecting(self):
        """非采集步的 on_train_iter_end 不应 flush."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(50)
        probe._scalars['train']['manual'] = 1.0  # 手动塞入模拟残留
        probe.on_train_iter_end(50)
        # 非 collect 步, 不 flush, 缓冲区保留
        assert 'manual' in probe._scalars['train']

    def test_disabled_probe_no_collect(self):
        """未启用的 Probe 不应采集."""
        probe.on_train_iter_begin(0)
        probe.record_scalar('test/key', 1.0)
        assert len(probe._scalars['train']) == 0


# ============================================================
# record_scalar (训练)
# ============================================================

class TestRecordScalar:
    def test_basic(self):
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_scalar('loss/cls', 0.5)
        assert probe._scalars['train']['loss/cls'] == 0.5

    def test_float_conversion(self):
        """int 值应转为 float."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_scalar('count/n', 42)
        assert isinstance(probe._scalars['train']['count/n'], float)
        assert probe._scalars['train']['count/n'] == 42.0

    def test_disabled_noop(self):
        probe.record_scalar('loss/cls', 0.5)
        assert len(probe._scalars['train']) == 0

    def test_overwrites_same_key(self):
        """相同 key 后写覆盖前写."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_scalar('loss/cls', 0.5)
        probe.record_scalar('loss/cls', 0.6)
        assert probe._scalars['train']['loss/cls'] == 0.6

    def test_multiple_keys(self):
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_scalar('a', 1.0)
        probe.record_scalar('b', 2.0)
        probe.record_scalar('c', 3.0)
        assert len(probe._scalars['train']) == 3


# ============================================================
# record_tensor_stats (训练)
# ============================================================

class TestRecordTensorStats:
    def test_scalar_stats_recorded(self):
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        t = torch.randn(4, 8)
        probe.record_tensor_stats('cascade/time_emb', t)
        # 应记录 mean/std/min/max/norm
        assert 'cascade/time_emb/mean' in probe._scalars['train']
        assert 'cascade/time_emb/std' in probe._scalars['train']
        assert 'cascade/time_emb/min' in probe._scalars['train']
        assert 'cascade/time_emb/max' in probe._scalars['train']
        assert 'cascade/time_emb/norm' in probe._scalars['train']

    def test_histogram_recorded(self):
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        t = torch.randn(100)
        probe.record_tensor_stats('acts/layer0', t)
        assert 'acts/layer0' in probe._histograms['train']
        hist = probe._histograms['train']['acts/layer0']
        assert len(hist['hist']) == _HIST_BINS

    def test_histogram_disabled(self):
        """histogram=False 时不记录直方图, 仅记录标量."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        t = torch.randn(100)
        probe.record_tensor_stats('acts/layer0', t, histogram=False)
        assert 'acts/layer0' not in probe._histograms['train']
        assert 'acts/layer0/mean' in probe._scalars['train']

    def test_empty_tensor_skipped(self):
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        t = torch.tensor([])
        probe.record_tensor_stats('empty', t)
        assert len(probe._scalars['train']) == 0
        assert len(probe._histograms['train']) == 0

    def test_non_tensor_skipped(self):
        """非 Tensor 输入应被跳过, 不报错."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_tensor_stats('bad', [1, 2, 3])  # list, 不是 Tensor
        assert len(probe._scalars['train']) == 0

    def test_disabled_noop(self):
        probe.record_tensor_stats('x', torch.randn(10))
        assert len(probe._scalars['train']) == 0

    def test_stats_match_compute_function(self):
        """record_tensor_stats 记录的统计应与 _compute_tensor_stats 一致."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        t = torch.randn(50)
        expected = _compute_tensor_stats(t)
        probe.record_tensor_stats('x', t, histogram=False)
        for k, v in expected.items():
            assert probe._scalars['train'][f'x/{k}'] == pytest.approx(v)


# ============================================================
# record_detail (训练)
# ============================================================

class TestRecordDetail:
    def test_basic(self):
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_detail('box_renewal', {'rate': 0.3, 'n_renewed': 50})
        assert probe._details['train']['box_renewal'] == {'rate': 0.3, 'n_renewed': 50}

    def test_numeric_fields_extracted_to_scalars(self):
        """dict 中的数值字段应额外提取到 scalars."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_detail('box_renewal', {'rate': 0.3, 'n_renewed': 50, 'label': 'x'})
        assert probe._scalars['train']['box_renewal/rate'] == 0.3
        assert probe._scalars['train']['box_renewal/n_renewed'] == 50.0
        # 非数值字段不提取
        assert 'box_renewal/label' not in probe._scalars['train']

    def test_disabled_noop(self):
        probe.record_detail('x', {'a': 1})
        assert len(probe._details['train']) == 0


# ============================================================
# 推理缓冲区
# ============================================================

class TestInferenceBuffer:
    def test_inference_scalar(self):
        probe.enable(inference_enabled=True)
        probe.on_inference_begin()
        probe.record_inference_scalar('inference/eta_str', 0.42)
        assert probe._inference_scalars['inference']['inference/eta_str'] == 0.42

    def test_inference_tensor_stats(self):
        probe.enable(inference_enabled=True)
        probe.on_inference_begin()
        t = torch.randn(20)
        probe.record_inference_tensor_stats('inference/step0/x0', t)
        assert 'inference/step0/x0/mean' in probe._inference_scalars['inference']
        assert 'inference/step0/x0' in probe._inference_histograms['inference']

    def test_inference_detail(self):
        probe.enable(inference_enabled=True)
        probe.on_inference_begin()
        probe.record_inference_detail('step0', {'score': 0.9})
        assert probe._inference_details['inference']['step0'] == {'score': 0.9}
        assert probe._inference_scalars['inference']['step0/score'] == 0.9

    def test_inference_begin_clears_buffer(self):
        """on_inference_begin 应清空上一次推理的缓冲区."""
        probe.enable(inference_enabled=True)
        probe.on_inference_begin()
        probe.record_inference_scalar('a', 1.0)
        assert len(probe._inference_scalars['inference']) > 0
        # 第二次推理开始, 应清空
        probe.on_inference_begin()
        assert len(probe._inference_scalars['inference']) == 0

    def test_inference_end_flushes(self):
        """on_inference_end 应清空推理缓冲区."""
        probe.enable(inference_enabled=True)
        probe.on_inference_begin()
        probe.record_inference_scalar('a', 1.0)
        probe.on_inference_end()
        assert len(probe._inference_scalars['inference']) == 0

    def test_inference_disabled_noop(self):
        """inference_enabled=False 时所有推理方法为 no-op."""
        probe.enable(inference_enabled=False)
        probe.on_inference_begin()
        probe.record_inference_scalar('a', 1.0)
        probe.record_inference_tensor_stats('b', torch.randn(10))
        probe.record_inference_detail('c', {'x': 1})
        probe.on_inference_end()
        assert len(probe._inference_scalars['inference']) == 0
        assert len(probe._inference_histograms['inference']) == 0

    def test_probe_disabled_inference_noop(self):
        """Probe 完全禁用时推理方法为 no-op."""
        # 不 enable
        probe.on_inference_begin()
        probe.record_inference_scalar('a', 1.0)
        probe.on_inference_end()
        assert len(probe._inference_scalars['inference']) == 0


# ============================================================
# 缓冲区隔离 (train vs inference)
# ============================================================

class TestBufferIsolation:
    def test_train_and_inference_buffers_separate(self):
        """训练标量不应混入推理缓冲区, 反之亦然."""
        probe.enable(train_interval=100, inference_enabled=True)
        probe.on_train_iter_begin(0)
        probe.on_inference_begin()
        probe.record_scalar('train_key', 1.0)
        probe.record_inference_scalar('infer_key', 2.0)
        assert 'train_key' in probe._scalars['train']
        assert 'train_key' not in probe._inference_scalars['inference']
        assert 'infer_key' in probe._inference_scalars['inference']
        assert 'infer_key' not in probe._scalars['train']

    def test_flush_train_does_not_clear_inference(self):
        probe.enable(train_interval=100, inference_enabled=True)
        probe.on_train_iter_begin(0)
        probe.on_inference_begin()
        probe.record_scalar('train_key', 1.0)
        probe.record_inference_scalar('infer_key', 2.0)
        probe.on_train_iter_end(0)
        # 训练 flush 不应清空推理缓冲区
        assert 'infer_key' in probe._inference_scalars['inference']

    def test_flush_inference_does_not_clear_train(self):
        probe.enable(train_interval=100, inference_enabled=True)
        probe.on_train_iter_begin(0)
        probe.on_inference_begin()
        probe.record_scalar('train_key', 1.0)
        probe.record_inference_scalar('infer_key', 2.0)
        probe.on_inference_end()
        # 推理 flush 不应清空训练缓冲区
        assert 'train_key' in probe._scalars['train']


# ============================================================
# 梯度 hook
# ============================================================

class TestGradHooks:
    def test_register_grad_hooks(self):
        """register_grad_hooks 应在目标模块上注册 hook."""
        probe.enable()
        model = _MockModel()
        probe.register_grad_hooks(model)
        # 应注册: backbone + neck + time_mlp + 3 cascade heads = 6
        assert len(probe._grad_hooks) == 6

    def test_register_without_bbox_head(self):
        """缺少 bbox_head 的模型应只注册 backbone/neck."""
        probe.enable()
        model = nn.Sequential(nn.Linear(8, 8))
        model.backbone = nn.Linear(8, 8)
        model.neck = nn.Linear(8, 8)
        probe.register_grad_hooks(model)
        assert len(probe._grad_hooks) == 2

    def test_grad_hook_fires_on_backward(self):
        """反向传播时梯度 hook 应记录梯度统计."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        model = _MockModel()
        probe.register_grad_hooks(model)

        x = torch.randn(2, 8)
        out = model(x).sum()
        out.backward()

        # backbone 梯度应被记录
        assert 'grads/backbone/norm' in probe._scalars['train']
        assert 'grads/backbone/mean' in probe._scalars['train']
        assert 'grads/backbone/std' in probe._scalars['train']
        assert 'grads/backbone/max' in probe._scalars['train']
        # norm 应为正
        assert probe._scalars['train']['grads/backbone/norm'] > 0

    def test_grad_hook_for_cascade_heads(self):
        """每个 cascade head 应独立记录梯度."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        model = _MockModel()
        probe.register_grad_hooks(model)

        x = torch.randn(2, 8)
        model(x).sum().backward()

        for i in range(3):
            key = f'grads/cascade_head_{i}/norm'
            assert key in probe._scalars['train']
            assert probe._scalars['train'][key] > 0

    def test_grad_hook_skips_when_not_collecting(self):
        """非采集步梯度 hook 不应记录."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(50)  # 非 100 倍数
        model = _MockModel()
        probe.register_grad_hooks(model)

        x = torch.randn(2, 8)
        model(x).sum().backward()

        assert 'grads/backbone/norm' not in probe._scalars['train']

    def test_remove_grad_hooks(self):
        """disable 应移除所有梯度 hook."""
        probe.enable()
        model = _MockModel()
        probe.register_grad_hooks(model)
        assert len(probe._grad_hooks) > 0
        probe.disable()
        assert len(probe._grad_hooks) == 0
        # disable 后反向传播不应报错 (hook 已移除)
        x = torch.randn(2, 8)
        model(x).sum().backward()

    def test_reregister_replaces_old_hooks(self):
        """重复 register 应先移除旧 hook 再注册新 hook."""
        probe.enable()
        model = _MockModel()
        probe.register_grad_hooks(model)
        first_count = len(probe._grad_hooks)
        # 再次注册
        probe.register_grad_hooks(model)
        assert len(probe._grad_hooks) == first_count  # 不应翻倍

    def test_register_when_disabled_noop(self):
        """未启用时 register_grad_hooks 为 no-op."""
        # 不 enable
        model = _MockModel()
        probe.register_grad_hooks(model)
        assert len(probe._grad_hooks) == 0


# ============================================================
# _flush_train / _flush_inference (SwanLab 上传)
# ============================================================

class TestFlush:
    def test_flush_train_calls_swanlab(self, monkeypatch):
        """_flush_train 应调用 _swanlab_log."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)
        probe.record_scalar('a', 1.0)

        calls = []
        def fake_log(data, step):
            calls.append((data, step))
        monkeypatch.setattr(probe, '_swanlab_log', fake_log)
        monkeypatch.setattr(probe, '_swanlab_log_histogram', lambda k, h, s: None)

        probe.on_train_iter_end(0)
        assert len(calls) == 1
        assert calls[0][1] == 0
        assert 'a' in calls[0][0]

    def test_flush_train_empty_skips(self, monkeypatch):
        """空缓冲区不应调用 swanlab."""
        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)

        called = []
        monkeypatch.setattr(probe, '_swanlab_log', lambda d, s: called.append(True))
        monkeypatch.setattr(probe, '_swanlab_log_histogram', lambda k, h, s: called.append(True))

        probe.on_train_iter_end(0)
        assert len(called) == 0

    def test_flush_inference_calls_swanlab(self, monkeypatch):
        probe.enable(inference_enabled=True)
        probe.on_inference_begin()
        probe.record_inference_scalar('a', 1.0)

        calls = []
        monkeypatch.setattr(probe, '_swanlab_log', lambda d, s: calls.append((d, s)))
        monkeypatch.setattr(probe, '_swanlab_log_histogram', lambda k, h, s: None)

        probe.on_inference_end()
        assert len(calls) == 1
        assert 'a' in calls[0][0]

    def test_swanlab_log_silent_fail(self):
        """_swanlab_log 在 swanlab 不可用时应静默失败, 不抛异常."""
        # 不 mock, swanlab 未初始化时会抛异常, 应被捕获
        probe._swanlab_log({'a': 1.0}, 0)  # 不应抛异常

    def test_swanlab_log_histogram_silent_fail(self):
        """_swanlab_log_histogram 失败时应静默."""
        hist_data = {
            'min': 0.0, 'max': 1.0, 'mean': 0.5, 'std': 0.3,
            'hist': [0.5, 0.5], 'bin_edges': [0.0, 0.5, 1.0],
        }
        # 不应抛异常
        probe._swanlab_log_histogram('test', hist_data, 0)


# ============================================================
# 集成: 模拟训练 step 流程
# ============================================================

class TestIntegrationTrainStep:
    def test_full_train_iter_cycle(self, monkeypatch):
        """模拟完整训练 iter: begin → record → end → flush."""
        probe.enable(train_interval=100)
        monkeypatch.setattr(probe, '_swanlab_log', lambda d, s: None)
        monkeypatch.setattr(probe, '_swanlab_log_histogram', lambda k, h, s: None)

        # step 0 (采集步)
        probe.on_train_iter_begin(0)
        probe.record_scalar('train/loss/cls', 0.5)
        probe.record_scalar('train/loss/bbox', 0.3)
        probe.record_tensor_stats('cascade/time_emb', torch.randn(8, 16))
        probe.record_detail('criterion', {'num_pos': 10, 'fg_ratio': 0.2})
        probe.on_train_iter_end(0)
        # flush 后缓冲区应清空
        assert len(probe._scalars['train']) == 0
        assert len(probe._histograms['train']) == 0
        assert len(probe._details['train']) == 0

    def test_multiple_steps_only_collects_at_interval(self, monkeypatch):
        """多个 step 中只有 interval 倍数步采集."""
        probe.enable(train_interval=100)
        monkeypatch.setattr(probe, '_swanlab_log', lambda d, s: None)
        monkeypatch.setattr(probe, '_swanlab_log_histogram', lambda k, h, s: None)

        for step in [0, 50, 100, 150, 200]:
            probe.on_train_iter_begin(step)
            probe.record_scalar('loss', float(step))
            probe.on_train_iter_end(step)

        # 非采集步 (50, 150) 不应记录
        # 采集步 (0, 100, 200) 应记录后 flush 清空
        # 最终缓冲区应为空 (最后一个采集步 200 已 flush)
        assert len(probe._scalars['train']) == 0

    def test_train_then_inference(self, monkeypatch):
        """训练采集后切换到推理, 两者互不干扰."""
        probe.enable(train_interval=100, inference_enabled=True)
        monkeypatch.setattr(probe, '_swanlab_log', lambda d, s: None)
        monkeypatch.setattr(probe, '_swanlab_log_histogram', lambda k, h, s: None)

        # 训练步
        probe.on_train_iter_begin(0)
        probe.record_scalar('train/loss', 1.0)
        probe.on_train_iter_end(0)

        # 推理
        probe.on_inference_begin()
        probe.record_inference_scalar('inference/eta_str', 0.1)
        probe.record_inference_tensor_stats('inference/step0/x0', torch.randn(10))
        probe.on_inference_end()

        # 推理结束后缓冲区清空
        assert len(probe._inference_scalars['inference']) == 0
        assert len(probe._inference_histograms['inference']) == 0


# ============================================================
# 线程安全
# ============================================================

class TestThreadSafety:
    def test_concurrent_record_scalar(self):
        """多线程并发 record_scalar 不应崩溃."""
        import threading

        probe.enable(train_interval=100)
        probe.on_train_iter_begin(0)

        def worker(tid):
            for i in range(100):
                probe.record_scalar(f'thread_{tid}/iter', float(i))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 每个 thread 最后一次写覆盖前几次, 应有 8 个 key
        assert len(probe._scalars['train']) == 8
