"""集成测试: 24obj 主路线消融实验 (A0/A1/A2/A3)

测试层级:
1. 配置解析: 4 个消融配置能被正确解析
2. 模型构建: 从配置构建完整 LDMDet 模型
3. 关键组件激活: 验证 RF/Heun/AdaLN/StochOT 在对应配置中真实激活
4. 数据集切换: 验证数据集正确切换到 24obj
5. 损失前向: 用 dummy 数据跑 loss forward
6. 梯度回传: loss.backward() 不报错
7. 推理前向: model.predict() 不报错

消融组件递进:
  A0 baseline:  ddpm(default) | euler(default) | ts=1 | none     | random
  A1 RF+Heun:   rectified_flow | heun           | ts=4 | none     | random
  A2 +AdaLN:    rectified_flow | heun           | ts=4 | adaln_zero| random
  A3 Full SOTA: rectified_flow | heun           | ts=4 | adaln_zero| ot_flow eps5
"""

import os
import sys

import pytest
import torch

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 注册所有模块 (和 tools/train.py 一致)
import mmdet.models  # noqa: F401, E402
from mmengine.registry import init_default_scope  # noqa: E402

init_default_scope('mmdet')

import experiments.mmdet_bridge.registry  # noqa: F401, E402
import experiments.mmdet_bridge.detector  # noqa: F401, E402
import experiments.mmdet_bridge.hooks  # noqa: F401, E402

from mmengine.config import Config  # noqa: E402
from mmengine.registry import MODELS  # noqa: E402
from mmengine.structures import InstanceData  # noqa: E402
from mmdet.structures import DetDataSample  # noqa: E402


# ================================================================
# 常量
# ================================================================

CONFIG_DIR = os.path.join(
    _PROJECT_ROOT, 'experiments', 'configs', 'ldmdet', 'directions',
    'mainline_ablation_24obj'
)

# 5 个消融配置
ABLATION_CONFIGS = [
    ('a0_baseline', 'a0_baseline_24obj.py'),
    ('a1_rf_heun', 'a1_rf_heun_24obj.py'),
    ('a2_rf_heun_adaln', 'a2_rf_heun_adaln_24obj.py'),
    ('a3_full_sota', 'a3_full_sota_24obj.py'),
    ('a4_dpm_pp', 'a4_dpm_pp_24obj.py'),
]

DUMMY_BS = 2
DUMMY_IMG_SIZE = 256
DUMMY_NUM_GT = 5
DUMMY_NUM_CLASSES = 24


# ================================================================
# 辅助函数
# ================================================================


def _config_path(filename: str) -> str:
    return os.path.join(CONFIG_DIR, filename)


def _build_model(config_filename: str):
    cfg = Config.fromfile(_config_path(config_filename))
    model = MODELS.build(cfg.model)
    return model, cfg


def _make_dummy_batch(bs: int = DUMMY_BS, img_size: int = DUMMY_IMG_SIZE,
                      num_gt: int = DUMMY_NUM_GT, with_gt: bool = True):
    batch_inputs = torch.randn(bs, 3, img_size, img_size)
    batch_data_samples = []
    for i in range(bs):
        ds = DetDataSample()
        ds.set_metainfo(dict(
            img_shape=(img_size, img_size),
            pad_shape=(img_size, img_size),
            ori_shape=(img_size, img_size),
            scale_factor=[1.0, 1.0, 1.0, 1.0],
        ))
        if with_gt:
            gt_instances = InstanceData()
            x1 = torch.rand(num_gt) * (img_size - 60)
            y1 = torch.rand(num_gt) * (img_size - 60)
            w = torch.rand(num_gt) * 50 + 10
            h = torch.rand(num_gt) * 50 + 10
            gt_instances.bboxes = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)
            gt_instances.labels = torch.randint(0, DUMMY_NUM_CLASSES, (num_gt,))
            ds.gt_instances = gt_instances
        batch_data_samples.append(ds)
    return batch_inputs, batch_data_samples


# ================================================================
# 1. 配置解析测试
# ================================================================


class TestConfigParsing:
    """测试消融配置文件能被正确解析"""

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_config_parse(self, name, filename):
        cfg = Config.fromfile(_config_path(filename))
        assert cfg is not None

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_config_has_model(self, name, filename):
        cfg = Config.fromfile(_config_path(filename))
        assert 'model' in cfg
        assert cfg.model.type == 'LDMDet'


class TestDatasetSwitch:
    """测试数据集正确切换到 24obj"""

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_data_root_is_24obj(self, name, filename):
        cfg = Config.fromfile(_config_path(filename))
        data_root = cfg.train_dataloader.dataset.data_root
        assert '24_chromosomes_object' in data_root, \
            f'{name}: data_root 应包含 24_chromosomes_object, 实际: {data_root}'

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_classwise_enabled(self, name, filename):
        """项目硬约束: val_evaluator 必须启用 classwise=True"""
        cfg = Config.fromfile(_config_path(filename))
        assert cfg.val_evaluator.get('classwise') is True, \
            f'{name}: val_evaluator.classwise 应为 True'


# ================================================================
# 2. 关键组件激活测试 (核心: 验证组件真实被配置)
# ================================================================


class TestComponentActivation:
    """验证消融组件在对应配置中正确激活"""

    def test_a0_baseline_no_rf(self):
        """A0: 无 RF (diffusion_type 为默认)"""
        cfg = Config.fromfile(_config_path('a0_baseline_24obj.py'))
        bh = cfg.model.bbox_head
        # baseline 不应设置 diffusion_type 或为非 rectified_flow
        assert bh.get('diffusion_type', None) != 'rectified_flow', \
            'A0 baseline 不应使用 rectified_flow'

    def test_a0_baseline_euler_single_step(self):
        """A0: Euler 1步采样"""
        cfg = Config.fromfile(_config_path('a0_baseline_24obj.py'))
        bh = cfg.model.bbox_head
        assert bh.get('sampling_timesteps', 1) == 1
        assert bh.get('solver_type', 'euler') == 'euler'

    def test_a1_has_rf(self):
        """A1: rectified_flow 已激活"""
        cfg = Config.fromfile(_config_path('a1_rf_heun_24obj.py'))
        assert cfg.model.bbox_head.diffusion_type == 'rectified_flow'

    def test_a1_has_heun(self):
        """A1: Heun 采样器已激活"""
        cfg = Config.fromfile(_config_path('a1_rf_heun_24obj.py'))
        assert cfg.model.bbox_head.solver_type == 'heun'
        assert cfg.model.bbox_head.sampling_timesteps == 4

    def test_a1_no_adaln(self):
        """A1: 无 AdaLN (time_conditioning 为默认)"""
        cfg = Config.fromfile(_config_path('a1_rf_heun_24obj.py'))
        sh = cfg.model.bbox_head.single_head
        assert sh.get('time_conditioning', None) != 'adaln_zero', \
            'A1 不应使用 adaln_zero'

    def test_a2_has_adaln(self):
        """A2: AdaLN-Zero 已激活"""
        cfg = Config.fromfile(_config_path('a2_rf_heun_adaln_24obj.py'))
        sh = cfg.model.bbox_head.single_head
        assert sh.get('time_conditioning') == 'adaln_zero'

    def test_a2_no_stochot(self):
        """A2: 无 StochOT (coupling 为默认 random)"""
        cfg = Config.fromfile(_config_path('a2_rf_heun_adaln_24obj.py'))
        bh = cfg.model.bbox_head
        # A2 不应设置 ot_flow coupling
        if 'coupling' in bh:
            assert bh.coupling.get('type', 'random') != 'ot_flow', \
                'A2 不应使用 ot_flow coupling'

    def test_a3_has_stochot(self):
        """A3: StochOT eps5 已激活"""
        cfg = Config.fromfile(_config_path('a3_full_sota_24obj.py'))
        bh = cfg.model.bbox_head
        assert bh.coupling.type == 'ot_flow'
        assert bh.coupling.epsilon == 5.0

    def test_a3_has_adaln(self):
        """A3: AdaLN-Zero 已激活 (继承自 A2 层)"""
        cfg = Config.fromfile(_config_path('a3_full_sota_24obj.py'))
        sh = cfg.model.bbox_head.single_head
        assert sh.get('time_conditioning') == 'adaln_zero'

    def test_a4_has_dpm_solver_pp(self):
        """A4: DPM-Solver++ 采样器已激活"""
        cfg = Config.fromfile(_config_path('a4_dpm_pp_24obj.py'))
        assert cfg.model.bbox_head.solver_type == 'dpm_solver_pp'
        assert cfg.model.bbox_head.sampling_timesteps == 4

    def test_a4_inherits_sota_components(self):
        """A4: 继承 A3 的 RF+AdaLN+StochOT 组件"""
        cfg = Config.fromfile(_config_path('a4_dpm_pp_24obj.py'))
        bh = cfg.model.bbox_head
        assert bh.diffusion_type == 'rectified_flow'
        assert bh.single_head.get('time_conditioning') == 'adaln_zero'
        assert bh.coupling.type == 'ot_flow'
        assert bh.coupling.epsilon == 5.0

    def test_component_progression(self):
        """验证消融组件递进关系: A0 ⊂ A1 ⊂ A2 ⊂ A3"""
        cfgs = {}
        for name, filename in ABLATION_CONFIGS:
            cfgs[name] = Config.fromfile(_config_path(filename))

        # A0 → A1: +RF + Heun
        assert cfgs['a0_baseline'].model.bbox_head.get('diffusion_type') != \
            cfgs['a1_rf_heun'].model.bbox_head.get('diffusion_type')

        # A1 → A2: +AdaLN
        a1_tc = cfgs['a1_rf_heun'].model.bbox_head.single_head.get(
            'time_conditioning')
        a2_tc = cfgs['a2_rf_heun_adaln'].model.bbox_head.single_head.get(
            'time_conditioning')
        assert a1_tc != a2_tc

        # A2 → A3: +StochOT
        a2_has_ot = 'coupling' in cfgs['a2_rf_heun_adaln'].model.bbox_head and \
            cfgs['a2_rf_heun_adaln'].model.bbox_head.coupling.get('type') == 'ot_flow'
        a3_has_ot = cfgs['a3_full_sota'].model.bbox_head.coupling.type == 'ot_flow'
        assert not a2_has_ot and a3_has_ot


# ================================================================
# 3. 模型构建测试
# ================================================================


class TestModelBuilding:
    """测试从配置构建完整模型"""

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_build_model(self, name, filename):
        model, _ = _build_model(filename)
        assert model is not None

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_model_has_params(self, name, filename):
        model, _ = _build_model(filename)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_model_params_require_grad(self, name, filename):
        model, _ = _build_model(filename)
        grad_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        assert grad_params / total_params > 0.9


# ================================================================
# 4. AdaLN-Zero 真实计算验证 (核心: 参数级别验证)
# ================================================================


class TestAdaLNRealComputation:
    """验证 AdaLN-Zero 在模型中真实存在且参数被正确初始化"""

    def _find_adaln_params(self, model):
        """查找模型中所有 AdaLN 相关参数"""
        return [(n, p) for n, p in model.named_parameters()
                if 'adaln' in n.lower() or 'modulation' in n.lower()]

    def test_a2_has_adaln_params(self):
        """A2 模型应包含 AdaLN 相关参数 (adaln_mlp)"""
        model, _ = _build_model('a2_rf_heun_adaln_24obj.py')
        adaln_params = self._find_adaln_params(model)
        assert len(adaln_params) > 0, \
            f'A2 应包含 AdaLN 相关参数 (如 adaln_mlp)'

    def test_a3_has_adaln_params(self):
        """A3 模型应包含 AdaLN 相关参数 (adaln_mlp)"""
        model, _ = _build_model('a3_full_sota_24obj.py')
        adaln_params = self._find_adaln_params(model)
        assert len(adaln_params) > 0

    def test_a1_no_adaln_params(self):
        """A1 模型不应包含 AdaLN 相关参数"""
        model, _ = _build_model('a1_rf_heun_24obj.py')
        adaln_params = self._find_adaln_params(model)
        assert len(adaln_params) == 0, \
            f'A1 不应包含 AdaLN 参数, 实际找到: {[n for n, _ in adaln_params]}'

    def test_adaln_zero_init(self):
        """AdaLN-Zero 的调制参数应被零初始化 (adaln_mlp.1 层)"""
        model, _ = _build_model('a2_rf_heun_adaln_24obj.py')
        adaln_params = self._find_adaln_params(model)
        # adaln_mlp.1 (第二层 Linear) 应被零初始化 (AdaLN-Zero 特性)
        zero_params = [(n, p) for n, p in adaln_params
                       if 'adaln_mlp.1' in n and p.abs().sum().item() == 0]
        assert len(zero_params) > 0, \
            'AdaLN-Zero 的 adaln_mlp.1 层应被零初始化'


# ================================================================
# 5. 损失前向测试
# ================================================================


class TestLossForward:
    """测试损失前向传播"""

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_loss_forward(self, name, filename):
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        assert isinstance(losses, dict)
        assert len(losses) > 0

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_loss_has_main_keys(self, name, filename):
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses


# ================================================================
# 6. 梯度回传测试
# ================================================================


class TestGradientBackward:
    """测试梯度回传"""

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_backward_no_error(self, name, filename):
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        total_loss = sum(v for v in losses.values()
                         if isinstance(v, torch.Tensor))
        total_loss.backward()

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_adaln_params_have_grad(self, name, filename):
        """验证 AdaLN 参数 (如果存在) 有梯度"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        total_loss = sum(v for v in losses.values()
                         if isinstance(v, torch.Tensor))
        total_loss.backward()

        # 检查 AdaLN 参数是否有梯度
        adaln_grad_count = 0
        for n, p in model.named_parameters():
            if ('adaln' in n.lower() or 'modulation' in n.lower()) \
                    and p.grad is not None:
                if p.grad.abs().sum() > 0:
                    adaln_grad_count += 1

        # A2/A3 应有 AdaLN 参数梯度, A0/A1 不应有
        cfg = Config.fromfile(_config_path(filename))
        tc = cfg.model.bbox_head.single_head.get('time_conditioning')
        if tc == 'adaln_zero':
            assert adaln_grad_count > 0, \
                f'{name}: AdaLN 参数应有梯度, 但未找到非零梯度'


# ================================================================
# 7. 推理前向测试
# ================================================================


class TestPredictForward:
    """测试推理前向传播"""

    @pytest.mark.parametrize('name,filename', ABLATION_CONFIGS)
    def test_predict_no_error(self, name, filename):
        model, _ = _build_model(filename)
        model.eval()
        batch_inputs, batch_data_samples = _make_dummy_batch(
            bs=1, with_gt=False)
        with torch.no_grad():
            results = model.predict(batch_inputs, batch_data_samples)
        assert len(results) == 1
        pred = results[0].pred_instances
        assert hasattr(pred, 'bboxes')
        assert hasattr(pred, 'scores')
        assert hasattr(pred, 'labels')
