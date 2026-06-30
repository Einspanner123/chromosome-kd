"""集成测试: 方向四 (非线性轨迹) 端到端训练流程验证

测试层级 (从底到顶):
1. 配置解析: Config.fromfile 能正确解析方向四的配置文件
2. 模型构建: 从配置构建完整 LDMDet 模型 (backbone + neck + bbox_head + 方向四组件)
3. 损失前向: 用 dummy 数据跑 loss forward, 验证损失项正确出现
4. 梯度回传: loss.backward() 不报错, 验证参数有梯度
5. 推理前向: model.predict() 不报错, 验证推理逻辑
6. 1-iter 训练循环: optimizer.step() 后参数更新, 验证完整训练流程

每个测试用最小 dummy 数据 (bs=2, 256x256, 5 GT/图), CPU 运行, 不依赖 GPU/数据集.
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
from mmengine.registry import init_default_scope  # noqa: F402

init_default_scope('mmdet')

import experiments.mmdet_bridge.registry  # noqa: F401, E402
import experiments.mmdet_bridge.detector  # noqa: F401, E402
import experiments.mmdet_bridge.hooks  # noqa: F401, E402

from mmengine.config import Config  # noqa: F402
from mmengine.registry import MODELS  # noqa: F402
from mmengine.structures import InstanceData  # noqa: F402
from mmdet.structures import DetDataSample  # noqa: F402


# ================================================================
# 常量
# ================================================================

CONFIG_DIR = os.path.join(_PROJECT_ROOT, 'experiments', 'configs', 'ldmdet')

# 方向四: 非线性轨迹主配置 + 消融配置
DIRECTION_CONFIGS = [
    ('direction_4_nonlinear_trajectory', 'nonlinear_trajectory.py'),
    ('e41_scale_only', 'nonlinear_trajectory_e41.py'),
    ('e42_ot_only', 'nonlinear_trajectory_e42.py'),
    ('e43_eps2', 'nonlinear_trajectory_e43_eps2.py'),
    ('e43_eps3', 'nonlinear_trajectory_e43_eps3.py'),
    ('e43_multinomial', 'nonlinear_trajectory_e43_multinomial.py'),
]

# dummy 数据参数
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
    """从配置文件构建模型"""
    cfg = Config.fromfile(_config_path(config_filename))
    model = MODELS.build(cfg.model)
    return model, cfg


def _make_dummy_batch(bs: int = DUMMY_BS, img_size: int = DUMMY_IMG_SIZE,
                      num_gt: int = DUMMY_NUM_GT, with_gt: bool = True):
    """构造 dummy 输入数据

    Args:
        bs: batch size
        img_size: 图像尺寸
        num_gt: 每张图的 GT 数
        with_gt: 是否包含 GT (推理时不需要)

    Returns:
        batch_inputs: [bs, 3, H, W]
        batch_data_samples: List[DetDataSample]
    """
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
            # 生成有效的 xyxy 框 (x2 > x1, y2 > y1)
            x1 = torch.rand(num_gt) * (img_size - 60)
            y1 = torch.rand(num_gt) * (img_size - 60)
            w = torch.rand(num_gt) * 50 + 10
            h = torch.rand(num_gt) * 50 + 10
            gt_instances.bboxes = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)
            gt_instances.labels = torch.randint(0, DUMMY_NUM_CLASSES, (num_gt,))
            ds.gt_instances = gt_instances
        batch_data_samples.append(ds)
    return batch_inputs, batch_data_samples


def _collect_grad_params(model):
    """收集所有有梯度的参数 (param, name)"""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


# ================================================================
# 1. 配置解析测试
# ================================================================


class TestConfigParsing:
    """测试方向四的配置文件能被正确解析"""

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_config_parse(self, name, filename):
        """配置文件能被 Config.fromfile 解析"""
        cfg = Config.fromfile(_config_path(filename))
        assert cfg is not None

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_config_has_model(self, name, filename):
        """配置包含 model 字段"""
        cfg = Config.fromfile(_config_path(filename))
        assert 'model' in cfg
        assert 'type' in cfg.model

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_config_model_type(self, name, filename):
        """model.type 应为 LDMDet"""
        cfg = Config.fromfile(_config_path(filename))
        assert cfg.model.type == 'LDMDet'


# ================================================================
# 2. 模型构建测试
# ================================================================


class TestModelBuilding:
    """测试从配置构建完整模型"""

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_build_model(self, name, filename):
        """能从配置构建模型"""
        model, _ = _build_model(filename)
        assert model is not None

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_model_has_params(self, name, filename):
        """模型有参数"""
        model, _ = _build_model(filename)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_model_params_require_grad(self, name, filename):
        """大部分参数应需要梯度"""
        model, _ = _build_model(filename)
        grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        # 至少 90% 的参数需要梯度
        assert grad_params / total_params > 0.9


# ================================================================
# 3. 损失前向测试
# ================================================================


class TestLossForward:
    """测试损失前向传播"""

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_loss_forward(self, name, filename):
        """loss forward 不报错, 返回 loss 字典"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        assert isinstance(losses, dict)
        assert len(losses) > 0

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_loss_values_are_tensors(self, name, filename):
        """所有 loss 值应为标量 Tensor"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f'{key} is not Tensor'
            assert val.dim() == 0, f'{key} is not scalar: shape={val.shape}'

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_loss_has_main_keys(self, name, filename):
        """应包含主损失键: loss_cls, loss_bbox, loss_giou"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses


# ================================================================
# 4. 梯度回传测试
# ================================================================


class TestGradientBackward:
    """测试梯度回传"""

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_backward_no_error(self, name, filename):
        """loss.backward() 不报错"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        total_loss.backward()

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_gradients_exist(self, name, filename):
        """backward 后参数应有梯度"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        total_loss.backward()
        # 检查至少部分参数有梯度
        has_grad = sum(1 for p in model.parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0)
        assert has_grad > 0


# ================================================================
# 5. 推理前向测试
# ================================================================


class TestPredictForward:
    """测试推理前向传播"""

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_predict_no_error(self, name, filename):
        """predict 不报错"""
        model, _ = _build_model(filename)
        model.eval()
        batch_inputs, batch_data_samples = _make_dummy_batch(bs=1, with_gt=False)
        with torch.no_grad():
            results = model.predict(batch_inputs, batch_data_samples)
        assert len(results) == 1

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_predict_has_detections(self, name, filename):
        """推理结果应包含检测框"""
        model, _ = _build_model(filename)
        model.eval()
        batch_inputs, batch_data_samples = _make_dummy_batch(bs=1, with_gt=False)
        with torch.no_grad():
            results = model.predict(batch_inputs, batch_data_samples)
        pred = results[0].pred_instances
        assert hasattr(pred, 'bboxes')
        assert hasattr(pred, 'scores')
        assert hasattr(pred, 'labels')
        # bboxes 应为 [N, 4]
        if pred.bboxes.numel() > 0:
            assert pred.bboxes.shape[1] == 4
            assert pred.scores.shape[0] == pred.bboxes.shape[0]
            assert pred.labels.shape[0] == pred.bboxes.shape[0]


# ================================================================
# 6. 1-iter 训练循环测试
# ================================================================


class TestOneIterTraining:
    """测试 1 个 iteration 的完整训练循环"""

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_one_iter_training(self, name, filename):
        """1-iter 训练: forward → backward → step → 参数更新"""
        model, _ = _build_model(filename)
        model.train()

        # 记录初始参数
        params_before = {n: p.detach().clone()
                         for n, p in model.named_parameters() if p.requires_grad}

        # 构造 optimizer
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, momentum=0.9,
        )

        # 1-iter 训练
        batch_inputs, batch_data_samples = _make_dummy_batch()
        losses = model.loss(batch_inputs, batch_data_samples)
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 验证参数已更新
        updated_count = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in params_before:
                if not torch.equal(params_before[n], p.detach()):
                    updated_count += 1
        assert updated_count > 0, '至少部分参数应被更新'

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_two_iter_training(self, name, filename):
        """2-iter 训练: 验证连续训练不报错 (状态一致性)"""
        model, _ = _build_model(filename)
        model.train()

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, momentum=0.9,
        )

        for _ in range(2):
            batch_inputs, batch_data_samples = _make_dummy_batch()
            losses = model.loss(batch_inputs, batch_data_samples)
            total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


# ================================================================
# 7. 边界情况测试
# ================================================================


class TestEdgeCases:
    """测试边界情况"""

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_empty_gt(self, name, filename):
        """空 GT: loss forward 不报错"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch(num_gt=0)
        losses = model.loss(batch_inputs, batch_data_samples)
        assert isinstance(losses, dict)

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_single_gt(self, name, filename):
        """单个 GT: loss forward 不报错"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs, batch_data_samples = _make_dummy_batch(num_gt=1)
        losses = model.loss(batch_inputs, batch_data_samples)
        assert isinstance(losses, dict)

    @pytest.mark.parametrize('name,filename', DIRECTION_CONFIGS)
    def test_different_gt_count_per_image(self, name, filename):
        """不同图 GT 数不同: loss forward 不报错"""
        model, _ = _build_model(filename)
        model.train()
        batch_inputs = torch.randn(2, 3, DUMMY_IMG_SIZE, DUMMY_IMG_SIZE)
        batch_data_samples = []
        for num_gt in [3, 7]:
            ds = DetDataSample()
            ds.set_metainfo(dict(
                img_shape=(DUMMY_IMG_SIZE, DUMMY_IMG_SIZE),
                pad_shape=(DUMMY_IMG_SIZE, DUMMY_IMG_SIZE),
                ori_shape=(DUMMY_IMG_SIZE, DUMMY_IMG_SIZE),
                scale_factor=[1.0, 1.0, 1.0, 1.0],
            ))
            gt_instances = InstanceData()
            x1 = torch.rand(num_gt) * (DUMMY_IMG_SIZE - 60)
            y1 = torch.rand(num_gt) * (DUMMY_IMG_SIZE - 60)
            w = torch.rand(num_gt) * 50 + 10
            h = torch.rand(num_gt) * 50 + 10
            gt_instances.bboxes = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)
            gt_instances.labels = torch.randint(0, DUMMY_NUM_CLASSES, (num_gt,))
            ds.gt_instances = gt_instances
            batch_data_samples.append(ds)
        losses = model.loss(batch_inputs, batch_data_samples)
        assert isinstance(losses, dict)
