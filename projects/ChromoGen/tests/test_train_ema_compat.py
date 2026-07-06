"""train.py EMAModel 兼容转发层集成测试

补充覆盖：之前 RED 阶段只测了 models.ema.EMAModel，漏测了 train.py 中的兼容转发层。
导致 GREEN 阶段测试全绿但实际运行报错：
  EMAModel.__init__() got an unexpected keyword argument 'decay_start'

红绿重构纪律要求：测试必须覆盖实际代码路径，而非只测新模块。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


# ============================================================
# train.py EMAModel 兼容转发层集成测试
# ============================================================


class TestTrainEMAModelCompatLayer:
    """验证 train.py 中的 EMAModel 兼容层能接受并转发所有新参数

    这是真正的集成测试：从 train.py 入口导入 EMAModel，
    而非从 models.ema 直接导入新实现。
    """

    def _import_train_emamodel(self):
        """从 train.py 导入 EMAModel 类（兼容转发层）"""
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'tools')
        )
        from train import EMAModel

        return EMAModel

    def test_train_emamodel_accepts_decay_start(self):
        """train.py 的 EMAModel 应接受 decay_start 参数"""
        EMAModel = self._import_train_emamodel()
        model = torch.nn.Linear(4, 4)
        ema = EMAModel(
            model, decay=0.9999, decay_start=0.999, warmup_steps=100
        )
        assert ema.decay_start == 0.999

    def test_train_emamodel_accepts_warmup_steps(self):
        """train.py 的 EMAModel 应接受 warmup_steps 参数"""
        EMAModel = self._import_train_emamodel()
        model = torch.nn.Linear(4, 4)
        ema = EMAModel(
            model, decay=0.9999, decay_start=0.999, warmup_steps=100
        )
        assert ema.warmup_steps == 100

    def test_train_emamodel_decay_start_none_default(self):
        """train.py 的 EMAModel 不传 decay_start 时默认为 None（无 warmup）"""
        EMAModel = self._import_train_emamodel()
        model = torch.nn.Linear(4, 4)
        ema = EMAModel(model, decay=0.9999)
        assert ema.decay_start is None
        assert ema.warmup_steps == 0

    def test_train_emamodel_forwards_get_current_decay(self):
        """train.py 的 EMAModel 应转发 get_current_decay 方法"""
        EMAModel = self._import_train_emamodel()
        model = torch.nn.Linear(4, 4)
        ema = EMAModel(
            model, decay=0.9999, decay_start=0.999, warmup_steps=100
        )
        # warmup 期间 decay 应递增
        ema._impl.cur_step = 0
        d0 = ema.get_current_decay()
        ema._impl.cur_step = 100
        d100 = ema.get_current_decay()
        assert d0 < d100, f'warmup decay 应递增: {d0} < {d100}'

    def test_train_emamodel_shadow_property(self):
        """train.py 的 EMAModel shadow 属性应保持兼容"""
        EMAModel = self._import_train_emamodel()
        model = torch.nn.Linear(4, 4)
        ema = EMAModel(model, decay=0.9999)
        # shadow 应是 dict（旧代码访问 self.shadow）
        assert isinstance(ema.shadow, dict)
        # 更新后 shadow 应更新
        old_shadow = dict(ema.shadow)
        ema.update(model)
        # shadow 应仍存在
        assert isinstance(ema.shadow, dict)
        # 应有相同 keys
        assert set(ema.shadow.keys()) == set(old_shadow.keys())

    def test_train_emamodel_apply_restore_roundtrip(self):
        """train.py 的 EMAModel apply_shadow/restore 应可逆

        语义：
          apply_shadow: 临时把 EMA shadow 参数应用到模型（用于评估）
          restore: 恢复到 apply_shadow 之前的训练参数
        """
        EMAModel = self._import_train_emamodel()
        model = torch.nn.Linear(4, 4)
        orig_weight = model.weight.data.clone()  # EMA shadow 初始值
        ema = EMAModel(model, decay=0.9999)
        # 训练时修改参数（与 shadow 不同）
        with torch.no_grad():
            model.weight.data.fill_(0.0)
        train_weight = model.weight.data.clone()  # 应为全 0
        # apply_shadow：weight 应变为 shadow 值（=orig）
        ema.apply_shadow(model)
        assert torch.allclose(model.weight.data, orig_weight), (
            'apply_shadow 后 weight 应等于 EMA shadow'
        )
        # restore：weight 应恢复到 apply_shadow 之前的训练值（=0）
        ema.restore(model)
        assert torch.allclose(model.weight.data, train_weight), (
            'restore 后 weight 应回到 apply_shadow 前的训练值'
        )


# ============================================================
# train.py 实际使用场景测试（端到端集成）
# ============================================================


class TestTrainEMAModelActualUsage:
    """模拟 train.py 中实际使用 EMAModel 的方式

    这是之前漏测的关键场景：train.py 中实际是按这种构造方式调用的
    """

    def test_train_emamodel_actual_construction_pattern(self):
        """模拟 train.py 中的实际构造模式

        train.py 中的实际代码：
            ema = EMAModel(
                model,
                decay=ema_decay,
                decay_start=ema_decay_start,
                warmup_steps=ema_warmup_steps,
            )

        其中 ema_decay_start 可能从 cfg.get('ema_decay_start') 返回 None
        """
        EMAModel = self._import_train_emamodel()
        model = torch.nn.Linear(4, 4)

        # 模拟 cfg.get('ema_decay_start') 返回 None 的情况
        ema_decay = 0.9999
        ema_decay_start = None  # cfg.get('ema_decay_start')
        ema_warmup_steps = 0  # cfg.get('ema_warmup_steps', 0)

        # 不应报错
        ema = EMAModel(
            model,
            decay=ema_decay,
            decay_start=ema_decay_start,
            warmup_steps=ema_warmup_steps,
        )
        assert ema is not None

        # 模拟 v2 配置：decay_start=0.999, warmup_steps=1000
        ema_v2 = EMAModel(
            model,
            decay=0.9999,
            decay_start=0.999,
            warmup_steps=1000,
        )
        assert ema_v2.decay_start == 0.999
        assert ema_v2.warmup_steps == 1000

    def _import_train_emamodel(self):
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'tools')
        )
        from train import EMAModel

        return EMAModel


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
