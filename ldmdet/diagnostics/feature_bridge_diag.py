"""FeatureBridgeDiagnosticsHook — 方向六 FBM 诊断 Hook.

监控 FBM (FeatureBridgeModule) 和 ChromoGen UNet 的训练状态,
帮助定位哪个模块在贡献、哪个在拖累。

监控维度:
- gate 统计: 各层 sigmoid gate 的 mean/std/min/max (CG 特征贡献度)
- 特征范数: LD/CG/fused 各层 L2 范数 (谁在主导)
- proj 层参数: 投影层权重 mean/std (投影是否在学习)
- gate 梯度: 各层 gate 的梯度范数 (gate 是否在被优化)
- UNet 梯度: 解冻的 UNet block 梯度范数 (哪些 block 在学习)

采样频率: 默认每 50 iter, 可配置。
"""

import logging
from typing import Dict

from mmengine.hooks import Hook
from mmdet.registry import HOOKS

from ldmdet.diagnostics.hooks import swanlab_log

logger = logging.getLogger(__name__)


@HOOKS.register_module(force=True)
class FeatureBridgeDiagnosticsHook(Hook):
    """FBM 诊断 Hook.

    Args:
        interval: 采样间隔 (iter), 默认 50
        log_gate: 是否记录 gate 统计
        log_norms: 是否记录特征范数
        log_proj: 是否记录 proj 层参数
        log_gate_grad: 是否记录 gate 梯度
        log_unet_grad: 是否记录 UNet 各 block 梯度
    """

    priority = 'LOW'

    def __init__(
        self,
        interval: int = 50,
        log_gate: bool = True,
        log_norms: bool = True,
        log_proj: bool = True,
        log_gate_grad: bool = True,
        log_unet_grad: bool = True,
    ):
        self.interval = interval
        self.log_gate = log_gate
        self.log_norms = log_norms
        self.log_proj = log_proj
        self.log_gate_grad = log_gate_grad
        self.log_unet_grad = log_unet_grad

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        """训练 iter 后采集 FBM 诊断数据."""
        step = runner.iter
        if step == 0 or step % self.interval != 0:
            return

        model = self._get_model(runner)
        if not hasattr(model, 'get_feature_bridge_diagnostics'):
            return

        try:
            diag = model.get_feature_bridge_diagnostics()
        except Exception as e:
            logger.warning(f'FeatureBridge diagnostics failed: {e}')
            return

        if not diag:
            return

        # 按 category 分组上传
        log_data: Dict[str, float] = {}

        for key, value in diag.items():
            if not isinstance(value, (int, float)):
                continue

            # 分类路由
            if key.startswith('gate_') and self.log_gate:
                log_data[f'fbm/gate/{key}'] = float(value)
            elif key.startswith('ld_') and self.log_norms:
                log_data[f'fbm/norms/{key}'] = float(value)
            elif key.startswith('cg_') and self.log_norms:
                log_data[f'fbm/norms/{key}'] = float(value)
            elif key.startswith('fused_') and self.log_norms:
                log_data[f'fbm/norms/{key}'] = float(value)
            elif key.startswith('param_gate_') and self.log_gate:
                log_data[f'fbm/param_gate/{key}'] = float(value)
            elif key.startswith('param_proj_') and self.log_proj:
                log_data[f'fbm/param_proj/{key}'] = float(value)
            elif key.startswith('grad_gate_') and self.log_gate_grad:
                log_data[f'fbm/grad_gate/{key}'] = float(value)
            elif key.startswith('unet_') and self.log_unet_grad:
                log_data[f'fbm/unet/{key}'] = float(value)
            else:
                log_data[f'fbm/other/{key}'] = float(value)

        if log_data:
            swanlab_log(log_data, step)

    def _get_model(self, runner):
        """获取模型 (处理 DataParallel)."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model
