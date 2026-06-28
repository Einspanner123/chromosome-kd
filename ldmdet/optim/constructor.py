"""MuonHybridOptimWrapperConstructor — 自定义参数路由构造器

将模型参数按名称模式和维度分类为 Muon 组 (2D 矩阵) 和 AdamW 组 (1D/4D 参数)。

默认路由规则:
- backbone.*, neck.* → AdamW (4D Conv2d 权重 + 1D BN)
- bbox_head.* 且 ndim==2 → Muon (2D Linear/Attention 权重)
- bbox_head.* 且 ndim!=2 → AdamW (1D bias/LayerNorm)
- 其他 → 2D 用 Muon, 否则 AdamW

可通过 paramwise_cfg.custom_keys 覆盖:
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(use_muon=False),
            'neck': dict(use_muon=False),
        },
    )
"""

import logging
from typing import List

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.optim.optimizer.default_constructor import DefaultOptimWrapperConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MuonHybridConstructor(DefaultOptimWrapperConstructor):
    """Muon 混合优化器的参数路由构造器

    在 DefaultOptimWrapperConstructor 基础上, 根据参数名和维度自动设置
    param_group['use_muon'] 标志, 实现 Muon/AdamW 自动路由。

    paramwise_cfg 额外支持:
    - muon_patterns (list[str]): 强制使用 Muon 的参数名子串列表
    - adam_patterns (list[str]): 强制使用 AdamW 的参数名子串列表
    """

    def add_params(self, params: List[dict], module: nn.Module, prefix: str = '',
                   is_dcn_module=None) -> None:
        """递归添加参数, 自动设置 use_muon 标志"""
        # 获取自定义路由配置
        muon_patterns = self.paramwise_cfg.get('muon_patterns', [])
        adam_patterns = self.paramwise_cfg.get('adam_patterns', [])
        # 默认 adam_patterns 包含 backbone 和 neck
        if not adam_patterns:
            adam_patterns = ['backbone.', 'neck.']

        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)

        # 判断是否是 norm 层
        from mmengine.utils.dl_utils.parrots_wrapper import (
            _BatchNorm, _InstanceNorm,
        )
        from torch.nn import GroupNorm, LayerNorm
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            full_name = f'{prefix}.{name}' if prefix else name

            if bypass_duplicate and self._is_in(param_group, params):
                print_log(
                    f'{full_name} is duplicate. It is skipped since '
                    f'bypass_duplicate={bypass_duplicate}',
                    logger='current', level=logging.WARNING)
                continue

            if not param.requires_grad:
                print_log(
                    f'{full_name} is skipped since its '
                    f'requires_grad={param.requires_grad}',
                    logger='current', level=logging.WARNING)
                continue

            # 1. 先检查 custom_keys (优先级最高, 用户显式指定)
            is_custom = False
            for key in sorted_keys:
                if key in full_name:
                    is_custom = True
                    custom_cfg = custom_keys[key]
                    lr_mult = custom_cfg.get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_cfg.get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    # 传递所有自定义配置 (包括 use_muon)
                    for k, v in custom_cfg.items():
                        if k not in ('lr_mult', 'decay_mult'):
                            param_group[k] = v
                    break

            # 2. 若未匹配 custom_keys, 按 muon_patterns/adam_patterns 路由
            if not is_custom:
                use_adam = any(pat in full_name for pat in adam_patterns)
                use_muon = False
                if not use_adam and muon_patterns:
                    use_muon = any(pat in full_name for pat in muon_patterns)

                # 默认规则: 2D 且不在 backbone/neck → Muon
                if not use_adam and not use_muon:
                    if param.ndim == 2:
                        use_muon = True
                    else:
                        use_adam = True

                param_group['use_muon'] = use_muon

                # norm 层 weight_decay 处理 (继承父类逻辑)
                if is_norm and self.base_wd is not None:
                    norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult')
                    if norm_decay_mult is not None:
                        param_group['weight_decay'] = self.base_wd * norm_decay_mult

            params.append(param_group)

        # 递归处理子模块
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            # 检测 DCN 模块 (继承父类逻辑)
            is_dcn = self._is_dcn_module(child_mod)
            self.add_params(params, child_mod, prefix=child_prefix,
                            is_dcn_module=is_dcn)

    def _is_dcn_module(self, module) -> bool:
        """简化版 DCN 检测 (实际 DCN 不在 LDMDet 中使用)"""
        return False
