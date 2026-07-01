"""ChromoGenFeatureExtractor — 从 ChromoGen UNet 编码器提取多尺度特征

方向六：生成模型感知迁移。

由于 ChromoGen UNet 是冻结的生成模型，特征提取器负责:
1. 将 UNet 设为 eval 模式并冻结参数
2. 在 forward 时收集 down_block/mid_block 的中间特征
3. 以 no_grad 方式提取 (节省显存)

兼容两种 UNet:
- diffusers UNet2DConditionModel (通过 forward_hook 提取)
- MockUNet (测试用, 通过 _feature_cache 属性提取)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class UNetFeatureConfig:
    """UNet 特征提取配置

    Args:
        extract_down1: 是否提取 down1 (320ch, H/8)
        extract_down2: 是否提取 down2 (640ch, H/16)
        extract_down3: 是否提取 down3 (1280ch, H/32)
        extract_mid: 是否提取 mid (1280ch, H/64)
    """

    extract_down1: bool = True
    extract_down2: bool = True
    extract_down3: bool = True
    extract_mid: bool = True


class ChromoGenFeatureExtractor(nn.Module):
    """从 ChromoGen UNet 编码器提取多尺度特征

    Args:
        unet: ChromoGen UNet 模型 (diffusers UNet2DConditionModel 或兼容模型)
        config: 特征提取配置
    """

    def __init__(self, unet=None, config=None, unet_cfg=None, checkpoint=None):
        """初始化特征提取器

        Args:
            unet: ChromoGen UNet 模型实例 (可选, 与 unet_cfg 二选一)
            config: UNetFeatureConfig 实例或 dict (可选)
            unet_cfg: UNet 构建配置 dict (可选, 用于通过 MODELS.build 构建)
            checkpoint: UNet 权重路径 (可选, 加载预训练权重)
        """
        super().__init__()

        # 构建 UNet
        if unet is None and unet_cfg is not None:
            from mmdet.registry import MODELS

            unet = MODELS.build(unet_cfg)
        if unet is None:
            raise ValueError('Must provide unet or unet_cfg')

        self.unet = unet

        # 构建 config
        if config is None:
            self.config = UNetFeatureConfig()
        elif isinstance(config, dict):
            cfg = {k: v for k, v in config.items() if k != 'type'}
            self.config = UNetFeatureConfig(**cfg)
        elif isinstance(config, UNetFeatureConfig):
            self.config = config
        else:
            raise TypeError(f'config must be UNetFeatureConfig or dict, got {type(config)}')

        # 加载权重
        if checkpoint is not None:
            import torch

            state_dict = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # ChromoGenPipeline checkpoint 的 UNet key 格式: 'unet.unet.*'
            # ChromoUNet 包装 diffusers UNet2DConditionModel (self.unet.unet)
            # 需要提取 'unet.unet.' 前缀的 key, 去除前缀后加载
            unet_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('unet.unet.'):
                    unet_state_dict[key[len('unet.unet.'):]] = value
                elif not any(
                    key.startswith(p) for p in ['vae.', 'condition_encoder.', 'bbox_head.']
                ):
                    # 兼容独立 UNet checkpoint (无前缀)
                    unet_state_dict[key] = value

            # 若 self.unet 是 ChromoGenUNet wrapper (内含 self.unet.unet),
            # 需要给 key 加 'unet.' 前缀以匹配 wrapper 的 state_dict
            if hasattr(self.unet, 'unet') and hasattr(self.unet.unet, 'down_blocks'):
                unet_state_dict = {
                    f'unet.{k}': v for k, v in unet_state_dict.items()
                }

            if unet_state_dict:
                missing, unexpected = self.unet.load_state_dict(
                    unet_state_dict, strict=False
                )
                if missing:
                    print(f'[ChromoGenFeatureExtractor] Missing keys: {len(missing)}')
                if unexpected:
                    print(f'[ChromoGenFeatureExtractor] Unexpected keys: {len(unexpected)}')
            else:
                print('[ChromoGenFeatureExtractor] Warning: no UNet keys found in checkpoint')

        self._hooks = []
        self._feature_buffer = {}

        self._register_hooks()
        self.set_frozen()

    def _register_hooks(self):
        """注册 forward hook 收集 UNet 中间特征

        对于 diffusers UNet2DConditionModel:
            down_blocks[0] 输出 → down1 (320ch)
            down_blocks[1] 输出 → down2 (640ch)
            down_blocks[2] 输出 → down3 (1280ch)
            mid_block 输出 → mid (1280ch)

        对于 MockUNet:
            直接读取 _feature_cache 属性
        """
        # 尝试 diffusers UNet 结构
        if hasattr(self.unet, 'down_blocks') and hasattr(self.unet, 'mid_block'):
            block_map = [
                ('down1', 0, self.config.extract_down1),
                ('down2', 1, self.config.extract_down2),
                ('down3', 2, self.config.extract_down3),
            ]
            for name, idx, should_extract in block_map:
                if should_extract:
                    hook = self.unet.down_blocks[idx].register_forward_hook(
                        self._make_hook(name)
                    )
                    self._hooks.append(hook)
            if self.config.extract_mid:
                hook = self.unet.mid_block.register_forward_hook(
                    self._make_hook('mid')
                )
                self._hooks.append(hook)

    def _make_hook(self, name):
        """创建 forward hook"""

        def hook(module, input, output):
            # diffusers DownBlock 输出可能是 tuple (hidden_states, res_samples)
            if isinstance(output, tuple):
                # 取最后一个 res_sample (最深层特征)
                feat = output[0] if len(output) == 1 else output[-1]
                if isinstance(feat, tuple):
                    feat = feat[-1]
            else:
                feat = output
            self._feature_buffer[name] = feat

        return hook

    def set_frozen(self):
        """冻结 UNet 参数并设为 eval 模式

        注意: 若 ChromoGenUNet 已通过 freeze_last_n_blocks 部分解冻,
        此处不再强制覆盖 requires_grad (尊重 UNet 自身的冻结策略)。
        但仍设为 eval 模式以关闭 dropout/BN 更新。
        """
        # 检查 UNet 是否有自定义冻结策略 (ChromoGenUNet wrapper)
        has_custom_freeze = (
            hasattr(self.unet, 'freeze_last_n_blocks')
            and getattr(self.unet, 'freeze_last_n_blocks', 0) != 0
        )
        if not has_custom_freeze:
            # 默认行为: 全部冻结
            for param in self.unet.parameters():
                param.requires_grad = False
        # 始终设为 eval 模式 (关闭 dropout, 但保留 requires_grad)
        self.unet.eval()

    def has_trainable_params(self) -> bool:
        """检查 UNet 是否有可训练参数 (用于决定是否用 no_grad)"""
        return any(p.requires_grad for p in self.unet.parameters())

    def forward(self, latent, timestep, encoder_hidden_states):
        """提取 ChromoGen UNet 编码器特征

        Args:
            latent: VAE 编码后的潜变量 (B, 4, H/8, W/8)
            timestep: 扩散时间步 (B,) 或标量
            encoder_hidden_states: 条件编码 (B, seq_len, 768)

        Returns:
            feats: 字典 {'down1': ..., 'down2': ..., 'down3': ..., 'mid': ...}
                   (仅包含 config 中启用的层)
        """
        self._feature_buffer.clear()

        # 如果 UNet 全部冻结, 用 no_grad 节省显存
        # 如果有解冻参数 (E6.3), 保留梯度以反向传播
        ctx = torch.no_grad() if not self.has_trainable_params() else _dummy_ctx()

        with ctx:
            # 调用 UNet forward (输出被忽略, 特征通过 hook 收集)
            _ = self.unet(
                latent,
                timestep,
                encoder_hidden_states,
                return_dict=True,
            )

            # 如果 hook 未收集到特征 (如 MockUNet), 尝试读取 _feature_cache
            if not self._feature_buffer and hasattr(self.unet, '_feature_cache'):
                cache = self.unet._feature_cache
                for key in ['down1', 'down2', 'down3', 'mid']:
                    extract_flag = getattr(self.config, f'extract_{key}', False)
                    if extract_flag and key in cache:
                        self._feature_buffer[key] = cache[key]

        # 过滤出 config 启用的层
        result = {}
        key_flags = [
            ('down1', self.config.extract_down1),
            ('down2', self.config.extract_down2),
            ('down3', self.config.extract_down3),
            ('mid', self.config.extract_mid),
        ]
        for key, should_extract in key_flags:
            if should_extract and key in self._feature_buffer:
                result[key] = self._feature_buffer[key]

        return result

    def get_unet_grad_diagnostics(self):
        """返回 UNet 各 block 的梯度统计 (仅对解冻的 block 有效)

        用于诊断哪些 UNet block 在被有效训练
        """
        diag = {}
        if not hasattr(self.unet, 'down_blocks'):
            return diag

        with torch.no_grad():
            # down_blocks 梯度
            for i, block in enumerate(self.unet.down_blocks):
                grad_norms = []
                param_count = 0
                trainable_count = 0
                for param in block.parameters():
                    param_count += 1
                    if param.requires_grad:
                        trainable_count += 1
                        if param.grad is not None:
                            grad_norms.append(param.grad.norm().item())
                diag[f'unet_down{i}_trainable'] = trainable_count
                diag[f'unet_down{i}_grad_norm'] = (
                    sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
                )

            # mid_block 梯度
            if hasattr(self.unet, 'mid_block'):
                grad_norms = []
                param_count = 0
                trainable_count = 0
                for param in self.unet.mid_block.parameters():
                    param_count += 1
                    if param.requires_grad:
                        trainable_count += 1
                        if param.grad is not None:
                            grad_norms.append(param.grad.norm().item())
                diag['unet_mid_trainable'] = trainable_count
                diag['unet_mid_grad_norm'] = (
                    sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
                )

        return diag

    def remove_hooks(self):
        """移除所有 forward hook"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class _dummy_ctx:
    """空 context manager (用于不使用 no_grad 的情况)"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
