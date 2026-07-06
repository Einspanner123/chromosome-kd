"""指数移动平均（EMA）模型

支持 warmup：decay 从 decay_start 渐进到 decay，避免训练早期 EMA 被随机初始化拖偏。
"""

import torch


class EMAModel:
    """指数移动平均

    Args:
        model: 被跟踪的模型
        decay: 最终 EMA decay（如 0.9999）
        decay_start: warmup 起点 decay（如 0.999），warmup 期间 decay 从此值渐进到 decay
        warmup_steps: warmup 步数，超过此步数后 decay 固定为 decay
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        decay_start: float = None,
        warmup_steps: int = 0,
    ):
        self.decay = decay
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.cur_step = 0
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.backup = {}

    def get_current_decay(self) -> float:
        """返回当前 step 的有效 decay

        warmup 期间：从 decay_start 线性渐进到 decay
        warmup 后：固定为 decay
        """
        if self.warmup_steps <= 0 or self.decay_start is None:
            return self.decay
        if self.cur_step >= self.warmup_steps:
            return self.decay
        # 线性渐进
        progress = self.cur_step / self.warmup_steps
        return self.decay_start + (self.decay - self.decay_start) * progress

    def update(self, model: torch.nn.Module):
        """更新 EMA shadow 参数"""
        cur_decay = self.get_current_decay()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(cur_decay).add_(
                    param.data, alpha=1 - cur_decay
                )
        self.cur_step += 1

    def apply_shadow(self, model: torch.nn.Module):
        """将 EMA 参数应用到模型（用于评估/推理）"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """恢复原始参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
