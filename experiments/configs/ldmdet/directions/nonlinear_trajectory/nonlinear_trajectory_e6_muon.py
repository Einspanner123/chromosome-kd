"""LDMDet + Nonlinear Trajectory E6.2-Muon: Muon 混合优化器

方向六优化器改进实验 2: 在方向四最佳配置 (E4.3, mAP=0.752) 基础上,
将 bbox_head 的 2D 矩阵参数 (transformer attention, FFN, dynamic conv) 切换到
Muon 优化器 (Newton-Schulz 正交化动量), backbone/neck 保持 AdamW.

改进点:
- MuonHybrid 优化器: 对 2D 矩阵参数正交化动量, 提高训练稳定性
  * bbox_head.head_series (transformer layers) 2D 权重 → Muon
  * bbox_head.head_series (transformer layers) 1D 偏置 → AdamW
  * backbone (ResNet conv 4D 权重) → AdamW
  * neck (FPN conv 4D 权重) → AdamW
- Muon 学习率 0.02 (Muon 推荐), AdamW 学习率 5e-5 (与 baseline 一致)
- Newton-Schulz 5 步迭代, momentum=0.95, Nesterov

对比:
- E4.3 baseline (nonlinear_trajectory.py): AdamW, mAP=0.752, 后期波动 ±1.0%
- E6.2-Muon (本配置): MuonHybrid, 预期 mAP↑, 训练更稳定

理论依据:
- Muon 对 2D 矩阵参数的更新方向正交化, 避免梯度方向震荡
- 正交化后谱归一化, 更新幅度与 AdamW 量级一致 (RMS-aligned scaling)
- backbone/neck 保持 AdamW (4D Conv 权重不适合 Muon 正交化)
"""

_base_ = ['nonlinear_trajectory.py']

# ── 降低 batch_size 适配 16GB GPU (NS 正交化需要额外显存) ──
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = dict(batch_size=2)

# ── MuonHybrid 优化器 ─────────────────────────────
# 使用 MuonHybridConstructor 自动路由参数:
# - bbox_head 的 2D 参数 → Muon (use_muon=True)
# - bbox_head 的 1D 参数 + backbone/neck → AdamW (use_muon=False)
optim_wrapper = dict(
    constructor='MuonHybridConstructor',
    optimizer=dict(
        type='MuonHybrid',
        lr=5e-5,                 # AdamW 学习率 (backbone/neck/bias)
        muon_lr=0.01,            # Muon 学习率 (bbox_head 2D 权重) - 从 0.02 降低, 缓解 mAP 下降
        weight_decay=1e-4,       # 与 baseline 一致
        momentum=0.95,           # Muon 动量
        betas=(0.9, 0.999),      # AdamW betas
        eps=1e-8,
        ns_steps=5,              # Newton-Schulz 迭代步数
        nesterov=True,           # Muon Nesterov 动量
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        # 强制 backbone/neck 使用 AdamW (4D Conv 权重不适合 Muon)
        adam_patterns=['backbone.', 'neck.'],
        # DynamicConv 的 dynamic_layer (32768,256) 和 out_layer (256,12544)
        # 过大, NS 中间矩阵 (32768,32768) 会 OOM, 强制使用 AdamW
        custom_keys={
            'inst_interact.dynamic_layer': dict(use_muon=False),
            'inst_interact.out_layer': dict(use_muon=False),
        },
        # bbox_head 其他 2D 参数自动使用 Muon (transformer attention, FFN)
        muon_patterns=[],
    ),
)
