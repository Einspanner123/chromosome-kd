# FBM 改进方案：借鉴 GDD (CVPR 2025) 的特征对齐与知识迁移技术

> **创建时间**: 2026-07-02
> **关联实验**: E6.2-E6.4 (chromo 单域, 已证伪) + 跨数据集预训练 (24obj → chromo, 运行中)
> **参考论文**: Generalized Diffusion Detector (GDD), CVPR 2025
> **参考代码**: https://github.com/heboyong/Generalized-Diffusion-Detector
> **目标**: 以实现目标最优为最高目的，不考虑改造成本

---

## 1. 背景

### 1.1 FBM 现状

FBM (Feature Bridge Module) 将 ChromoGen 生成模型的 UNet 特征注入 LDMDet FPN，旨在为检测器提供染色体领域感知特征。

**chromo 单域实验 (E6.2-E6.4, 已证伪)**:

| 实验 | 融合方式 | mAP | vs baseline (0.746) | 问题 |
|------|---------|-----|---------------------|------|
| E6.2 frozen | 标量 alpha gate (零初始化) | 0.737 | -0.009 | alpha 趋零, CG 特征被忽略 |
| E6.3 enhanced | per-channel sigmoid gate (0.1初始化) | 0.703 | -0.043 | gate 不学习, 模型拒绝 CG 特征 |
| E6.3b frozen_enhanced | 同 E6.3 但 UNet 全冻结 | 0.696 | -0.050 | 隔离测试, FBM 架构本身无效 |
| E6.4 crossattn | Cross-Attention + gamma 零初始化 | 0.733 | -0.013 | gamma 零初始化致 attention 路径未激活 |

**跨数据集预训练 (24obj → chromo, 运行中)**:

| 实验 | 融合方式 | mAP | 状态 |
|------|---------|-----|------|
| FBM SimpleGate | simple gate | — | OOM 失败 (非性能问题) |
| FBM CrossAttn | cross-attention | 0.810 @ ep10 | 运行中 (24obj 源预训练) |

### 1.2 核心问题

FBM 失败的表层原因: gamma 零初始化、gate 不学习、特征分布不匹配。
深层原因: **ChromoGen (生成优化) 与 LDMDet FPN (判别优化) 的特征空间异构**, 现有融合方式 (simple gate / cross-attention) 无法有效桥接这种异构性。

---

## 2. GDD 论文技术分析

### 2.1 论文概述

**标题**: Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection
**会议**: CVPR 2025
**作者**: Boyong He et al., 厦门大学

**核心思路**: 从 Stable Diffusion (SD-1.5) 的去噪过程中提取多时间步中间特征 → 构建域不变检测器 (F_diff) → 通过特征级和目标级对齐, 将泛化能力蒸馏到轻量检测器 (F_comm)。

### 2.2 GDD 三项关键技术

#### 2.2.1 特征级对齐: PCC (Pearson Correlation Coefficient)

**问题**: 扩散特征和常规检测器特征的幅度分布不同 (heterogeneous detectors), MSE 强制幅度匹配导致特征支配 (feature dominance)。

**GDD 方案**: 用 PCC 替代 MSE, 只关注相关性模式, 对幅度不变。

```python
# GDD 的 PCC 对齐损失 (KDLoss)
def pcc_loss(student_feat, teacher_feat):
    """Pearson Correlation Coefficient based loss"""
    # 标准化: 零均值单位方差
    s_mean = student_feat.mean(dim=-1, keepdim=True)
    s_std = student_feat.std(dim=-1, keepdim=True)
    s = (student_feat - s_mean) / (s_std + 1e-6)

    t_mean = teacher_feat.mean(dim=-1, keepdim=True)
    t_std = teacher_feat.std(dim=-1, keepdim=True)
    t = (teacher_feat - t_mean) / (t_std + 1e-6)

    # PCC = 两个标准化向量的内积均值
    pcc = (s * t).mean(dim=-1)
    return (1 - pcc).mean()  # 最大化 PCC = 最小化 1-PCC
```

**GDD 还使用特征对齐 (align_features)**: 在计算 cross-loss 前, 将 student 特征的均值/方差对齐到 teacher 特征, 消除幅度差异:

```python
def align_features(input_feats, refer_feats):
    """将 input 特征的统计量对齐到 refer 特征"""
    aligned = []
    for inp, ref in zip(input_feats, refer_feats):
        N, C, H, W = inp.size()
        inp_flat = inp.permute(1, 0, 2, 3).reshape(C, -1)
        ref_flat = ref.permute(1, 0, 2, 3).reshape(C, -1)

        inp_mean = inp_flat.mean(dim=-1, keepdim=True)
        inp_std = inp_flat.std(dim=-1, keepdim=True)
        ref_mean = ref_flat.mean(dim=-1, keepdim=True)
        ref_std = ref_flat.std(dim=-1, keepdim=True)

        # 标准化 input, 再用 refer 的统计量还原
        aligned_feat = (inp_flat - inp_mean) / (inp_std + 1e-6)
        aligned_feat = aligned_feat * ref_std + ref_mean
        aligned_feat = aligned_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
        aligned.append(aligned_feat)
    return tuple(aligned)
```

#### 2.2.2 目标级对齐: 共享检测头

**GDD 方案**: Teacher (diffusion detector) 和 Student (conventional detector) 共享 RPN proposals, 双向 cross-loss:

```python
# GDD 的 cross-loss (双向)
def loss_cross(self, batch_inputs, batch_data_samples):
    student_x = self.model.student.extract_feat(batch_inputs)
    dift_x = self.model.dift_detector.extract_feat(batch_inputs)

    # Cross-loss 1: student 特征 → teacher 检测头
    if "student_to_dift" in self.cross_type:
        aligned_student_x = self.align_features(student_x, dift_x)
        losses.update(self.cross_loss_student_to_dift(batch_data_samples, aligned_student_x))

    # Cross-loss 2: teacher 特征 → student 检测头
    if "dift_to_student" in self.cross_type:
        aligned_dift_x = self.align_features(dift_x, student_x)
        losses.update(self.cross_loss_dift_to_student(batch_data_samples, aligned_dift_x))

    # Feature KD loss (PCC)
    for s_feat, d_feat in zip(student_x, dift_x):
        feature_loss += self.feature_loss(s_feat, d_feat) / len(dift_x)

    # Object-level: 共享 RPN proposals
    _, rpn_results = self.model.student.rpn_head.loss_and_predict(student_x, ...)
    roi_losses_kd = self.roi_head_loss_with_kd(student_x, dift_x, rpn_results, ...)
    return losses
```

**关键设计**:
- Student RPN 生成 proposals → 同时喂给 student 和 teacher 的 ROI head
- 两个检测头对同一 proposals 做分类和回归
- 双向 loss 强制特征空间一致

#### 2.2.3 多层级特征投影 (HyperFeatureEncoder)

GDD 从 SD UNet 的多个 down/mid/up blocks 提取特征, 投影到 FPN 4 级:

```python
# GDD 的 HyperFeatureEncoder 配置
dift_config = dict(
    projection_dim=[2048, 1024, 512, 256],  # 匹配 FPN P2-P5
    model_id="../stable-diffusion-v1-5",
    diffusion_mode="inversion",
    input_resolution=[512, 512],
    scheduler_timesteps=[80, 60, 40, 20, 1],  # 多时间步
    save_timestep=[4, 3, 2, 1, 0],
    num_timesteps=5,
    idxs=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],
          [2,0],[2,1],[2,2],[3,0],[3,1],[3,2]],  # UNet 各 block
)
```

### 2.3 GDD 与我们的差异

**目标差异**: GDD 解决域泛化 (跨视觉域, 如 Cityscapes → FoggyCityscapes), 我们解决同域迁移 (24obj → chromo, 都是染色体)。

**特征源差异**: GDD 用预训练 SD-1.5 (自然图像, 域不变), 我们用 ChromoGen (染色体数据, 领域特定)。预训练 SD 对染色体无语义价值, ChromoGen 反而更合适。

**结论**: GDD 的"域不变性"理论不适用于我们的场景, 但其**特征对齐和知识迁移的具体技术方法**对 FBM 有直接参考价值。

---

## 3. 完整改进方案

以实现目标最优为最高目的, 不考虑改造成本。分三个阶段递进:

### Phase 1: PCC 特征对齐损失 (优先级最高)

#### 3.1.1 问题诊断

当前 FBM 的融合方式:
- Simple gate: `fused = ld * (1-g) + g * cg_proj` — 线性混合, 不约束特征分布
- Cross-attention: LD=Q, CG=K/V — 理论可学习对齐, 但 gamma 零初始化失效, 且 attention 本身不约束分布对齐

**核心缺陷**: 没有任何损失项约束 ChromoGen 特征和 LDMDet 特征的**相关性模式**一致, 模型无法学到有意义的特征对齐。

#### 3.1.2 方案: 增加 PCC 辅助损失

在 FBM 融合之外, 增加 PCC 对齐损失作为辅助训练信号:

```python
class PCCAlignmentLoss(nn.Module):
    """Pearson Correlation Coefficient 特征对齐损失

    参考 GDD (CVPR 2025): 用 PCC 替代 MSE 对齐异构检测器特征,
    只关注相关性模式, 对幅度分布不变。
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: LDMDet FPN 特征 [B, C, H, W]
            teacher_feat: ChromoGen 投影后特征 [B, C, H, W]

        Returns:
            loss: 1 - PCC, 最大化 PCC = 最小化 loss
        """
        # 展平为 [B, C, H*W]
        B, C, H, W = student_feat.shape
        s = student_feat.reshape(B, C, -1)  # [B, C, HW]
        t = teacher_feat.reshape(B, C, -1)

        # 标准化: 零均值单位方差 (沿空间维度)
        s_mean = s.mean(dim=-1, keepdim=True)
        s_std = s.std(dim=-1, keepdim=True)
        s_norm = (s - s_mean) / (s_std + 1e-6)

        t_mean = t.mean(dim=-1, keepdim=True)
        t_std = t.std(dim=-1, keepdim=True)
        t_norm = (t - t_mean) / (t_std + 1e-6)

        # PCC = 标准化向量的余弦相似度 (沿空间维度)
        pcc = (s_norm * t_norm).mean(dim=-1)  # [B, C]
        return (1 - pcc).mean() * self.loss_weight
```

#### 3.1.3 集成到 head.py

```python
# DiffusionDetHead.loss() 中增加
if self.feature_bridge is not None and self.pcc_loss is not None:
    # ldmdet_fpn: backbone → FPN 的原始特征 (融合前)
    # cg_fpn: ChromoGen 特征经投影后的特征 (融合前的 CG 侧)
    # fused: 融合后的特征
    #
    # PCC 对齐: 约束 LD 和 CG 投影特征的相关性模式一致
    for level in range(len(ldmdet_fpn)):
        if cg_proj_feats[level] is not None:
            pcc_loss = self.pcc_loss(ldmdet_fpn[level], cg_proj_feats[level])
            losses[f'pcc_align_L{level}'] = pcc_loss
```

#### 3.1.4 同时修复 gamma 初始化

```python
# E6.4 问题: gamma_down3 = nn.Parameter(torch.zeros(1))
# 修复: 非零初始化 (0.1), 提供非零梯度信号激活 cross-attention 路径
self.gamma_down3 = nn.Parameter(torch.full((1,), 0.1))
```

#### 3.1.5 预期效果

- PCC 损失约束 LD 和 CG 特征的相关性模式对齐, 解决幅度分布不匹配
- gamma 非零初始化激活 cross-attention 路径, 避免退化为 simple gate
- 辅助损失引导 gate/gamma 朝有意义的方向学习, 而非趋零

---

### Phase 2: 目标级对齐 (共享检测头 + Cross-loss)

#### 3.2.1 问题诊断

当前 FBM 只有特征级融合, 缺乏目标级 (task-level) 监督。ChromoGen 特征注入 FPN 后直接进入 LDMDet 检测头, 没有"ChromoGen 特征本身能否做好检测"的独立监督信号。

#### 3.2.2 方案: 为 ChromoGen 特征接独立检测头

GDD 的 teacher 本身是检测器 (用扩散特征的 Faster R-CNN), 可直接共享检测头。我们的 ChromoGen 不是检测器, 需要**额外接一个检测头**做目标级监督。

```
架构:
                    ┌─────────────────────────────────┐
                    │  ChromoGen UNet (frozen/partial) │
                    └──────────┬──────────────────────┘
                               │ cg_feats
                    ┌──────────▼──────────┐
                    │  CG Feature Proj    │  (多层级投影)
                    └──────────┬──────────┘
                               │ cg_proj_feats
                    ┌──────────▼──────────┐
  LDMDet FPN ──────►│  FBM Fusion         │◄──── PCC 对齐损失
                    └──────────┬──────────┘
                               │ fused_feats
                    ┌──────────▼──────────┐
                    │  LDMDet Head (主)   │  ← 原始检测头
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  CG Aux Head (辅助) │  ← 新增: ChromoGen 特征的独立检测头
                    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Cross-loss 双向    │
                    │  1. cg_feats → LDMDet Head
                    │  2. ld_feats → CG Aux Head
                    └─────────────────────┘
```

#### 3.2.3 CG Aux Head 实现

```python
class CGAuxDetectionHead(nn.Module):
    """ChromoGen 特征的辅助检测头

    用于目标级对齐: 让 ChromoGen 特征独立做检测,
    提供任务级监督信号, 引导 CG 特征朝检测友好方向优化。

    架构: 轻量级检测头 (与 LDMDet SingleDiffusionDetHead 同构),
    但仅用于训练 (推理时不使用)。
    """

    def __init__(self, num_classes, feat_channels=256, num_proposals=500):
        super().__init__()
        # 复用 SingleDiffusionDetHead 结构
        # 但参数独立, 不与 LDMDet 主检测头共享
        self.cls_head = nn.Linear(feat_channels, num_classes)
        self.reg_head = nn.Linear(feat_channels, 4)
        # ... (完整实现参考 SingleDiffusionDetHead)

    def forward(self, cg_proj_feats, proposals, time_emb):
        """从 CG 特征 + 共享 proposals 做检测

        Args:
            cg_proj_feats: ChromoGen 投影后特征 [B, C, H, W]
            proposals: 来自 LDMDet 主检测头的 proposals (共享)
            time_emb: 时间嵌入
        """
        # ROI extract → cls + reg
        ...
```

#### 3.2.4 Cross-loss 双向对齐

```python
def loss_with_cross_alignment(self, features, img_metas, gt_bboxes, gt_labels):
    """带双向 cross-loss 的训练逻辑"""

    # 1. 提取两路特征
    ldmdet_fpn = self.backbone_neck(features)  # LDMDet 原始 FPN
    cg_feats = self.chromogen_extractor(features)  # ChromoGen 特征
    cg_proj = self.feature_bridge.proj_only(cg_feats)  # 投影 (不融合)

    # 2. 特征级: PCC 对齐损失
    pcc_loss = self.pcc_loss(ldmdet_fpn, cg_proj)

    # 3. 融合特征
    fused_fpn = self.feature_bridge(ldmdet_fpn, cg_feats)

    # 4. LDMDet 主检测头 (用融合特征)
    ld losses = self.ldmdet_head.loss(fused_fpn, img_metas, gt_bboxes, gt_labels)

    # 5. 目标级: Cross-loss 双向
    # 5a. CG 特征 → LDMDet 检测头 (共享 proposals)
    aligned_cg = self.align_features(cg_proj, ldmdet_fpn)  # 统计量对齐
    cg_to_ld_losses = self.ldmdet_head.loss(aligned_cg, img_metas, gt_bboxes, gt_labels)
    # 重命名避免冲突
    cg_to_ld_losses = {f'cg_to_ld_{k}': v for k, v in cg_to_ld_losses.items()}

    # 5b. LDMDet 特征 → CG 辅助检测头
    aligned_ld = self.align_features(ldmdet_fpn, cg_proj)
    ld_to_cg_losses = self.cg_aux_head.loss(aligned_ld, img_metas, gt_bboxes, gt_labels)
    ld_to_cg_losses = {f'ld_to_cg_{k}': v for k, v in ld_to_cg_losses.items()}

    # 6. ROI 级 KD: 共享 proposals, 对齐 ROI 特征
    # proposals 来自 LDMDet 主检测头
    roi_kd_loss = self.roi_kd_loss(fused_fpn, cg_proj, proposals, ...)

    return {**ld_losses, **cg_to_ld_losses, **ld_to_cg_losses,
            'pcc_align': pcc_loss, 'roi_kd': roi_kd_loss}
```

#### 3.2.5 预期效果

- CG 特征有独立检测监督, 不再只是"被动注入"
- Cross-loss 双向对齐强制两路特征空间一致
- ROI 级 KD 在目标级别 (而非仅特征级别) 对齐

---

### Phase 3: 多层级特征投影 + 特征融合增强

#### 3.3.1 问题诊断

当前 FBM 只取 ChromoGen UNet 的 down1/down2/down3/mid 四层, 投影方式简单 (1×1 conv + GroupNorm + GELU)。GDD 的 HyperFeatureEncoder 从 SD UNet 的 12 个 block 位置提取特征, 做更丰富的多层级融合。

#### 3.3.2 方案: 增强 ChromoGen 特征提取

```python
class EnhancedChromoGenExtractor(nn.Module):
    """增强的 ChromoGen 特征提取器

    参考 GDD HyperFeatureEncoder:
    - 从 UNet 的更多 block 位置提取特征 (down blocks + up blocks)
    - 多层级投影匹配 FPN 4 级
    - 可选: 多时间步特征 (如果 ChromoGen 支持多步去噪)
    """

    def __init__(self, unet, projection_dims=[256, 256, 256, 256]):
        super().__init__()
        self.unet = unet

        # 从 UNet 的 down blocks 和 up blocks 提取特征
        # down1: H/16, 320ch → P1 (stride 8)
        # down2: H/32, 640ch → P2 (stride 16)
        # down3: H/64, 1280ch → P3 (stride 32)
        # mid:   H/64, 1280ch → P3 (stride 32, 补充)
        # up1:   H/32, 1280ch → P2 (stride 16, 上采样回补)
        # up2:   H/16, 640ch  → P1 (stride 8, 上采样回补)

        self.projections = nn.ModuleDict({
            'down1': self._build_proj(320, 256),
            'down2': self._build_proj(640, 256),
            'down3': self._build_proj(1280, 256),
            'mid':   self._build_proj(1280, 256),
            'up1':   self._build_proj(1280, 256),  # 新增: up blocks
            'up2':   self._build_proj(640, 256),   # 新增: up blocks
        })

    def _build_proj(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.GroupNorm(32, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        """提取多层级 ChromoGen 特征

        Returns:
            dict: 投影后的多层级特征, 匹配 FPN 4 级
        """
        # UNet 前向, 收集中间特征
        cg_raw = self.unet.extract_features(x)

        # 投影 + 融合 (down + up 对应层级拼接)
        feats = {}
        feats['L1'] = self.projections['down1'](cg_raw['down1'])  # H/16 → upsample to H/8
        feats['L1'] = feats['L1'] + F.interpolate(
            self.projections['up2'](cg_raw['up2']),  # H/16 → upsample
            size=feats['L1'].shape[2:], mode='bilinear', align_corners=False
        )

        feats['L2'] = self.projections['down2'](cg_raw['down2'])
        feats['L2'] = feats['L2'] + F.interpolate(
            self.projections['up1'](cg_raw['up1']),
            size=feats['L2'].shape[2:], mode='bilinear', align_corners=False
        )

        feats['L3'] = self.projections['down3'](cg_raw['down3'])
        feats['L3'] = feats['L3'] + self.projections['mid'](cg_raw['mid'])

        return feats
```

#### 3.3.3 特征融合增强: Deformable Cross-Attention

替换当前 Level 3 的标准 cross-attention 为 deformable attention, 降低计算量并支持多尺度:

```python
class DeformableCrossAttnBridge(nn.Module):
    """Deformable Cross-Attention 特征桥接

    相比标准 cross-attention:
    - 只在参考点附近采样, 计算 O(N*K) 而非 O(N*M)
    - 支持多尺度特征 (CG down3 + mid + up1 同时作为 K/V)
    - 可扩展到 Level 1/2 (标准 attention 在大分辨率上不可行)
    """

    def __init__(self, ld_channels=256, num_heads=4, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points

        # Deformable attention 参数
        self.sampling_offsets = nn.Linear(ld_channels, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(ld_channels, num_heads * num_points)
        self.value_proj = nn.Conv2d(ld_channels, ld_channels, 1)
        self.output_proj = nn.Conv2d(ld_channels, ld_channels, 1)

        # zero-init output
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, ld_feat, cg_feats_list):
        """
        Args:
            ld_feat: LDMDet 特征 [B, C, H, W] (Query)
            cg_feats_list: 多个 ChromoGen 特征 [B, C, H', W'] (K/V)
        """
        # 实现 deformable attention...
        pass
```

---

## 4. 实施路线图

### 4.1 优先级排序 (以目标最优为准)

| 阶段 | 改进 | 复杂度 | 预期收益 | 依赖 |
|------|------|--------|---------|------|
| **Phase 1A** | gamma 非零初始化 (0.1) | 极低 | 激活 cross-attention 路径 | 无 |
| **Phase 1B** | PCC 特征对齐损失 | 低 | 解决幅度分布不匹配 | Phase 1A |
| **Phase 2A** | CG Aux 检测头 | 中 | 目标级监督 | Phase 1B |
| **Phase 2B** | Cross-loss 双向对齐 | 中 | 强制特征空间一致 | Phase 2A |
| **Phase 2C** | ROI 级 KD | 中 | 目标级知识迁移 | Phase 2A |
| **Phase 3A** | 增强 ChromoGen 特征提取 (up blocks) | 中 | 更丰富多层级特征 | Phase 1B |
| **Phase 3B** | Deformable Cross-Attention | 高 | 多尺度高效融合 | Phase 3A |

### 4.2 实验设计

#### 4.2.1 chromo 单域验证 (快速迭代)

```
baseline: rf_heun_adaln = 0.746
E6.2 (simple gate, alpha=0) = 0.737  [-0.009]
E6.3 (per-channel gate, 0.1 init) = 0.703  [-0.043]
E6.4 (cross-attn, gamma=0) = 0.733  [-0.013]

改进实验:
E6.5 (cross-attn, gamma=0.1)              → 验证 gamma 修复效果
E6.6 (E6.5 + PCC loss)                    → 验证 PCC 对齐效果
E6.7 (E6.6 + CG Aux Head + cross-loss)    → 验证目标级对齐效果
E6.8 (E6.7 + enhanced CG extractor)       → 验证多层级增强效果
E6.9 (E6.8 + deformable cross-attn)       → 完整方案
```

#### 4.2.2 跨数据集验证 (最终目标)

```
源域预训练 (24obj):
  baseline: ldmdet_sota_24obj (无 FBM)
  E7.1 (E6.9 FBM + PCC + cross-loss)  → 24obj 源预训练

目标域微调 (chromo):
  baseline: 24obj pretrained → chromo finetune
  E7.2 (E7.1 pretrained → chromo finetune)  → 验证跨数据集迁移效果
```

### 4.3 消融实验

| 消融项 | 目的 |
|--------|------|
| 去掉 PCC loss | 验证 PCC 对齐的必要性 |
| 去掉 CG Aux Head | 验证目标级监督的必要性 |
| 去掉 cross-loss (只保留单向) | 验证双向对齐的必要性 |
| gamma=0 vs gamma=0.1 | 验证 gamma 初始化的影响 |
| CG frozen vs partial unfreeze | 验证 ChromoGen 解冻策略 |
| 单层级 vs 多层级 CG 特征 | 验证多层级投影的效果 |

---

## 5. 与现有代码的集成点

### 5.1 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `ldmdet/feature_bridge/feature_bridge_module.py` | gamma 非零初始化 + PCC 损失接口 |
| `ldmdet/feature_bridge/cross_attn_bridge.py` | gamma 非零初始化 + PCC 损失接口 |
| `ldmdet/feature_bridge/cg_aux_head.py` (新建) | CG 辅助检测头 |
| `ldmdet/feature_bridge/enhanced_extractor.py` (新建) | 增强 ChromoGen 特征提取 |
| `ldmdet/feature_bridge/pcc_loss.py` (新建) | PCC 对齐损失 |
| `projects/LDMDet/model.py` | 集成 PCC loss + CG Aux Head + cross-loss |
| `experiments/configs/few_shot/source_pretrain/ldmdet_fbm_crossattn_24obj.py` | 新增配置项 |

### 5.2 配置设计

```python
# experiments/configs/few_shot/source_pretrain/ldmdet_fbm_v2.py
model = dict(
    type='LDMDet',
    backbone=...,
    neck=...,
    bbox_head=dict(
        ...,
        # FBM 改进配置
        feature_bridge=dict(
            type='CrossAttnFeatureBridgeModule',
            cg_channels=(320, 640, 1280, 1280),
            ld_channels=256,
            num_heads=4,
            gamma_init=0.1,  # 修复: 非零初始化
        ),
        chromogen_extractor=dict(
            type='EnhancedChromoGenExtractor',
            extract_up_blocks=True,  # Phase 3: 提取 up blocks
        ),
        # PCC 对齐损失
        pcc_loss=dict(
            type='PCCAlignmentLoss',
            loss_weight=0.1,
        ),
        # CG 辅助检测头
        cg_aux_head=dict(
            type='CGAuxDetectionHead',
            num_classes=24,
            feat_channels=256,
        ),
        # Cross-loss 配置
        cross_loss=dict(
            enable_cross_loss=True,
            cross_type=['student_to_dift', 'dift_to_student'],
            enable_feature_loss=True,
            feature_loss_type='pkd',
            feature_loss_weight=0.1,
        ),
    ),
)
```

---

## 6. 风险与注意事项

### 6.1 显存风险

- CG Aux Head 增加一个完整检测头, 显存开销显著增加
- Cross-loss 需要两路前向 (LD + CG), 显存翻倍
- **缓解**: 使用梯度检查点 (gradient checkpointing), 或仅在 Level 3 做 cross-loss

### 6.2 训练稳定性

- PCC 损失在训练早期可能不稳定 (特征统计量变化大)
- **缓解**: warmup 策略, 前 10 epoch 不加 PCC loss, 之后线性增加到目标权重

### 6.3 过拟合风险

- CG Aux Head + cross-loss 增加大量参数, 在小数据集 (chromo) 上可能过拟合
- **缓解**: 跨数据集场景 (24obj 源预训练) 数据量更大, 优先在该场景验证

### 6.4 ChromoGen 生成 vs 判别冲突

- ChromoGen 优化目标是重建 (生成), 不是判别 (检测)
- 即使有 cross-loss, CG 特征可能仍不适合检测
- **缓解**: Phase 3 的 enhanced extractor 可选择性解冻 UNet 后几层, 用检测 loss 微调

---

## 7. 总结

GDD (CVPR 2025) 的域泛化目标不适用于我们 (同域迁移), 但其三项核心技术对 FBM 有直接参考价值:

1. **PCC 特征对齐** (Phase 1): 解决 ChromoGen 和 LDMDet 特征幅度分布不匹配 — **最优先**
2. **目标级对齐 + Cross-loss** (Phase 2): 为 CG 特征提供独立检测监督 — **核心突破**
3. **多层级特征投影** (Phase 3): 更丰富的 ChromoGen 特征提取 — **锦上添花**

**预期**: Phase 1 (PCC + gamma 修复) 即可解决 E6.4 的 gamma 失效问题, 可能带来正向收益。Phase 2 (目标级对齐) 是完整方案的关键, 使 FBM 从"被动特征注入"变为"主动知识迁移"。Phase 3 在前两阶段验证有效后进一步优化。
