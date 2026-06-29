# 方向 G：LAMFPN 局部注意力特征金字塔 (Local Attention Module FPN)

> **目标**：用 LAMFPN 替换标准 FPN, 通过局部注意力模块 (LAM) 和双向注意力机制增强特征金字塔的特征表达质量, 提升检测精度。
>
> **瓶颈证据**：
> - 诊断显示 mAP=0.8583 但 mAP_75=0.9677 vs mAP_90=0.4834, 高 IoU 精度不足
> - 分类误差占 43.4% (同组形态相似类混淆: C10↔C9, B5→B4, D13↔D14)
> - 现有 FPN 仅做简单相加 (target + source), 缺乏自适应特征融合
> - 方向 A 的 P1+Deformable 主要解决"小目标分辨率", 而本方向聚焦"特征融合质量"
>
> **当前代码位置**：
> - 现有 FPN: [experiments/configs/ldmdet/rf_heun_adaln.py](../../experiments/configs/ldmdet/rf_heun_adaln.py#L37-L42) 中 `neck=dict(type='FPN', num_outs=4)`
> - LAMFPN 参考实现: [lamfpn.py](../../lamfpn.py), [lamfpn_bifpn.txt](../../lamfpn_bifpn.txt)

---

## 1. 背景与动机

### 1.1 现有 FPN 的局限

标准 FPN 自顶向下路径采用**简单相加**融合相邻层级特征：

```python
laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], ...)
```

这种融合方式的问题:
- **无选择性**: target 和 source 特征等权相加, 无法区分有用信号和噪声
- **无空间感知**: 全局相加, 无法针对不同空间位置调整融合权重
- **单向路径**: 仅 top-down, 缺少 bottom-up 反馈
- **无跨层交互**: 跨多个层级的语义信息无法直接融合

### 1.2 LAMFPN 的核心思路

LAMFPN 通过 3 个核心机制增强特征金字塔:

1. **LAM (Local Attention Module)**: 用 softmax 注意力权重代替简单相加, 自适应融合 target+source 特征
2. **DualAttention**: 通道注意力 (avg+max 共享 MLP) + 空间注意力 (7x7 conv) 双重增强
3. **CrossLayerAttention**: 跨层注意力, 让每个层级融合其他所有层级的信息

### 1.3 与方向 A (P1+Deformable) 的区别

| 维度 | 方向 A (P1+Deformable) | 方向 G (LAMFPN) |
|------|----------------------|----------------|
| **改进点** | 增加 P1 高分辨率层 + 可变形 RoI 采样 | 改进特征金字塔的融合方式 |
| **解决瓶颈** | 小目标分辨率不足 | 特征融合质量差 |
| **机制** | 空间分辨率提升 | 注意力自适应融合 |
| **位置** | FPN 输出层 + RoIExtractor | FPN 内部融合路径 |
| **可组合** | 是 (可叠加 LAMFPN) | 是 (可叠加 P1) |

**两者互补**: 方向 A 解决"看得清", 方向 G 解决"融得好"。

---

## 2. 子方向说明

### G1: 单阶段 LAMFPN (轻量版)

**问题**: 标准 FPN 的 top-down 简单相加无法自适应融合特征。

**方案**: 继承 FPN, 在 top-down 路径的关键层级 (P3, P4) 用 LAMModule 替换简单相加, 并在输出层应用 DualAttention。

**实现** (基于 [lamfpn.py](../../lamfpn.py)):
```python
@MODELS.register_module()
class LAMFPN(FPN):
    def __init__(self, ..., apply_lam_levels=[1, 2], attention_type='dual', ...):
        super().__init__(...)
        # LAM 模块: 仅在指定层级应用 (默认 P3, P4)
        self.lam_modules = nn.ModuleList([
            LAMModule(out_channels, out_channels, ...) if i in apply_lam_levels else None
            for i in range(num_outs - 1)
        ])
        # DualAttention: 通道+空间双重增强
        self.feature_attention = nn.ModuleList([
            DualAttention(out_channels, ...) if i in [1, 2] and attention_type == 'dual' else None
            for i in range(num_outs)
        ])
```

**复杂度**: 低 — 仅修改 FPN 内部, 接口完全兼容, 配置改动最小。

**预期收益**: mAP +0.005~0.015 (特征融合质量提升)。

### G2: 多阶段 BiFPN 风格 LAMFPN (重量版)

**问题**: 单阶段 top-down 路径仅有单向信息流, 特征融合不充分。

**方案**: 采用 BiFPN 风格的多阶段双向 (top-down + bottom-up) 路径, 每个阶段都应用 LAM 和注意力机制。

**实现** (基于 [lamfpn_bifpn.txt](../../lamfpn_bifpn.txt)):
```python
@MODELS.register_module()
class LAMFPN(BaseModule):
    def __init__(self, ..., num_stages=3, num_outs=5, ...):
        # 多阶段堆叠
        self.stages = nn.ModuleList([
            LAMFPNStage(out_channels, ...) for _ in range(num_stages)
        ])

    def forward(self, inputs):
        laterals = [lateral_conv(feat) for feat, lateral_conv in zip(feats, self.lateral_convs)]
        features = laterals
        for stage in self.stages:
            features = stage(features)  # 每个阶段: top-down + bottom-up + attention
        return tuple(self.fpn_convs[i](feat) for i, feat in enumerate(features))
```

**复杂度**: 中 — 需调整 num_outs=5 (P3-P7), 训练耗时增加 ~30%。

**预期收益**: mAP +0.010~0.025 (双向特征流 + 多阶段增强)。

### G3: 跨层注意力 (CrossLayerAttention)

**问题**: FPN 各层级独立处理, 无法直接融合跨层级语义信息 (如 P2 的细节 + P5 的语义)。

**方案**: 在 LAMFPN 末尾加 CrossLayerAttention, 让每个层级融合其他所有层级的加权特征。

**实现**:
```python
class CrossLayerAttention(BaseModule):
    def __init__(self, channels, num_levels, ...):
        self.shared_transform = ConvModule(channels, channels, 1, ...)  # 共享变换
        self.shared_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, num_levels - 1, 1),  # 生成其他层级的权重
            nn.Sigmoid()
        )
        self.fuse_conv = ConvModule(channels, channels, 3, ...)

    def forward(self, features):
        # 对每个层级, 加权融合其他所有层级 (resize 到同尺寸)
        for level_idx, feat in enumerate(features):
            weights = self.shared_weight_gen(feat)  # (B, num_levels-1, 1, 1)
            fused = sum(other_feat * weights[:, i:i+1] for i, other_feat in enumerate(other_feats))
            output = self.fuse_conv(feat + fused)  # 残差
```

**复杂度**: 低 (可附加在 G1/G2 之上, `use_cross_layer_attention=True`)。

**预期收益**: mAP +0.003~0.008 (跨层语义补充, 边际收益递减)。

---

## 3. LAMModule 核心机制详解

### 3.1 结构

```
输入: target_feat (B, C, H, W), source_feat (B, C, H, W)
  │
  ├─ concat → (B, 2C, H, W)
  │
  ├─ attention_conv (DepthwiseSeparableConv 3x3, 通道压缩 C→C/4)
  │
  ├─ attention_weights (Conv 1x1 → 2 通道)
  │
  ├─ softmax(dim=1) → (B, 2, H, W)  [空间位置的 2 维权重]
  │
  ├─ fused = w0 * target + w1 * source  [加权融合]
  │
  └─ fusion_conv (DepthwiseSeparableConv 3x3) → 输出 (B, C, H, W)
```

### 3.2 与 BiFPN FastNormalizedFusion 的对比

| 特性 | BiFPN | LAMModule |
|------|-------|-----------|
| 融合方式 | `w_i * F_i / Σw_j` (标量权重) | `softmax(w0, w1)` 逐空间位置权重 |
| 权重粒度 | 每个特征图一个权重 | 每个空间位置一组权重 |
| 输入特征 | 3 路 (top-down + bottom-up) | 2 路 (target + source) |
| 非线性 | 无 (仅归一化) | 有 (conv + norm + act) |

**LAMModule 优势**: 逐空间位置的注意力权重, 能针对不同区域 (如染色体主体 vs 背景) 调整融合策略。

### 3.3 计算开销

- LAMModule 参数量: ~2C²/4 (attention_conv) + 2C (weights) + C² (fusion_conv) ≈ 1.5C²
- 对 C=256: ~100K 参数/模块, 2 个模块共 ~200K (相比 FPN 总量 ~5M, 增加 4%)
- 训练耗时增加: ~10-15%

---

## 4. DualAttention 机制详解

### 4.1 通道注意力 (Channel Attention)

```
avg_pool = AdaptiveAvgPool2d(x, 1)  → (B, C, 1, 1)
max_pool = AdaptiveMaxPool2d(x, 1)  → (B, C, 1, 1)
         ↓
shared_mlp: Conv1x1(C→C/16) → SiLU → Conv1x1(C/16→C)
         ↓
channel_att = Sigmoid(avg_out + max_out)
x = x * channel_att
```

**特点**: 共享 MLP 处理 avg+max pool, 减少参数, 同时捕获全局统计的不同方面。

### 4.2 空间注意力 (Spatial Attention)

```
spatial_avg = mean(x, dim=1)  → (B, 1, H, W)
spatial_max = max(x, dim=1)   → (B, 1, H, W)
         ↓ concat → (B, 2, H, W)
         ↓
spatial_att = Sigmoid(Conv7x7)
x = x * spatial_att
```

**特点**: 7x7 大卷积核捕获空间邻域关系, 适合染色体这种有空间结构的目标。

### 4.3 跳过机制 (Skip Attention)

```python
with torch.no_grad():
    importance = torch.mean(F.adaptive_avg_pool2d(lateral, 1))
if importance > self.skip_attention_thresh:  # 默认 0.05
    # 应用注意力
else:
    # 跳过, 直接用原特征
```

**作用**: 对低能量特征图跳过注意力, 节省计算, 避免过度增强噪声。

---

## 5. 整合方案

### 5.1 文件组织

```
ldmdet/
├── necks/
│   ├── __init__.py              # 导出 LAMFPN, LAMFPNBiFPN
│   ├── lam_fpn.py               # G1: 单阶段 LAMFPN (基于 lamfpn.py)
│   └── lam_fpn_bifpn.py         # G2: 多阶段 BiFPN 风格 (基于 lamfpn_bifpn.txt)
experiments/
├── mmdet_bridge/
│   └── registry.py              # 注册 LAMFPN (或直接在 necks/__init__.py 注册)
├── configs/ldmdet/
│   ├── direction_g_lamfpn.py            # G1 配置
│   ├── direction_g_lamfpn_bifpn.py       # G2 配置
│   └── direction_g_lamfpn_cross.py      # G3 配置 (G1 + cross attention)
└── tests/
    └── test_direction_g.py               # 单元测试
```

### 5.2 配置示例 (G1 单阶段)

```python
_base_ = ['./rf_heun_adaln.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='LAMFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        # G1 关键参数
        apply_lam_levels=[1, 2],          # 在 P3, P4 应用 LAM
        attention_type='dual',            # 启用 DualAttention
        use_cross_layer_attention=False,  # G3 独立实验
        use_dw_conv=True,                 # 深度可分离卷积减少参数
        channel_reduction=4,              # LAM 通道压缩比
        use_modern_norm=True,             # GroupNorm
        use_modern_act='silu',            # SiLU 激活
        use_checkpoint=True,              # 梯度检查点省显存
    ),
)
```

### 5.3 配置示例 (G2 多阶段 BiFPN)

```python
_base_ = ['./rf_heun_adaln.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='LAMFPNBiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_stages=3,                     # 3 次双向融合
        num_outs=5,                        # P3-P7
        apply_lam_levels=[1, 2],
        attention_type='dual',
        use_cross_layer_attention=False,
    ),
)
```

### 5.4 与现有方向的兼容性

- **方向 A (P1+Deformable)**: 可组合, 但 P1 是 FPN 输出层, LAMFPN 是 FPN 内部, 需在 LAMFPN 末尾接 P1 逻辑 (或二选一)
- **方向 B (DecoupledHead)**: 完全兼容, 互不影响
- **方向 C (ShapeAttention)**: 完全兼容, ShapeAttention 在 RoI 特征上, LAMFPN 在 FPN 特征上
- **方向 D (BoxRefine)**: 完全兼容
- **方向 F (StructuredPrior)**: 完全兼容

**推荐组合**: G1 (轻量 LAMFPN) + B (DecoupledHead) + D (BoxRefine) — 三者分别改进特征、分类、定位, 互不冲突。

---

## 6. 测试计划

### 6.1 单元测试 (红色阶段)

```python
# ldmdet/tests/test_direction_g.py

class TestLAMModule:
    def test_output_shape(self):
        """LAMModule 输出 shape 应与输入一致"""
    def test_attention_weights_sum_to_one(self):
        """softmax 后权重和为 1"""
    def test_gradient_flow(self):
        """梯度应能通过 attention_conv 和 fusion_conv 回传"""

class TestDualAttention:
    def test_channel_attention_range(self):
        """通道注意力输出在 [0, 1]"""
    def test_spatial_attention_shape(self):
        """空间注意力输出 (B, 1, H, W)"""
    def test_skip_mechanism(self):
        """低能量特征应跳过注意力"""

class TestLAMFPN:
    def test_forward_shape(self):
        """前向输出 num_outs 个特征图"""
    def test_backward_compatible_with_fpn(self):
        """应能替换标准 FPN, 接口兼容"""
    def test_lam_modules_at_correct_levels(self):
        """LAM 应只在 apply_lam_levels 指定层级应用"""

class TestCrossLayerAttention:
    def test_all_levels_fused(self):
        """每个输出层级都融合了其他所有层级"""
    def test_weight_sum_to_one(self):
        """跨层权重和接近 1 (sigmoid)"""
```

### 6.2 集成测试

```python
def test_direction_g_integration():
    """端到端 forward+backward"""
    cfg = Config.fromfile('experiments/configs/ldmdet/direction_g_lamfpn.py')
    model = MODELS.build(cfg.model)
    # 模拟输入
    batch = {'inputs': [torch.randn(3, 256, 256)], 'data_samples': [...]}
    losses = model.loss(**batch)
    total = sum(losses.values())
    total.backward()
    # 验证 LAM 模块有梯度
    assert model.bbox_head.neck.lam_modules[0].attention_conv.weight.grad is not None
```

---

## 7. 实验计划

### 7.1 主实验

| 实验名 | 子方向 | 配置 | 预期 |
|--------|--------|------|------|
| direction_g_lamfpn | G1 | 单阶段 LAMFPN | mAP +0.005~0.015 |
| direction_g_lamfpn_bifpn | G2 | 多阶段 BiFPN 风格 | mAP +0.010~0.025 |
| direction_g_lamfpn_cross | G3 | G1 + CrossLayerAttention | mAP +0.008~0.020 |

### 7.2 消融实验

| 消融项 | 配置变化 | 验证目标 |
|--------|---------|---------|
| w/o LAM | `apply_lam_levels=[]` | LAM 融合 vs 简单相加的贡献 |
| w/o DualAttention | `attention_type=None` | 通道+空间注意力的贡献 |
| w/o CrossLayerAttention | `use_cross_layer_attention=False` | 跨层注意力的贡献 |
| LAM levels = [0,1,2,3] | 全层级应用 | 是否过度增强 |
| num_stages = 1,2,3,4 | 多阶段数量 | BiFPN 阶段数收益 |

### 7.3 训练命令

```bash
# G1 单阶段 (推荐先做, 轻量)
python experiments/runners/train.py experiments/configs/ldmdet/direction_g_lamfpn.py \
    --work-dir work_dirs/direction_g_lamfpn --seed 42 --gpu-id 0

# G2 多阶段 BiFPN
python experiments/runners/train.py experiments/configs/ldmdet/direction_g_lamfpn_bifpn.py \
    --work-dir work_dirs/direction_g_lamfpn_bifpn --seed 42 --gpu-id 0

# G3 G1 + CrossLayerAttention
python experiments/runners/train.py experiments/configs/ldmdet/direction_g_lamfpn_cross.py \
    --work-dir work_dirs/direction_g_lamfpn_cross --seed 42 --gpu-id 0
```

---

## 8. 风险与缓解

### 8.1 显存风险

**风险**: G2 多阶段 + CrossLayerAttention 显存占用增加 ~30-40%。

**缓解**:
- 启用 `use_checkpoint=True` (梯度检查点, 以计算换显存)
- 降低 batch_size (从 8 降到 4)
- 先做 G1 (轻量版) 验证收益, 再决定是否做 G2

### 8.2 过拟合风险

**风险**: 注意力模块参数增加, 在小数据集 (4680 张) 上可能过拟合。

**缓解**:
- 使用深度可分离卷积 (`use_dw_conv=True`) 减少参数
- 通道压缩 (`channel_reduction=4`)
- 跳过机制 (`skip_attention_thresh=0.05`) 避免过度增强噪声
- 监控 train/val loss 差距, 必要时增加 dropout

### 8.3 兼容性风险

**风险**: LAMFPN 替换 FPN 后, RoIExtractor 的 `featmap_strides` 可能不匹配。

**缓解**:
- G1 保持 `num_outs=4` (与 baseline 一致), 无需改 RoIExtractor
- G2 改 `num_outs=5` 时, 同步更新 RoIExtractor 的 `featmap_strides=[4, 8, 16, 32, 64]`

---

## 9. 成功指标

| 指标 | baseline | 目标 | 验证方式 |
|------|---------|------|---------|
| mAP | 0.8583 | +0.010 | COCO eval |
| mAP_75 | 0.9677 | +0.005 | COCO eval |
| mAP_90 | 0.4834 | +0.020 | 高 IoU 精度提升 |
| 训练耗时 | 1.0x | ≤1.3x | wall time |
| 显存 | 12GB | ≤18GB | nvidia-smi |

---

## 10. 实现路线图

1. **Phase 1: G1 单阶段 LAMFPN** (优先)
   - 从 [lamfpn.py](../../lamfpn.py) 迁移到 `ldmdet/necks/lam_fpn.py`
   - 红绿测试: LAMModule, DualAttention, LAMFPN
   - 集成测试: forward+backward
   - 配置: `direction_g_lamfpn.py`
   - 训练验证

2. **Phase 2: G3 CrossLayerAttention** (G1 收益确认后)
   - 启用 `use_cross_layer_attention=True`
   - 测试跨层注意力独立贡献
   - 配置: `direction_g_lamfpn_cross.py`

3. **Phase 3: G2 多阶段 BiFPN** (G1 收益显著时)
   - 从 [lamfpn_bifpn.txt](../../lamfpn_bifpn.txt) 迁移到 `ldmdet/necks/lam_fpn_bifpn.py`
   - 调整 num_outs=5, 同步 RoIExtractor
   - 配置: `direction_g_lamfpn_bifpn.py`

---

## 11. 参考文献

- FPN: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
- BiFPN: Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020
- CBAM (DualAttention 灵感): Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- LAM (Local Attention Module): 项目内部实现, 参考 [lamfpn.py](../../lamfpn.py)
