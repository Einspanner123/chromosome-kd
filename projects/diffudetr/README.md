# DiffuDETR 复现项目

> DiffuDETR (ICLR 2026) 核心逻辑移植到 mmdet 框架

## 复现状态

已完成核心模块移植和验证，模型可构建、训练前向、推理前向均通过测试。

## 项目结构

```
projects/diffudetr/
├── __init__.py                      # 注册 DiffuDETRDetector
├── configs/
│   ├── __init__.py
│   └── diffudetr_24obj.py           # Dataset2 染色体配置 (24类)
├── models/
│   ├── __init__.py
│   ├── diffusion_scheduler.py       # cosine schedule + q_sample + DDIM
│   ├── timestep_block.py            # FiLM TimeStepBlock (时间步注入)
│   ├── transformer.py               # 简化版 DINO decoder (标准 attention)
│   ├── diffudetr_head.py            # 扩散头: 训练 + 推理 (DDIM + box_renewal)
│   ├── diffudetr_detector.py        # mmdet BaseDetector 桥接
│   └── criterion.py                 # SNR 加权损失 + Hungarian matching
└── README.md
```

## 原仓库参考

- 原仓库: https://github.com/MBadran2000/DiffuDETR
- 移植的核心文件:
  - `projects/diffu_dino/modeling/dino_diffu_det_noise.py` → `diffusion_scheduler.py` (调度器)
  - `projects/diffu_dino/modeling/bbox_embedd.py` → `timestep_block.py` (TimeStepBlock)
  - `projects/diffu_dino/modeling/dino_transformer.py` → `transformer.py` (Decoder)
  - `layers_diffu_detr/transformer.py` → `transformer.py` (BaseTransformerLayer 逻辑)
  - `layers_diffu_detr/denoising.py` → 参考噪声生成逻辑

## 关键设计决策

### 1. 扩散调度 (diffusion_scheduler.py)
- **cosine_beta_schedule** (1000 timesteps, s=0.008), 对齐原仓库 `register_schedule_old`
- **parameterization = "x0"**: 模型直接预测干净框 x0 (非预测 noise)
- **q_sample**: `x_t = sqrt(acp_t) * x_0 + sqrt(1-acp_t) * noise`
- **DDIM 采样**: 25 步, eta=0 (确定性)
- **scale = 2**: box [0,1] → [-2,2] 扩散空间, 匹配 N(0,1) 噪声尺度
- **SNR loss_weight**: `0.5 * sqrt(acp) / (2 - acp)`, 高 timestep 权重高

### 2. 时间步注入 (timestep_block.py)
- **FiLM 调制** (Feature-wise Linear Modulation):
  - `emb_layers`: SiLU → Linear(emb_ch=1024 → 2*out_ch=512)
  - 拆分为 (scale, shift): `h = x * (1 + scale) + shift`
  - 残差: `return x + h`
- 输入: time_embed (4*embed_dim=1024), 输出: out_channels (256)

### 3. 简化版 Transformer (transformer.py)
- 用标准 `nn.MultiheadAttention` 替代 `MultiScaleDeformableAttention` (当前环境无 MSDeformAttn)
- 去除 two-stage proposal, 改用 learned query embeddings (`tgt_embed`)
- 每层注入 timestep: `DiffuDETRDecoderLayer` 在 self_attn/cross_attn/ffn 前各加一个 `TimeStepBlock`
- 参数: num_layers=6, embed_dim=256, num_heads=8, dim_feedforward=2048
- `time_embed`: Sequential(Linear(256→1024), SiLU, Linear(1024→1024))

### 4. 扩散头 (diffudetr_head.py)
- **训练**: GT box → pad 到 num_queries (随机框+背景标签) → 固定随机 shuffle → q_sample(t) → decoder 预测 x0 → SNR 加权损失
- **推理**: 纯噪声 randn → DDIM 25步 → box_renewal (自适应阈值) → ensemble + NMS
- **box_renewal**: `threshold = time_next / 100`, 低分框替换为 randn
  - 高 timestep: 阈值大 → 几乎全部 renew (预测不可靠)
  - 低 timestep: 阈值小 → 保留可靠预测
- **num_classes=24, num_queries=300, 背景标签 = num_classes (24)**
- 每层 class_embed 前注入 TimeStepBlock (对齐原仓库 ClassEmbed)

### 5. mmdet 桥接 (diffudetr_detector.py)
- 参考 `experiments/mmdet_bridge/setdiff_detector.py` 桥接模式
- backbone/neck 通过 mmdet `MODELS.build` 构建
- bbox_head 通过纯 PyTorch 构建 (`DiffuDETRHead`)
- GT 转换: xyxy 像素 → cxcywh 归一化 → [-scale, scale] 扩散空间
- 预测转换: [-scale, scale] → [0,1] → cxcywh 像素 → xyxy 像素

### 6. 损失函数 (criterion.py)
- **HungarianMatcher**: cost = cost_class + cost_bbox + cost_giou
- **SetCriterion**: sigmoid focal loss (分类) + L1 + GIoU (回归)
- **SNR 加权**: 回归损失 × loss_weight[t]
- **辅助损失**: 每个 decoder 中间层计算损失 (deep supervision)

## 配置 (configs/diffudetr_24obj.py)
- 数据集: `data/24_chromosomes_object/coco/`
- 24类: A1-A3, B4-B5, C6-C12, D13-D15, E16-E18, F19-F20, G21-G22, X, Y
- backbone: ResNet-50 + FPN (mmdet 标准)
- 训练: AdamW lr=5e-5, 50 epochs, batch_size=2
- val_evaluator: classwise=True (输出 24 per-class AP)
- SwanLab: project='diffudetr-24obj'

## 使用方式

### 训练
```bash
cd /home/linkst/workspace/chromosome-kd
python tools/train.py projects/diffudetr/configs/diffudetr_24obj.py
```

### 配置引用
```python
custom_imports = dict(
    imports=['projects.diffudetr'],
    allow_failed_imports=False,
)
model = dict(type='DiffuDETR', ...)
```

## 验证结果

以下测试均已通过 (chromo conda 环境: PyTorch 2.1.0 + cu118 + mmdet 3.3.0):
1. `projects.diffudetr` 包导入成功
2. `DiffusionScheduler`: cosine schedule, q_sample, DDIM step, 25 time_pairs
3. `TimeStepBlock`: FiLM 调制, 输出 shape 正确
4. `DiffuDETRTransformer`: 6 层 decoder, 输出 [6, B, 300, 256]
5. `DiffuDETRHead` 训练前向: loss 计算 + 反向传播成功
6. `DiffuDETRHead` 推理前向: DDIM 25步 + box_renewal + ensemble + NMS
7. `DiffuDETRDetector` loss(): mmdet DetDataSample 格式兼容
8. `DiffuDETRDetector` predict(): 输出 pred_instances
9. 配置文件加载: 所有参数正确
10. 从配置构建模型: 51.3M 参数

## 已知限制

1. **标准 attention 替代 deformable attention**: 当前环境无 `MultiScaleDeformableAttention`,
   用标准 `nn.MultiheadAttention` 替代。这会导致:
   - 训练收敛速度可能慢于原版 (deformable attention 的多尺度采样效率更高)
   - 显存占用可能更高 (标准 attention 对 HW 的复杂度是 O(N*HW))
   - mAP 可能低于原版报告值

2. **无 two-stage proposal**: 原仓库使用 two-stage (encoder 输出 proposal → decoder 精炼),
   简化版改用 learned query embeddings, 可能影响小目标检测性能

3. **无 DN-DETR 去噪训练**: 原仓库支持 CDN (Contrastive Denoising) 查询,
   简化版未实现 (denoising.py 的 GenerateCDNQueries 逻辑未移植)

4. **box_renewal 阈值**: `threshold = time_next/100` 在高 timestep 时阈值 > 1,
   导致几乎所有框被 renew (替换为 randn)。这是原仓库的设计, 但可能需要调参

5. **ensemble 策略**: 当前累积所有 DDIM 步的预测后做 NMS,
   与原仓库的 ensemble 方式可能略有差异

6. **未实际训练验证 mAP**: 模型可构建和前向, 但未进行完整训练验证最终 mAP
