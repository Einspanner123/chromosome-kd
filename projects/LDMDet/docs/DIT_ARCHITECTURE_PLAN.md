# LDMDet-DiT 架构重构方案

> 从 RoIAlign + DynamicConv 到 Deformable Cross-Attention DiT 的完整重构计划
> 创建日期：2026-06-01
> 状态：实施中

______________________________________________________________________

## 一、重构动机

### 1.1 当前架构瓶颈

当前 LDMDet 的数据流：

```
Image → ResNet-50 → FPN → [RoIAlign → Self-Attn → DynamicConv → FFN] × 6 heads → cls/reg
```

| 瓶颈 | 说明 |
|------|------|
| RoIAlign 硬裁剪 | 框只决定 RoI 位置，不参与特征计算；裁剪操作不可微，破坏梯度流 |
| DynamicConv 单向局部 | proposal→image 单向交互，远不如 Cross-Attention 全局双向 |
| Self-Attn 缺图像特征 | 500 个 proposal 之间 Self-Attn 缺乏图像特征直接注入 |
| 框-图像间接耦合 | 框的位置信息必须经 RoIAlign 中转才能影响特征 |

### 1.2 DiT 重构目标

```
Image → ResNet-50 → FPN → [Self-Attn + Deformable Cross-Attn + FFN] × 6 blocks → cls/reg
                                    ↑
                          Box Tokens 直接从 FPN 采样特征
```

**核心改进**：用 Deformable Cross-Attention 替代 RoIAlign + DynamicConv，实现框-图像的双向全局交互。

______________________________________________________________________

## 二、设计决策记录

> 每个决策列出所有可选项，标注当前选择 ✅ 和备选方案

### 2.1 重构深度

| 选项 | 说明 | 状态 |
|------|------|------|
| 渐进式：保留 backbone，只重构检测头 | 风险最低，1-2 周出结果 | 备选 |
| **✅ 激进式：Unified DiT** | 彻底消除 RoIAlign，端到端 DiT | **当前选择** |
| 两步走：先渐进后激进 | 稳妥但耗时更长 | 备选 |

### 2.2 Box-Image 交互机制

| 选项 | 说明 | 状态 |
|------|------|------|
| Cross-Attention | Box 作为 Query，Image 作为 KV，DETR 系列经典方案 | 首选 |
| 全 Self-Attention | Box + Image 拼接做全注意力，交互更充分但 O((N_img+N_box)²) | 备选 |
| **✅ 先 Cross-Attn 后对比 Full Self-Attn** | 先验证 Cross-Attn 有效，再探索 Full Self-Attn | **当前选择** |

### 2.3 Image Token 来源

| 选项 | 说明 | 状态 |
|------|------|------|
| **✅ FPN 多尺度特征 (P2-P5)** | 多尺度对检测至关重要，可自然处理不同大小物体 | **当前选择** |
| 单尺度特征 | 简单但损失多尺度信息 | 不推荐 |
| FPN + Pixel Shuffle 降采样 | 减少 token 数量，降低注意力计算量 | 备选 |

### 2.4 Token 压缩策略

| 选项 | 计算量 | 说明 | 状态 |
|------|--------|------|------|
| Pixel Shuffle 4× 降采样 | O(N_box × N_img/16) | 简单高效，保留多尺度结构 | 备选 |
| **✅ Deformable Cross-Attention** | O(N_box × K × L) | 每个 box 采样 K 点，计算量最低 | **当前选择** |
| 丢弃 P2 + 轻度降采样 | O(N_box × N_img/4) | 仍偏多 | 不推荐 |
| 通道压缩 (256→64) | O(N_box × N_img/4) | 信息密度下降 | 不推荐 |

**Deformable Cross-Attention 计算量**：
- K=8 采样点，4 个 FPN 层级
- 每层注意力：500 × 8 × 4 = 16K（vs Cross-Attention 500 × 88750 ≈ 44M）

### 2.5 Backbone 选择

| 选项 | 说明 | 状态 |
|------|------|------|
| **✅ 先 ResNet-50 后 ViT** | 先用 ResNet 验证 DiT 检测头，确认有效后换 ViT | **当前选择** |
| ResNet-50 + FPN | 可复用预训练权重，风险最低 | Phase 1 |
| ViT-Base + 简化 FPN | 特征更丰富但需重新训练，小数据集可能过拟合 | Phase 2 |

### 2.6 Box Token 初始化

| 选项 | 说明 | 状态 |
|------|------|------|
| **✅ 先双线性插值后对比可学习 Query** | 双线性插值保留空间锚定，与 RF 流程兼容 | **当前选择** |
| 双线性插值采样 | 用 bbox 中心在 FPN 上采样，保留空间锚定 | Phase 1 |
| 可学习 Query + 坐标编码 | 更简单但缺乏空间锚定 | 消融对比 |

### 2.7 迭代去噪策略

| 选项 | 说明 | 状态 |
|------|------|------|
| **✅ 先 6 Block 后探索单深 DiT** | 6 Block 保留 deep supervision，与现有管线兼容 | **当前选择** |
| 6 Block 串行 + Deep Supervision | 每个 block 输出预测，训练稳定 | Phase 1 |
| 单深 DiT (12-24 层) | 更简洁但失去 deep supervision | Phase 4 |

### 2.8 Deformable Attention 采样点数 K

| 选项 | 计算量 | 覆盖率 | 状态 |
|------|--------|--------|------|
| K=4 | 最低 | 可能对大物体不足 | 消融 |
| **✅ K=8** | 平衡 | Deformable DETR 默认 | **当前选择** |
| K=16 | 较高 | 覆盖最充分 | 消融 |

> 后续做 K 值消融实验验证最优值

### 2.9 Deformable Attention 实现方式

| 选项 | 说明 | 状态 |
|------|------|------|
| mmcv.ops.MultiScaleDeformableAttention | CUDA 优化，与 MMDet 兼容 | 备选 |
| **✅ 纯 PyTorch 实现** | 不依赖 mmcv，完全可控 | **当前选择** |

### 2.10 实施节奏

| 选项 | 说明 | 状态 |
|------|------|------|
| MVP 先行 | 先跑通最小版本再逐步添加 | 备选 |
| **✅ 完整实现一步到位** | 一步到位实现完整架构 | **当前选择** |

______________________________________________________________________

## 三、AdaLN-Zero 参数设计

### 3.1 参数数量选择

| 选项 | 结构 | 参数量/block | 状态 |
|------|------|-------------|------|
| 6 参数 (2×3) | Self-Attn+Cross-Attn 共享一组, FFN 一组 | 6×C | 备选 |
| **✅ 9 参数 (3×3)** | Self-Attn, Cross-Attn, FFN 各一组 | 9×C | **当前选择** |
| 7 参数 (混合) | 共享 γ/β, 独立 α | 7×C | 备选 |

**9 参数的理论依据**：

Self-Attn 和 Cross-Attn 在不同时间步需要差异化激活：

| 时间步 | Self-Attn 作用 | Cross-Attn 作用 | 理想 α |
|--------|---------------|----------------|--------|
| t≈1（噪声框）| 低（框间关系无意义）| 高（需从图像拉取特征）| α1≈0, α2≈大 |
| t≈0.5（半精化）| 中（框间关系开始有意义）| 高（继续精化）| α1≈中, α2≈中 |
| t≈0（近精化）| 高（框间协调、抑制重叠）| 低（位置已确定）| α1≈大, α2≈小 |

**9 参数的实践优势**：
- 参数量可忽略（9×256×6 blocks = 13,824，占模型 < 0.1%）
- 零初始化保证安全性
- 可视化 α 随 t 变化，验证差异化激活假说
- 消融实验成本低（9→6 只需改一行代码）

### 3.2 采样偏移的时间调制

| 选项 | 说明 | 状态 |
|------|------|------|
| **✅ 隐式调制（通过 AdaLN-Zero）** | query 经 AdaLN 调制后预测偏移，间接时间调制 | **当前选择** |
| 显式调制（额外 time_offset_proj） | 时间嵌入直接注入偏移预测 | 消融 P2 |

**隐式调制的理论依据**：
- AdaLN-Zero 零初始化 → 初始偏移为零（α2=0 → Cross-Attn 被门控为零）
- 训练过程中网络自然学习时间相关的偏移模式
- 显式调制增加实现复杂度，作为后续改进点

______________________________________________________________________

## 四、架构详细设计

### 4.1 整体数据流

```
Image (3, H, W)
    │
    ▼
ResNet-50 → FPN (P2-P5, 各 256ch)
    │
    │    噪声框 x_t (bs, 500, 4)
    │       │
    │       ▼
    │    BoxTokenizer → Box Tokens (bs, 500, 256)
    │       │  = bilinear_sample(FPN, bbox_center) + bbox_pos_embed + level_embed
    │       │
    │       ▼
    │  ┌──────────────────────────────────────┐
    │  │      DiT Block #1 (AdaLN-Zero)       │
    │  │                                       │
    │  │  1. Self-Attention (box↔box)         │── cls/reg head #1 (deep supervision)
    │  │  2. Deformable Cross-Attention       │
    │  │     (box→FPN, K=8, 4 levels)         │
    │  │  3. FFN                              │
    │  └──────────────────────────────────────┘
    │       │
    │       ▼
    │  ┌──────────────────────────────────────┐
    │  │      DiT Block #2 (AdaLN-Zero)       │── cls/reg head #2
    │  │      ...                              │
    │  └──────────────────────────────────────┘
    │       │
    │      ... (共 6 blocks)
    │       │
    │  ┌──────────────────────────────────────┐
    │  │      DiT Block #6 (AdaLN-Zero)       │── cls/reg head #6
    │  └──────────────────────────────────────┘
    │       │
    │       ▼
    │    最终预测: class_logits + pred_bboxes
    │
    │  时间 t → SinusoidalEmbed → MLP → AdaLN-Zero params
    │           (γ1,β1,α1, γ2,β2,α2, γ3,β3,α3 per block)
```

### 4.2 模块清单

| 模块 | 文件 | 职责 |
|------|------|------|
| MultiScaleDeformableAttention | `mods/deformable_attn.py` | 纯 PyTorch 可变形注意力 |
| BoxTokenizer | `mods/box_tokenizer.py` | bbox → Box Token 初始化 |
| DiTBlock | `mods/dit_block.py` | Self-Attn + Deformable Cross-Attn + FFN + AdaLN-Zero |
| DiTSingleHead | `mods/dit_single_head.py` | 单 Block + 预测头 |
| DiTDiffusionDetHead | `mods/dit_head.py` | 完整检测头（RF + OT + 损失） |

### 4.3 与现有系统的兼容性

| 组件 | 当前 | DiT 版本 | 兼容 |
|------|------|---------|------|
| RF 前向加噪 | `x_t = (1-t)x_0 + t·x_1` | 不变 | ✅ |
| Shifted Schedule | `t → st/(1+(s-1)t)` | 不变 | ✅ |
| Stochastic OT (ε=5) | Sinkhorn + multinomial | 不变 | ✅ |
| 检测损失 | Focal + L1 + GIoU | 不变 | ✅ |
| AdaLN-Zero | 单 head 内 | 扩展到 DiT Block | ✅ |
| Deep Supervision | 6 heads 各自输出 | 6 blocks 各自输出 | ✅ |
| Heun/DPM-Solver++ | 推理时迭代采样 | 不变 | ✅ |

______________________________________________________________________

## 五、实施步骤

| 步骤 | 内容 | 状态 |
|------|------|------|
| Step 1 | 实现 `deformable_attn.py` | ✅ 已完成 |
| Step 2 | 实现 `box_tokenizer.py` | ✅ 已完成 |
| Step 3 | 实现 `dit_block.py` | ✅ 已完成 |
| Step 4 | 实现 `dit_single_head.py` | ✅ 已完成 |
| Step 5 | 实现 `dit_head.py` | ✅ 已完成 |
| Step 6 | 编写配置文件 `ldmdet_dit.py` | ✅ 已完成 |
| Step 7 | 注册组件到 `model.py` | ✅ 已完成 |
| Step 8 | 单元测试：各模块 shape 验证 | ✅ 已完成 |
| Step 8.5 | 端到端 wrapper 测试 (backbone+FPN+DiT head) | ✅ 已完成 |
| Step 9 | 端到端训练测试：1 epoch 跑通 | ⬜ 待开始 |
| Step 10 | 完整训练 + 与 SOTA 对比 | ⬜ |

### 训练命令

```bash
# 单 GPU 训练
conda activate chromo
python tools/train.py projects/LDMDet/configs/ldmdet_dit.py --work-dir work_dirs/ldmdet_dit

# 多 GPU 训练 (2 GPU)
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py projects/LDMDet/configs/ldmdet_dit.py --work-dir work_dirs/ldmdet_dit --launcher pytorch
```

______________________________________________________________________

## 六、风险与缓解

| 风险 | 概率 | 缓解策略 |
|------|------|---------|
| Deformable Attn 纯 PyTorch 性能差 | 中 | 先用 F.grid_sample 实现，不够再考虑 CUDA kernel |
| 双线性插值初始化信息不足 | 中 | 备选：可学习 Query + 坐标编码 |
| 6 Block DiT 训练不稳定 | 低 | deep supervision + AdaLN-Zero 零初始化 |
| Deformable Cross-Attn 偏移不收敛 | 低 | 偏移零初始化 + 小学习率 |
| 整体 mAP 不如现有 SOTA | 中 | Phase 1 目标持平（≥0.753），不急于超越 |

______________________________________________________________________

## 七、后续探索路线

```
Phase 1 (当前)
  ResNet-50 + FPN + Deformable Cross-Attn DiT
  目标: mAP ≥ 0.753

    ↓ 验证成功后

Phase 2
  ViT-Base backbone + 简化 FPN
  目标: mAP > 0.76

    ↓ 验证成功后

Phase 3
  Full Self-Attention (Box + Image Tokens 拼接)
  目标: 验证全局注意力是否有额外收益

    ↓ 验证成功后

Phase 4
  单深 DiT (12-24 层, 无 deep supervision)
  目标: 更简洁的架构 + 更好的 scaling
```

______________________________________________________________________

## 八、变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-01 | 初始方案创建，确认所有核心设计决策 |
| 2026-06-02 | Step 1-8 全部完成，单元测试通过 |
| 2026-06-02 | 修复 Bug: deformable_attn bias shape (256,2)→(512,)；box_tokenizer grid_sample batch 维度不匹配；_assign_fpn_level 负面积导致 NaN；dit_single_head 3D→2D reshape |
| 2026-06-02 | 端到端 wrapper 测试通过：ResNet-18+FPN+DiT Head loss/backward/predict 全部正常 |
