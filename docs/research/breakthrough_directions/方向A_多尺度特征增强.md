# 方向 A：多尺度特征增强 (Multi-Scale Feature Enhancement)

> **目标**：突破大小目标 trade-off 瓶颈 (scale_aware_loss: mAPs +0.020 但 mAPl -0.081)，从特征金字塔层面提升小目标分辨率而不损害大目标。
>
> **瓶颈证据**：
> - 诊断显示 mAPs=0.510 远低于 mAPl=0.672，小目标定位是主要瓶颈
> - 2a 实验证实损失调参无法解决 (大小目标 trade-off 比例 4:1 ~ 6:1)
> - FPN 的 4 层输出 (stride 4/8/16/32) 对小染色体 (~32px) 分辨率不足
>
> **当前代码位置**：[experiments/configs/ldmdet/rf_heun_adaln.py](../../experiments/configs/ldmdet/rf_heun_adaln.py) 中 `neck=dict(type='FPN', num_outs=4)`

---

## 1. 子方向说明

### A1: 增加 P2 层 (高分辨率特征)

**问题**：当前 FPN 使用 stride 4/8/16/32 (P2-P5)，最小 stride=4 对 ~32px 小染色体 (如 G 组) 在 P2 上的有效像素仅 8x8，RoIAlign 后信息不足。

**方案**：将 backbone 的 `out_indices=(0,1,2,3)` 改为 `(0,1,2,3)` 保留，FPN 增加 P2 输出，使 num_outs=5 (P2-P6)，让小目标在 stride=2 的高分辨率特征上检测。

**实现**：
```python
# backbone: ResNet-50 已有 4 个 stage 输出 (stride 4/8/16/32)
# FPN 配置改为:
neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,  # P2-P6 (新增 P2 = stride 2 — 但 ResNet stage0 已是 stride 4)
    # 注意: 标准 FPN 从 backbone stage0 开始, stride 4
    # 要获得 stride 2, 需修改 backbone 或加额外 conv
)
```

**复杂度**：中 — 需确认 RoIExtractor 的 featmap_strides 是否覆盖新层。

### A2: 可变形 RoIAlign (DeformableRoIAlign)

**问题**：标准 RoIAlign 在固定网格采样，对小框 (G 组 ~32px) 的几何结构 (着丝粒位置、臂长比例) 提取不充分。

**方案**：将 [roi_extractor.py](../../ldmdet/core/roi_extractor.py) 中 `RoIAlign` 替换为 `DeformableRoIAlign`，让采样点自适应染色体形态 (弯曲、非刚性)。

**实现**：mmdet 提供 `DeformableRoIAlign`，只需修改配置：
```python
roi_extractor=dict(
    roi_layer=dict(type='DeformableRoIAlign', output_size=7, sampling_ratio=2),
    out_channels=256,
    featmap_strides=[4, 8, 16, 32],
)
```

**复杂度**：低 — 仅替换 roi_layer 类型，但需安装 deformable operator。

### A3: BiFPN 加权特征融合 (已有历史实验)

**查找结果**：在旧项目 `ldmdet-experiment/` 中已实现 BiFPN 替代品 [LAMFPN](../../lamfpn_bifpn.txt)，并嵌入多个 SOTA 配置 (phase5-9)。

**旧项目实验数据** (phase7_small_obj/smallobj_B_scaleaware_loglinear):
- best mAP = 0.744 (epoch 71), APs=0.497, APl=0.594
- 该实验同时使用了 scale-aware loss + BiFPN + OT 耦合，无法单独评估 BiPFN 贡献
- 旧项目 baseline (无 BiFPN) mAP ~0.726，最佳 SOTA (含 BiFPN) mAP ~0.744

**结论**：BiFPN 在旧项目中被使用但未单独消融。当前 LDMDet (重构后) 用标准 FPN 达到 mAP=0.745，与旧项目 SOTA 持平。BiFPN 历史实验**未显示明显增益**，因此本方向优先级降低，不作为新实验重点。

---

## 2. 推荐组合配置 (A1 + A2)

为验证方向 A 是否正确，建议**只跑一次组合实验**：

| 配置项 | baseline | A1+A2 组合 |
|--------|----------|-----------|
| neck | FPN, num_outs=4 | FPN, num_outs=5 (增加 P2 层) |
| roi_extractor | RoIAlign | DeformableRoIAlign |
| backbone out_indices | (0,1,2,3) | (0,1,2,3) + 额外 conv 生成 stride-2 |

**子方向互相抵消评估**：
- A1 提升小目标分辨率 → 提升 mAPs
- A2 自适应采样 → 同时提升 mAPs 和 mAPl (形态适应)
- 二者**方向一致** (都增强小目标特征)，不会互相抵消，可组合

**预期**：
- mAPs +0.03 ~ +0.05 (小目标分辨率提升)
- mAPl 保持不变或 +0.01 (可变形采样对大目标也有益)
- 若 mAPs 提升 > 0.03 且 mAPl 不下降，方向 A 确认可行

---

## 3. 实现步骤

1. 修改 [experiments/configs/baselines/](../../experiments/configs/baselines/) 新建 `multi_scale.py`
2. backbone 修改为输出 5 个 stage (或加 stride-2 conv)
3. FPN num_outs=5
4. roi_extractor 替换为 DeformableRoIAlign
5. 确认 `featmap_strides` 与 num_outs 对齐

---

## 4. 风险与可行性

| 风险 | 评估 |
|------|------|
| 显存增加 (P2 分辨率高) | A5000 24GB 应可承受 batch=4, 需测试 |
| DeformableRoIAlign 安装 | mmdet 自带, 无需额外编译 |
| A3 BiFPN 历史无增益 | 已放弃 A3, 仅做 A1+A2 |

**结论**：方向 A 可行，A1+A2 组合实验可一次性验证方向正确性。
