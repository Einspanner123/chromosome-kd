# 方向 D：扩散过程改进 (Diffusion Process Improvements)

> **目标**：突破扩散采样在定位精度上的上限。诊断显示 IoU=0.9 时 Loc+Miss 误差 58%，框回归的中位数 IoU=0.924 但 P25=0.86，说明 25% 的框定位不足。当前 RF + Heun 采样在少步 (4 步) 下累积误差。
>
> **理论依据**：
> - DiffusionDet: Du et al., "DiffusionDet: Diffusion Model for Object Detection" (CVPR 2023)
> - DPM-Solver++: Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model" (NeurIPS 2022)
> - RefineNet 思想: Cascade R-CNN 的多阶段精化
>
> **当前代码位置**：[ldmdet/diffusion/sampling.py](../../ldmdet/diffusion/sampling.py), [ldmdet/core/head.py](../../ldmdet/core/head.py)

---

## 1. 子方向说明

### D1: 框细化网络 (Box Refinement Net)

**问题**：扩散采样最后一步输出的框，对高 IoU 精度不足 (P25 IoU=0.86，意味着 25% 的框 IoU<0.86)。

**方案**：在扩散采样最后一步后，加一个独立的细粒度回归头，专门优化高 IoU 精度。类似 Cascade R-CNN 的级联精化，但不增加采样步数。

```python
# 在 DiffusionDetHead 中
self.box_refine = nn.Sequential(
    nn.Linear(feat_channels, feat_channels),
    nn.ReLU(),
    nn.Linear(feat_channels, 4),  # dx, dy, dw, dh
)
# forward 最后:
final_boxes = diffusion_sample(...)  # 扩散输出
refine_offset = self.box_refine(roi_features)  # 细化
final_boxes = final_boxes + refine_offset  # 残差精化
```

**收益**：mAP75/mAP90 显著提升，P25 IoU 从 0.86 提升到 0.90+。

### D2: 尺度条件化噪声调度 (已有部分实现)

**问题**：当前 RF 直线路径对所有尺度用相同 t 调度，小目标 SNR 低、去噪困难。

**方案**：已有 [ldmdet/diffusion/scale_conditioned_rf.py](../../ldmdet/diffusion/scale_conditioned_rf.py) 基础，让小目标去噪更早 (更陡的调度)。

**注意**：2a 的 scale_aware_loss 实验证实小目标 mAPs +0.020 但大目标 mAPl -0.081。D2 是**调度层面**的尺度适应 (不是损失权重)，理论上更根本，但仍需验证是否同样有 trade-off。

### D3: 多阶段采样

**问题**：4 步 Heun 采样对弯曲路径累积误差。

**方案**：第一阶段粗定位 (低 IoU 阈值，1-2 步)，第二阶段精定位 (高 IoU 阈值 + 细化网络，2 步)。

**收益**：在相同采样步数下，分配更多步数给精定位阶段。

---

## 2. 推荐组合配置 (D1 + D2 + D3)

**子方向互相抵消评估**：
- D1 框细化网络 → 提升高 IoU 定位
- D2 尺度条件化调度 → 提升小目标去噪
- D3 多阶段采样 → 重新分配采样步数

三者**方向一致** (都旨在提升定位精度)，但需注意:
- D2 的尺度适应可能仍有大小目标 trade-off (如 2a scale_aware_loss 所示)
- D1 的细化网络是事后精化，与 D2 的调度正交
- D3 的多阶段与 D1 的细化网络可融合 (第二阶段即 D1)

**潜在抵消**：D2 和 D1 可能竞争 — D2 调度改善小目标，D1 细化改善高 IoU，若 D2 导致大目标下降，D1 难以补救。因此建议:

**保守组合**：D1 + D3 (不含 D2)
- D1 框细化网络 + D3 多阶段采样 (第二阶段用 D1)
- 不引入 D2 的尺度调度 (避免大小目标 trade-off)
- 若 D1+D3 有效，再单独验证 D2

**激进组合**：D1 + D2 + D3
- 全部组合，一次性验证
- 风险：若 D2 导致大目标下降，无法定位原因

**推荐**：先跑**保守组合 (D1+D3)**，方向正确后再加 D2。

| 配置项 | baseline | D1+D3 保守组合 |
|--------|----------|---------------|
| 扩散采样 | 4 步 Heun 均匀 | 2 步粗 + 2 步精 (D3) |
| 精化网络 | 无 | box_refine 残差头 (D1) |
| 第二阶段 | 无 | box_refine 应用 (D1+D3 融合) |

**预期**：
- mAP75 +0.01 ~ +0.03
- P25 IoU 从 0.86 提升到 0.89+
- 若 mAP75 提升 > 0.015，方向 D 确认可行

---

## 3. 实现步骤

1. 新建 [ldmdet/core/box_refine.py](../../ldmdet/core/) — 细化网络模块
2. 修改 [ldmdet/diffusion/sampling.py](../../ldmdet/diffusion/sampling.py) — 多阶段采样逻辑
3. 新建 [experiments/configs/baselines/](../../experiments/configs/baselines/) `refine_sampling.py`
4. 配置验证后训练

---

## 4. 风险与可行性

| 风险 | 评估 |
|------|------|
| D1 细化网络增加推理时间 | 仅一个 MLP，开销可忽略 |
| D3 多阶段采样实现复杂 | 需修改 DiffusionSampler，但接口不变 |
| D2 尺度调度可能 trade-off | 保守组合不含 D2，规避此风险 |

**结论**：方向 D 可行，D1+D3 保守组合可一次性验证。若有效再考虑 D2。
