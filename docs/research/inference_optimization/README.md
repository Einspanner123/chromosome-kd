# 推理计算优化方向总览

> **目标**: 在不损失（或极少损失）mAP 的前提下，降低 LDMDet 推理延迟。
>
> **当前基线**: RF + Heun 4步 (**7 NFE**, 最后一步退化为 Euler) × 6 级联头 × 500 proposals = 42 次单头前向
>
> **实测延迟**: 335 ms/图 (Backbone 26ms + 采样循环 309ms)
>
> **代码位置**: [ldmdet/core/head.py](../../../ldmdet/core/head.py)、[ldmdet/diffusion/sampling.py](../../../ldmdet/diffusion/sampling.py)、[ldmdet/diffusion/rectified_flow.py](../../../ldmdet/diffusion/rectified_flow.py)、[ldmdet/core/single_head.py](../../../ldmdet/core/single_head.py)

> **⚠️ 更新 (2026-07-07)**: 原 IO1-IO5 方向已通过 [benchmark_inference.py](../../../experiments/analysis/benchmark_inference.py) 实测验证，多数被证伪。新方案见 [实证优化方案.md](./实证优化方案.md)。

---

> ⚠️ **暂时废弃**：以下瓶颈实测数据（335 ms/图、RoIAlign 52.2%、DynamicConv 37.5%、single_head_mean_ms=6.7662 等）基于旧数据集 Chromosome20240904 的 checkpoint（`work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth`，mAP≈0.753），24obj 数据集上的推理优化结论待验证。
>
> 注：优化方向 A-F 的理论分析（计算复杂度、跨步/跨头正交性、加速比推导）为架构层面，与数据集无关，保留有效；§4 验证方法中引用的 checkpoint 亦为旧数据集产物，后续需在 24obj checkpoint 上重测。

## 1. 实测瓶颈分析 (2026-07-07 实测)

### 1.1 组件级延迟 (单次 single_head)

| 组件 | 均值 (ms) | 占比 | 说明 |
|------|:---:|:---:|------|
| **RoIAlign** | **3.53** | **52.2%** | 第一瓶颈, O(N·P²) |
| **DynamicConv** | **2.54** | **37.5%** | 第二瓶颈, O(N·P²·d) |
| Self-Attn | 0.15 | 2.2% | 已用 SDPA, 非瓶颈 |
| 其他 | 0.22 | 3.3% | FFN + cls + reg |

**关键发现**: Self-Attn 仅占 2.2%，原 IO 方向假设的 "N² attention 瓶颈" 不成立。真正瓶颈是 RoIAlign + DynamicConv (89.7%)。

### 1.2 计算路径

```
backbone+FPN (1次, 26ms)
  └─ 采样循环 ×4步 (Heun, 7 NFE)
       ├─ _forward_at_t (每 NFE): 6级级联头
       │   每头: RoIAlign(500, 7×7) → DynamicConv → SelfAttn(SDPA) → FFN → cls/reg
       └─ heun_step model_fn (非末步): 再跑 6 头前向
```

**总开销**: 7 NFE × 6 级联头 = **42 次单头前向** (非原假设的 48 次)。

### 1.3 什么变了、什么没变

| 组件 | 跨时间步 | 跨级联头 | 说明 |
|------|:---:|:---:|------|
| Backbone/FPN 特征 | 不变 | 不变 | 已复用 ✓ |
| RoI 特征 | **变** | **变** | bbox 随 x_t 演化, 实测位移 93-124 px/步 |
| Self-Attn Q/K/V | **变** | **变** | 依赖 proposal 特征 |
| DynamicConv | **变** | **变** | 依赖 proposal + RoI |
| AdaLN 调制 | **变** | 不变 | 仅依赖 t，同 NFE 内 t 固定 |
| cls/reg 输出 | **变** | **变** | 依赖 fc_feature |

---

## 2. 原 IO1-IO5 方向实测判定

| 方向 | 核心思想 | 实测判定 | 实证依据 | 文档 |
|------|---------|:---:|------|------|
| **IO1** 自适应步数终止 | 收敛后提前结束采样循环 | **证伪** | x0_Δrel 最低 0.166 (>>0.01 阈值), cls 一致率仅 65% | [IO1](./方向IO1_自适应步数提前终止.md) |
| **IO2** 投机 Draft-Verify | 1步草稿+多步验证 | **证伪** | 早期步 cls 一致率 57%, draft 不可靠 | [IO2](./方向IO2_投机Draft_Verify.md) |
| **IO3** Top-K 框剪枝 | N: 500→100 | **验证成功, 推荐K=300** | K=300 加速 1.34x, mAP Δ=-0.001；无需重训 | [IO3](./方向IO3_TopK框剪枝.md) |
| **IO4** 级联头提前退出 | 跳过收敛的后续头 | **证伪** | 所有阈值退出率均为 0%，级联头是主动精炼设计而非冗余 | [IO4](./方向IO4_级联头提前退出.md) |
| **IO5** 跨步 RoI 特征缓存 | 复用 RoI 特征 | **证伪** | 框位移 93-124 px/步, RoI 特征剧变 | [IO5](./方向IO5_跨步RoI特征缓存.md) |

### 已实现方向

| 方向 | 状态 | 说明 |
|------|:---:|------|
| DPM-Solver++ | ✓ 已实现 | 历史 x0_pred 多项式插值，1 NFE/step。[rectified_flow.py](../../../ldmdet/diffusion/rectified_flow.py#L83) |
| FP16/BF16 AMP | ✓ 已实现 | `amp_dtype` + `attn_half`。[head.py#L126](../../../ldmdet/core/head.py#L126) |
| SDPA Flash Attention | ✓ 已实现 | `use_sdpa`。[single_head.py#L89](../../../ldmdet/core/single_head.py#L89) |
| **IO3 Top-K 剪枝** | ✓ 已实现 | `topk_k=300` 推理时剪枝，无需重训。[head.py](../../../ldmdet/core/head.py) |

---

## 3. 新实证优化方向

详见 **[实证优化方案.md](./实证优化方案.md)**

### 立即可验证 (推理修改, 无需重训)

| 方向 | 核心思想 | 预期加速 | 精度风险 |
|------|---------|:---:|:---:|
| **A: 级联头跳过** | 推理时跳过末 1-2 头 (修订 IO4) | 1.17-1.33x | 中 |
| **B: DPM-Solver++ 切换** | Heun 7NFE → DPM++ 4NFE (纯配置) | 1.75x | 低-中 |
| **C: Proposal 剪枝** | 第 1 步后截断到 top-300 (修订 IO3) | 1.23x | 低 |
| **D: 步数减少** | 4 步 → 3 步 | 1.4x | 中 |

### 需重训 (架构修改)

| 方向 | 核心思想 | 预期加速 | 精度风险 |
|------|---------|:---:|:---:|
| **E: RoIAlign 瘦身** | output_size 7→5, sr 2→0 (52.2%瓶颈) | 1.43x | 中 |
| **F: DynamicConv 瘦身** | dynamic_dim 64→32 (37.5%瓶颈) | 1.20x | 中 |

### 推荐组合

根据 IO3 实测：**IO3 K=300 单项即可获得 34% 加速，且 mAP 几乎无损**（Δ=-0.001），无需组合其他方向。

```
推荐默认配置: IO3 K=300 → 加速 1.34x, mAP 降 0.001
```

---

## 4. 验证方法

在 SOTA checkpoint (`a3_full_sota`, mAP=0.858) 上做**离线推理对比**：

```bash
# 延迟验证
python experiments/analysis/benchmark_inference.py --num-images 50

# 精度验证 (完整 val set)
python tools/test.py <config> <ckpt>
```

**通过标准**: mAP 下降 < 0.005 且加速 > 1.5x
