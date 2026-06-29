# 方向调整决策 (基于白盒插桩分析)

> 决策时间: 2026-06-30
> 依据: [INSTRUMENTATION_RESULTS.md](../../analysis/instrumentation/INSTRUMENTATION_RESULTS.md) 的 13 项核心发现
> 背景: 白盒插桩分析揭示了 6 项架构固有缺陷 (跨数据集一致) 和若干数据集效应,需据此重新评估各方向价值

## 一、接龙顺序调整

### 原顺序 (基于黑盒分析)
D (BoxRefineNet) → B (DecoupledHead) → C (Morphology) → F (StructuredPrior) → A (P1+Deformable) → E (ClassBalanced)

### 调整后顺序 (基于白盒分析, trajectory bug 修复后)
**E → F → D' (新增) → H (可选验证)**

跳过: C (瓶颈在 cls_head 不在特征), A (scale_affects=False, 尺度非主要瓶颈)

> 修正: 初版顺序为 "E → F → H → D'", 因 trajectory bug 误判 H 高价值。修复后 H 降为可选, D' 升级为 F 之后的首选。

### 各方向价值评估

| 方向 | 状态 | 白盒诊断 | 价值 | 决策 |
|------|------|---------|------|------|
| D (BoxRefineNet) | 完成 +0.002 | ❌ renewal 推理时 delta=0 未触发; trajectory 与 baseline 一致 | 已证伪 | 无需继续 |
| B (DecoupledHead) | 早停 at epoch 86, +0.004 | ⚠️ 未解决 Y 坍塌; 仅"更谦虚"获小幅提升 | 价值有限 | **已停** |
| C (Morphology) | 跳过 | ⚠️ 瓶颈在 cls_head (same_group_too_similar=False); ShapeAttention 作用于特征层 | 价值存疑 | **跳过** |
| A (P1+Deformable) | 跳过 | ⚠️ scale_affects=False (尺度对特征影响微弱) | 价值存疑 | **跳过** |
| F (StructuredPrior) | 排队 | ✅ 针对 early_x0_quality=0.280 (有改善空间); renewal 稳定 x0 步间一致性 | 有价值 | **保留** |
| E (ClassBalanced) | **在跑** | ✅✅ 直接针对 Y 类坍塌根因 (class_collapse=[23]) | 最高价值 | **优先** |
| **H (采样效率)** | 新增 (价值下调) | ⚠️ box 第 3 步收敛 (4步中), 减步空间仅 4→3 | 价值有限 | **可选验证** |
| **D' (reg 校正)** | 新增 | ✅ reg 系统性收缩 dw/dh≈-1.6 (架构固有, 跨数据集) | 高价值 | **新增** |

### 调整理由

1. **停 B**: epoch 86, best at 74, 后 12 epoch 零提升 → 已收敛; 白盒显示 +0.004 来自"谦虚"副作用, 未解决 Y 坍塌和同组混淆
2. **跳 C**: 白盒证实分类瓶颈在 cls_head 决策边界 (same_group_too_similar=False), 不在特征抽取; ShapeAttention 作用于特征层, 但特征已足够区分
3. **跳 A**: 白盒显示 scale_affects=False (小/中/大目标特征范数接近 5.0/4.9/4.8), 尺度非主要瓶颈
4. **优先 E**: Y 类坍塌在 7 个 ckpt 普遍存在, 是分类瓶颈根因; 不解决此问题, 其他分类改进都有上限
5. **新增 H (价值已下调)**: box 第 3 步收敛 (4 步中), 减步空间仅 4→3 (1.33× 加速), 且在收敛边界 — 从"高优先级"降为"可选验证"
6. **新增 D' (价值最高)**: reg 系统性收缩 dw/dh≈-1.6 是架构固有 (24obj 上 -1.3, 方向一致), 应针对此偏移设计 (区别于原 D 的 renewal 机制)

> 修正说明: 初版基于 trajectory 分析 bug 曾误判 "box 第 1 步收敛, H 方案 200× 加速"。修复后真实结果为 box 第 3 步收敛 (4 步采样), H 价值大幅下调。详见 [INSTRUMENTATION_RESULTS.md 附录 A](../../analysis/instrumentation/INSTRUMENTATION_RESULTS.md)。

## 二、方向 H: 扩散采样效率优化 (价值已下调)

### 白盒依据 (修复后真实数据)
- **发现 3 (修正)**: box 在第 3 步收敛 (convergence_step_mean=3.0, 4 步中倒数第 2 步)
- cls 在第 2 步收敛 (cls_converged_ratio=0.9-1.0), 比 box 更早
- 4 步采样对 box 是必要的: 减到 3 步刚好在收敛边界, 减到 2 步 box 未收敛
- 跨数据集一致 (24obj 也是第 3 步收敛), 确认为架构固有

### 已确认: 实际采样 4 步 (非 200)
- 配置 `sampling_timesteps=4`, 采样器 Rectified Flow + Heun solver
- 初版报告 n_steps=200 是 bug (跨图混合), 已修复

### 子方向设计

#### H1: 减少采样步数 (价值有限)
- **做法**: 修改配置 `sampling_timesteps` 从 4 → 3 → 2
- **实现**: 仅改配置, 无需改代码
- **验证**: 对比 mAP, 找到 mAP 不降的最小步数
- **预期 (修正)**:
  - 4→3: box 刚好在收敛边界, mAP 可能保持, 仅 1.33× 加速
  - 4→2: box 未收敛, mAP 可能下降
  - 4→1: box 远未收敛, mAP 肯定下降
- **结论**: 加速空间有限, 降为可选验证

#### H2: 非对称采样 (不适用)
- **做法**: 修改 `predict` 方法, box 在前 K 步更新, 后续步仅更新 cls_logits
- **问题**: 修复后数据显示 cls (第 2 步) 比 box (第 3 步) 更早收敛 → "cls 继续精化"无意义
- **结论**: 放弃 H2

#### H3: 动态步数 (复杂度高, 收益不确定)
- 略

### 实施计划
1. **H1 可选** (仅改配置): 若 E/F/D' 之间有空闲, 跑 `sampling_timesteps=3` 验证 mAP 是否保持
2. H2/H3 放弃

### 配置文件
```python
# experiments/configs/ldmdet/direction_h_sampling_efficiency.py
_base_ = ['./rf_heun_adaln.py']
model = dict(
    bbox_head=dict(
        sampling_timesteps=3,  # H1: 从 4 减到 3 (收敛边界)
    ),
)
```

## 三、方向 D': 回归偏移校正

### 白盒依据
- **发现 5**: reg 系统性收缩, dw/dh ≈ -1.6 (baseline), -1.3 (24obj), -1.4 (focal_gamma_3)
- **发现 8 (24obj 对照)**: 收缩方向跨数据集一致 (都偏负), 确认为架构固有
- 这**不是保守回归** (is_conservative=False, 因 |delta_mean| > 0.1), 而是系统性缩小框
- dx, dy 接近 0, 说明中心定位无偏, 仅尺度预测偏小

### 根因推测
- 训练数据中 GT 框可能偏大 (标注习惯), 模型学习到"收缩"补偿
- 或扩散过程的 box 初始化偏大, reg_head 学习补偿
- 或 L1 loss 对 dw/dh 的对称性导致模型学到均值偏移

### 子方向设计

#### D'1: 后处理尺度校准 (最简, 优先实施)
- **做法**: 推理时对预测框的 w/h 加校准因子 (e.g., ×1.16 补偿 -1.6 偏移)
- **实现**: 修改 predict 方法的后处理, 或加 test_pipeline 后处理
- **校准因子推导**: 从白盒分析的 per_dim_mean 反推
  - dw=-1.6 → exp(-1.6)≈0.20, 即预测 wh 是 GT 的 ~20%? 不对
  - 实际: delta 是 reg_head 输出, 需看 box 编码方式
  - 若 delta 直接是 log 空间偏移: 校准因子 = exp(1.6) ≈ 4.95? 需确认编码
- **预期**: 简单校准可能改善 mAP, 但全局校准不适应类别/尺度差异
- **风险**: 校准因子可能因类别/尺度不同而异, 全局校准可能损害部分类

#### D'2: reg_head 正则化 (训练时约束)
- **做法**: 训练时加 reg loss 项, 约束 dw/dh 均值接近 0
- **实现**: 修改 loss 计算, 加 `reg_bias_loss = (delta[..., 2:].mean()) ** 2`
- **预期**: 减少系统性偏移, 但可能增加训练难度
- **复杂度**: 中等

#### D'3: 数据增强 (box 尺度抖动)
- **做法**: 训练时对 GT 框加随机尺度抖动 (e.g., ±10%), 打破系统性偏移
- **实现**: 修改 train_pipeline, 加 ScaleJitter transform
- **预期**: 增强尺度鲁棒性, 减少偏移
- **风险**: 可能影响小目标检测

#### D'4: box 参数化改进
- **做法**: 改用不同的 box 编码 (e.g., 直接预测 xyxy 而非 cxcywh + delta)
- **实现**: 修改 box 编码/解码逻辑
- **复杂度**: 高 (涉及训练和推理)

### 实施计划
1. **D'1 优先** (后处理校准, 最简): 先确认 box 编码方式, 推导校准因子, 跑校准实验
2. D'1 验证后, 若有效但全局校准不够, 再做 D'2 (训练时正则)
3. D'3/D'4 作为备选

### 配置文件
```python
# experiments/configs/ldmdet/direction_d_prime_reg_calibration.py
# D'1: 后处理尺度校准
_base_ = ['./rf_heun_adaln.py']
model = dict(
    bbox_head=dict(
        reg_calibration=dict(  # 新增后处理校准
            type='RegCalibration',
            wh_scale=1.16,  # 校准因子, 需根据编码方式推导
        ),
    ),
)
```

## 四、接龙实验状态跟踪

| 方向 | 状态 | mAP | best_epoch | 备注 |
|------|------|-----|-----------|------|
| baseline (rf_heun_adaln) | 基线 | 0.745 | 102 | multi_seed_aug |
| D (BoxRefineNet) | ✅ 完成 | 0.747 | 55 | +0.002, renewal 未生效 |
| B (DecoupledHead) | ⏹ 早停 | 0.749 | 74 | +0.004, "谦虚"效应, epoch 86 停 |
| E (ClassBalanced) | 🔄 在跑 | — | — | PID 2692560, GPU 0 |
| F (StructuredPrior) | ⏳ 排队 | — | — | 下一轮 |
| D' (reg 校正) | ⏳ 规划中 | — | — | D'1 后处理校准 (F 之后首选) |
| H (采样效率) | ⏳ 可选 | — | — | H1 仅改配置 4→3 (价值有限) |

## 五、复现命令

```bash
# E (在跑)
python experiments/runners/train.py experiments/configs/ldmdet/direction_e_class_balanced.py \
    --work-dir work_dirs/direction_exps/direction_e_class_balanced --seed 42 --gpu-id 0

# F (排队)
python experiments/runners/train.py experiments/configs/ldmdet/direction_f_structured_prior.py \
    --work-dir work_dirs/direction_exps/direction_f_structured_prior --seed 42 --gpu-id 0

# H1 (仅改配置, 无需训练, 用已有 ckpt 推理)
# 待 H 配置创建后补充

# D'1 (后处理校准, 待实现)
# 待 D' 配置创建后补充
```
