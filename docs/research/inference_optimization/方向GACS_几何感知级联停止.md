# GACS：几何感知级联停止

> 状态：已实现并完成验证（2026-08-08）
> 结论：GACS 是**无额外网络前向（NFE）的动态 1/2 步推理机制**。它能在当前 BoxChart-RF 模型上保持 mAP，同时减少平均采样步数和平均延迟；现有实验**不支持其提升 mAP**，因此应归类为速度–精度 Pare托优化，而不是精度创新点。

## 1. 动机以及与旧 IO1 的区别

旧 IO1 使用相邻 solver 时间步的 $x_0$ 或类别变化判断收敛。它有三个问题：必须先支付前一 NFE、不同步数的时间网格并不嵌套、欧氏框坐标中的差值也不保证位于合法框流形上。旧结论见 `docs/research/archived/自适应步数提前终止.md`。

GACS（Geometry-Aware Cascade Stopping）不复用该判据。它在**第一次网络前向内部**比较最后两个级联头（head 5 与 head 6）的输出，若当前样本已经达到级联固定点附近，就直接采用第 1 步结果，否则执行第 2 步。适用范围被刻意限制为同一个精确的 1/2 步调度，不外推到非嵌套的 2/4 步调度。

## 2. 理论依据

令同一时间 $t$、同一 RoI 特征条件下的末级级联更新为

$$z_{k+1}=T_t(z_k),\qquad z=(u,p),$$

其中 $u$ 是 BoxChart 中的无约束框坐标，$p$ 是类别后验。若在当前样本的局部邻域内 $T_t$ 为压缩映射，存在 $q<1$ 使

$$\|T_t(a)-T_t(b)\|\le q\|a-b\|,$$

则 Banach 不动点定理给出唯一固定点 $z^*$，并有后验误差界

$$\|z_6-z^*\|\le \frac{\|z_6-z_5\|}{1-q}.$$

因此末两级残差小是“当前 NFE 已接近级联条件固定点”的可计算充分证据。这里不能把它理解为全局保证：网络并未被显式约束为全局压缩，$q$ 也未被直接估计，所以 GACS 还同时加入类别残差和置信度安全门。

对每张图像取最终置信度最高的 $K=100$ 个 proposals，定义：

$$r_{geo}=\frac1K\sum_{i=1}^K\sqrt{\frac1d\|u^{(6)}_i-u^{(5)}_i\|_2^2},$$

$$r_{cls}=\frac1K\sum_{i=1}^K\left|\max_c\sigma(l^{(6)}_{ic})-\max_c\sigma(l^{(5)}_{ic})\right|,$$

$$\bar s=\frac1K\sum_{i=1}^K\max_c\sigma(l^{(6)}_{ic}).$$

停止条件为

$$r_{geo}\le 0.02,\qquad r_{cls}\le0.0079,\qquad \bar s\ge0.9.$$

`r_geo` 在 BoxChart 的 RF 原始坐标中计算，其中形状使用合法 simplex 的 gap-ILR 表示；这使残差不受非法宽高或边界裁剪伪象污染。实验表明几何残差单独使用会伤害小目标 AP，而类别残差能更好地区分仍需第 2 步的困难样本。

## 3. 具体实现

### 3.1 代码与配置入口

- 主逻辑：`ldmdet/core/head.py`
  - 构造参数：`adaptive_stopping`、`adaptive_min_steps`、三个阈值和 `adaptive_topk`
  - `_cascade_consistency_metrics`：计算 head 5→6 的几何、类别、置信度及框尺度统计
  - `_adaptive_stop_mask`：组合安全门
  - `_forward_at_t`：从同一次 NFE 返回判据，不增加网络计算
  - 采样循环：第 1 步后仅在整批样本均满足时提前结束，并记录实际步数
- BoxChart：`ldmdet/diffusion/box_chart.py`
- 采样器：`ldmdet/diffusion/sampling.py`
- 最终配置：`experiments/configs/ldmdet/directions/boxchart_rf/gacs_boxchart_rf_24obj.py`
- 评估脚本：`experiments/analysis/eval_gacs.py`
- 单元测试：`tests/test_adaptive_stopping.py`，并与 `tests/test_box_chart.py`、`tests/test_mccr.py` 联合通过（8 passed）。

### 3.2 最终运行约束

- `sampling_timesteps=2`，`adaptive_min_steps=1`
- `use_ensemble=False`，确保输出是最后执行步而非多步集成
- Dataset 2 实验使用 `box_renewal=False`。GACS 残差在同一次 NFE 的级联内部计算，理论上不要求跨 solver 步保持 proposal 身份；Dataset 1 因此另行验证了 `box_renewal=True`：未早退样本在门控后正常 renewal 再进入第 2 步
- 验证批大小为 1；代码对批量的语义是“整批均通过才退出”，因此吞吐场景需另行验证动态分组
- checkpoint：`work_dirs/boxchart_rf_24obj/best_coco_bbox_mAP_epoch_3.pth`

### 3.3 已修复的基线错误

旧代码在 `use_ensemble=False` 时仍只保留第一步结果，注释却写“保留最后一步”。这使早期 step sweep 的 2-step 指标混入随机初始框差异，不能用于 GACS 精度结论。现已改为每步替换结果、最终保留最后执行步。

修复前的 `work_dirs/diagnosis/gacs_calibration_no_exit.json` 和 `work_dirs/diagnosis/gacs_geo_sweep.json` 仅作排错历史，不应引用为精度证据；后续表格只使用修复后结果。

## 4. 实验结果与来源

### 4.1 最终门控，多随机种子

| 随机种子 | 模式 | mAP | AP50 | AP75 | APs | APm | APl | 早退率 | 平均步数 | 平均延迟 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 固定 2 步 | 0.857 | 0.987 | 0.967 | 0.554 | 0.853 | 0.900 | 0 | 2.000 | 58.92 ms |
| 42 | GACS（最终精确配置） | 0.858 | 0.987 | 0.968 | 0.557 | 0.854 | 0.902 | 0.332 | 1.668 | 52.22 ms |
| 0 | 固定 2 步 | 0.858 | 0.987 | 0.967 | 0.556 | 0.854 | 0.903 | 0 | 2.000 | 58.70 ms |
| 0 | 分类门控验证 | 0.858 | 0.986 | 0.967 | 0.563 | 0.854 | 0.902 | 0.336 | 1.664 | 48.25 ms |
| 1 | 固定 2 步 | 0.858 | 0.987 | 0.967 | 0.500 | 0.854 | 0.901 | 0 | 2.000 | 58.92 ms |
| 1 | 分类门控验证 | 0.858 | 0.986 | 0.967 | 0.502 | 0.855 | 0.901 | 0.326 | 1.674 | 48.76 ms |

来源：

- seed 42 固定基线：`work_dirs/diagnosis/gacs_box_scale_calibration.json`
- seed 42 最终配置：`work_dirs/diagnosis/gacs_final_seed42_repeat.json`（首轮计时见 `gacs_final_seed42.json`，指标相同但平均延迟 58.65 ms，说明短程延迟测量存在系统抖动）
- seed 0：`work_dirs/diagnosis/gacs_seed0.json`
- seed 1：`work_dirs/diagnosis/gacs_seed1.json`

seed 0/1 验证了同一个 $r_{cls}=0.0079$ 阈值，但运行时关闭了几何和置信度安全上限；最终三门精确配置只在 seed 42 完整复验。由于新增门是保守过滤器，这支持稳健性，但不能冒充三个随机种子的完整配置复现。

### 4.2 阈值消融

| 判据 | 代表设置 | 结果 | 判定 | 来源 |
|---|---|---|---|---|
| 仅几何 | 0.0055–0.0092 | mAP 0.858–0.859，但 APs 0.504–0.526，低于固定 2 步 0.554 | 拒绝 | `work_dirs/diagnosis/gacs_geo_sweep_fixed.json` |
| 分类残差 | 0.0075 | 早退 25.0%，mAP 0.858，APs 0.555 | 可用但保守 | `gacs_cls_gate_00075.json` |
| 分类残差 | 0.0077 | 早退 28.6%，mAP 0.858，APs 0.556 | 可用 | `gacs_cls_gate_00077.json` |
| 分类残差 | 0.0079 | 早退 34.4%，mAP 0.858，APs 0.556 | 选定 | `gacs_cls_gate_00079.json` |
| 分类残差 | 0.0087 | 早退 50.2%，但 APs 降至 0.517 | 拒绝 | `gacs_cls_gate_00087.json` |
| 最小框尺度门 | 0.050/0.056 | 未恢复 APs | 拒绝并在最终配置置 0 | `gacs_small_gate_005.json`、`gacs_small_gate_0056.json` |

### 4.3 Dataset 1（Chromosome20240904）验证

Dataset 1 的旧框参数化、置信度和级联残差分布与 Dataset 2 明显不同，不能直接迁移阈值：其分类残差中位数约 0.0157，且 top-k 平均置信度最大值低于 0.9。按预注册原则，只用 seed42 checkpoint 校准，选择较保守的 `geo=0.005, cls=0.013, score=0.7`，然后冻结到 seed123/789。

生产相关配置保留 `box_renewal=True`：

| 训练种子 | 固定 2 步 mAP | GACS mAP | APs（固定→GACS） | 早退率 | 平均步数 |
|---:|---:|---:|---:|---:|---:|
| 42（校准） | 0.737 | 0.736 | 0.483→0.485 | 24.8% | 1.752 |
| 123（冻结） | 0.740 | 0.739 | 0.508→0.506 | 38.4% | 1.616 |
| 789（冻结） | 0.740 | 0.740 | 0.499→0.497 | 52.7% | 1.473 |
| **均值** | **0.7390** | **0.7383** | **0.4967→0.4960** | **38.6%** | **1.614** |

平均 ΔmAP=−0.0007，属于无可测损失；平均 solver 步数下降 19.3%。seed123/789 的平均延迟分别从 54.35/54.07 ms 降到 46.84/44.07 ms。短程计时仍受系统抖动影响，因此以步数为主要效率证据。

来源：

- 配置：`experiments/configs/ldmdet/directions/boxchart_rf/gacs_chr2024.py`
- 汇总：`work_dirs/diagnosis/gacs_chr2024_summary.json`
- 原始结果：`gacs_chr2024_renewal_seed{42,123,789}_fixed2.json`、`gacs_chr2024_renewal_seed{42,123,789}.json`
- seed42 重复计时：`gacs_chr2024_renewal_seed42_repeat.json`
- 无 renewal 机制对照：`gacs_chr2024_seed{42,123,789}*.json`，复现配置快照为 `gacs_chr2024_no_renewal.py`

必须同时报告的限制：Dataset 1 已训练 4 步 checkpoint 的三种子均值约 0.747，而 GACS 1/2 步为 0.738，仍低约 0.009。故 Dataset 1 结论是“相对固定 2 步无损并减少约 19% solver 步数”，**不是相对现有 4 步最高精度配置无损**。要进入生产默认配置，需要研究 2/4 步的嵌套候选或训练时显式适配低步推理。

## 5. 严格结论与局限

1. mAP 的 +0.001 处于评估随机性/量化范围内，不能声称提升；三种子结果支持的是“无可测损失”。
2. GACS 平均步数约下降 16.3%–16.8%，但端到端平均延迟受 GPU 热身和系统抖动影响，报告时应同时给步数、早退率与多轮计时。
3. 小目标是门控最敏感子集。几何收敛不等价于类别已稳定，解释了最终判据为何以分类残差为主。
4. GACS 是推理优化，可作为最终系统贡献之一，但不能承担论文所需的独立精度创新点。
