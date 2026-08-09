# CALU：COCO 阈值对齐的终点定位效用

## 1. 决策背景

本方向是在两个候选被 Phase-0 证伪后提出的：

- CC-RF（容量约束集合 RF）与既有 KCEC/非平衡 OT 实质重合。约束只改变训练配对，进入检测器后仍退化为独立 `(x_t, target)` 样本；KCEC 与非平衡 OT 的真实实验约为 0.744--0.748/0.746，未提供独立增益，因此不重复训练。
- LVD-RF（方向一致性）在 Dataset1 A4 seed42 的 100 张验证图上未通过门控。动态 matcher 下，四个求解时刻的目标方向平均余弦为 0.99983、0.99982、0.99968、0.99924，远高于预注册的 `<0.95` 非冗余门槛。原始随机 coupling 的方向余弦仅约 0.823--0.839，说明若坚持原 coupling 身份，反而会与检测器实际动态分配冲突。

诊断来源：

- `work_dirs/diagnosis/lvd_phase0_chr2024_seed42.json`
- `experiments/analysis/lvd_phase0.py`
- `docs/research/proposals/FEASIBLE_LVD_RF.md`

## 2. 精度瓶颈证据

Dataset1 A4 seed42 在固定 100 张验证子集上的基线 mAP 为 0.76094。各阈值 AP 从 AP75=0.85256 快速下降到 AP90=0.51710、AP95=0.11903。定位 oracle 为 0.95792（+0.19698），分类 oracle 仅为 0.78712（+0.02618），因此主要剩余空间在终点坐标，而非类别分数。

IoU=0.90 下，重叠目标 recall 为 0.48080，小目标 recall 为 0.42512；相应孤立目标为 0.70214，中目标为 0.72465。该现象同时覆盖通用的高精度定位、密集重叠和小目标场景。

诊断来源：`work_dirs/diagnosis/precision_bottleneck_a4_chr2024_seed42_100.json`，生成脚本为 `experiments/analysis/precision_bottleneck_diagnosis.py`。

## 3. 数学定义

令最终级联头的匹配正样本框与 GT 的 aligned IoU 为 `u`，COCO 阈值集合

`T={0.50,0.55,...,0.95}`。

定义软失败率

`L_CALU(u;s) = (1/|T|) sum_{tau in T} sigmoid((tau-u)/s)`。

当 `s -> 0+` 且 `u` 不恰好等于某阈值时，逐点有

`L_CALU -> (1/|T|) sum_tau 1[u<tau]`，

即一个框未通过的 COCO IoU 阈值比例。证明直接来自 sigmoid 的阶跃函数极限。损失有界于 `[0,1]`，其梯度集中在真实评估边界附近；L1/GIoU 继续提供全局几何梯度。与全局提高 GIoU 权重不同，CALU 不会持续放大远离评估边界的样本。

## 4. 实现与预注册门控

- 只对主输出/最后一级级联头添加 CALU；所有 auxiliary head 不添加。
- 快速门控只解冻最后一级 `reg_head`，骨干、颈部、分类器和前五级回归头冻结。
- 固定温度 `s=0.025`、权重 `0.5`、学习率 `1e-4`，从 A4 seed42 最佳 checkpoint 续训 12 epoch，不做超参数搜索。
- 推理图完全不变，因此理论上无速度开销。
- 第一门槛：完整验证 mAP 超过配对 A4 的 0.746，且 AP90/AP95 不以 AP50/AP75 的明显退化换取。
- 若通过，必须进行相同 seed 的完整重训；断点续训只能作为因果门控，不能作为论文最终结果。

配置：`experiments/configs/ldmdet/directions/capr/calu_terminal_reg_chr2024_seed42.py`。

## 5. 论文定位

CALU 本身是损失层创新，理论贡献强于“直接使用 RF/DPM-Solver++”，但单独仍不足以自动构成顶会级主创新。其价值在于把扩散检测器的终点回归与 COCO 的离散 IoU 效用直接连接，并且可无推理开销地迁移到通用集合检测器。只有在完整重训、多 seed、Dataset1/Dataset2 与 mini-COCO 上稳定成立，且高 IoU、小目标/拥挤分层增益一致时，才适合作为独立主创新点。
