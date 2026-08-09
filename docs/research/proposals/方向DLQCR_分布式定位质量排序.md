# D-LQCR：分布式定位质量排序

> 状态：Dataset 1 seed42 门控实验进行中（2026-08-09）  
> 基座：RF + DPM-Solver++，冻结 A4 seed42；仅训练末级质量头  
> 代码提交：`28119f61`；严格冻结修复：`25d369b1`

## 1. 动机

标量 LQCR 学习 $q(F)\approx\mathbb E[U\mid C=1,F]$，其中 $U$ 是预测框与
匹配真值的 IoU。单一期望不能唯一确定 COCO 各阈值上的真阳性概率：两个条件 IoU
分布可以具有相同均值，却在 $0.75$ 或 $0.95$ 阈值处具有完全不同的尾概率。因此，
D-LQCR 直接预测 COCO 十个阈值上的生存函数

$$
S_k(F)=P(U\ge \tau_k\mid C=1,F),\qquad
\tau_k\in\{0.50,0.55,\ldots,0.95\}.
$$

## 2. 排序依据

对类别 $c$ 和固定阈值 $\tau_k$，真阳性后验可精确分解为

$$
P(C=c,U\ge\tau_k\mid F)
=P(C=c\mid F)P(U\ge\tau_k\mid C=c,F).
$$

令 $p_c=P(C=c\mid F)$。当 COCO 阈值均匀取值时，单个预测成为真阳性的平均后验为

$$
\bar r_c(F)=\frac1K\sum_{k=1}^{K}p_cS_k(F)
=p_c\,\bar S(F).
$$

按后验概率排序在固定输出预算下最大化期望真阳性数（probability-ranking
principle）。这为 $p_c\bar S$ 提供了比经验式 $p_cq^\beta$ 更直接的目标对应关系；
但由于 AP 还包含全局排序与插值，本式不宣称对任意数据分布严格最大化 COCO AP。

## 3. 单调参数化

生存函数必须满足 $S_1\ge S_2\ge\cdots\ge S_K$。网络不独立输出十个无约束
logit，而是预测首个 logit $z_1$ 和非负递减量：

$$
z_k=z_1-\sum_{j=2}^{k}\operatorname{softplus}(a_j),\qquad
S_k=\sigma(z_k).
$$

因此对任意参数和输入均有 $S_{k+1}\le S_k$，无需额外单调正则或推理修正。

训练时，匹配正样本的第 $k$ 个标签为
$y_k=\mathbb 1[U\ge\tau_k]$；未匹配 proposal 的所有标签为零。几何 IoU 在构造
标签前停止梯度，故 D-LQCR 不改变框回归。所有匹配样本的十个阈值（包括真实 IoU
以上的零标签）都参与 BCE；未匹配样本沿用 focal 权重。

## 4. 实现与因果隔离

- 质量头与单调参数化：`ldmdet/core/single_head.py`
- $p_c\bar S$ 最终融合：`ldmdet/core/head.py`
- 阈值标签与损失：`ldmdet/criterion/criterion.py`
- Dataset 1 seed42 配置：
  `experiments/configs/ldmdet/directions/capr/dlqcr_iou_survival_chr2024_seed42.py`
- 实验目录：`work_dirs/dlqcr_iou_survival_chr2024_seed42/`
- 日志：`work_dirs/dlqcr_iou_survival_chr2024_seed42/train.log`

配置保持 `quality_calibration_mode='final_only'` 和
`quality_only_training=True`。backbone、neck、前五级 cascade、分类头、回归头、RF、
DPM-Solver++、renewal 与最终框坐标全部冻结；模型仅有末级质量头的 7 个参数张量可训练。

## 5. 验证记录与门槛

| 设置 | mAP | 相对 A4 | 相对标量 LQCR | 状态 |
|---|---:|---:|---:|---|
| A4 seed42 | 0.746 | — | -0.005 | 固定基座 |
| 标量 LQCR seed42 best | 0.751 | +0.005 | — | 已完成 |
| D-LQCR seed42 epoch 1 | 0.751 | +0.005 | 0.000 | 训练中 |

门槛在查看后续结果前固定如下：

- **强通过**：best mAP $\ge0.753$，即超过标量 LQCR 至少 0.002；
- **弱信号**：best mAP = 0.752，仅允许继续一个独立 seed；
- **不形成新贡献**：best mAP $\le0.751$，保留为 LQCR 的理论化替代消融，不进入主线。

epoch 1 的 `loss_quality` 已从约 0.73 收敛至约 0.09，梯度有限；首轮 mAP 为
0.751。完整 12 epoch 结果完成后更新本节，并按上述预注册门槛决定是否进入 seed123。
