# CAPR：规范轴形态—质量精炼的实现计划

> Canonical-Axis Profile and Quality Refinement（CAPR）
> 目标：形成独立于 RF/solver/GACS 的检测头结构创新，优先提升高 IoU、小而细长、相似类别目标；先在 Dataset 2 建立机制，再在 Dataset 1 验证低数据泛化。

> **2026-08-08 Phase-0 更新**：完整诊断表明分类 oracle 仅 +0.0043，
> 定位 oracle +0.1278；真实同类 IoU 仅重排分数时，$p q^2$ oracle
> 为 0.8999（相对 A4 0.8630 为 +0.0369）。step1→4 的末级 mAP
> 仅 +0.0009，而同一步 head1→6 为 +0.2769。故实施顺序调整为：先做
> 末级 quality-only C2；canonical-axis profile 暂缓，坐标精炼作为 C2
> 证伪后的备选。证据和口径见 `docs/research/精度瓶颈Phase0诊断_20260808.md`。

## 1. 为什么不继续原 M1

已有 M1 `MorphologyAwareRoIEncoder` 使用覆盖完整 7×7 RoI 的 `(7,1)`/`(1,7)` 方向卷积，并通过零初始化 `fuse` 残差接入。FP32 结果为 0.862，相对 +DPM++ 0.863 为 −0.001；方向卷积沿空间维度的能量比仅 1.01–1.02，最终退化成近似常数偏置。

根因不是“形态信息无效”，而是实现没有解决两个问题：

1. 轴对齐 RoI 中目标方向是自由旋转的，固定水平/垂直卷积把姿态变化和类别形态混在一起；
2. 完全零初始化的输出投影使上游方向分支初期梯度近零，无法学出空间差异。

证据来源：`docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md` §3.1.6–3.1.7，代码备份位于 `work_dirs/m1_morphology_aware_24obj_fp32/20260723_191429/ldmdet_backup/core/morphology_encoder.py`，训练日志位于同一实验目录。

CAPR 不恢复这套全核方向卷积，而是显式估计对象主轴、进行 $\pi$ 周期规范化，再提取一维轴向轮廓；同时把高 IoU 质量作为晚期头的直接训练目标。

## 2. 数学依据

### 2.1 消除旋转 nuisance

令 RoI 表征为 $F=g_\theta(M)+\epsilon$，$M$ 是类别相关形态，$g_\theta$ 是平面旋转作用。未规范化特征的类内协方差满足全方差分解

$$\operatorname{Cov}(F\mid Y)=\mathbb E_\theta[\operatorname{Cov}(F\mid Y,\theta)]
+\operatorname{Cov}_\theta(\mathbb E[F\mid Y,\theta]).$$

第二项完全来自姿态 nuisance。若估计主轴 $\hat\theta$ 并构造 $C(F)=g_{-\hat\theta}(F)$，且估计误差有界，则该项随角度误差方差下降；在不压缩类间形态差异时，Fisher 比率

$$J=\frac{\operatorname{tr}S_B}{\operatorname{tr}S_W}$$

上升。这是 CAPR 可能改善相似细长类别判别的理论来源。

### 2.2 主轴估计与 $\pi$ 不变性

由 RoI 特征预测软前景权重 $a_{ij}\ge0$，计算归一化二阶矩

$$\Sigma=\frac{\sum_{ij}a_{ij}(r_{ij}-\mu)(r_{ij}-\mu)^\top}{\sum_{ij}a_{ij}+\varepsilon},$$

主轴角为

$$\hat\theta=\frac12\operatorname{atan2}(2\Sigma_{xy},\Sigma_{xx}-\Sigma_{yy}).$$

染色体/细长物体的方向只有模 $\pi$ 意义。规范化后对正向与反向轴向 profile 共享编码并平均：

$$e_{shape}=\tfrac12\{h(P_{\hat\theta})+h(\operatorname{flip}P_{\hat\theta})\},$$

从结构上消除头尾任意性。

### 2.3 AP 排序与质量估计

在 IoU 阈值 $\tau$ 下，理想排序分数是

$$s_\tau(x)=P(Y=c,\operatorname{IoU}\ge\tau\mid x).$$

仅用分类概率不能表达严格定位质量。CAPR 在最后级联头预测 $q_\tau\approx P(\operatorname{IoU}\ge\tau\mid Y=c,x)$，推理分数采用

$$s=p_c\,q^\beta.$$

这直接对应当前“AP50 接近饱和、mAP/AP85–95 仍有空间”的瓶颈，而不是泛化地增加参数。

## 3. 模块设计

### 3.1 `CanonicalAxisProfileRefiner`

新文件：`ldmdet/core/canonical_axis_refiner.py`

输入：`roi_features [N,C,7,7]`、`proposal_features [N,C]`。输出：`delta_feature [N,C]`、`quality_logit [N,1]`、诊断字典。

计算流程：

1. `1×1 conv → sigmoid` 得到软前景图；
2. FP32 计算质心、协方差和主轴，避免 AMP 下 `atan2/eigh` 不稳定；
3. 用 `affine_grid/grid_sample` 旋转到规范轴；
4. 分别沿短轴平均和最大池化，得到长度为 7 的轴向 profile tokens；
5. 共享的轻量 1D depthwise/pointwise encoder 编码正反 profile，并平均保证 $\pi$ 不变；
6. `LayerNorm → Linear` 生成形态残差；输出层用正常初始化，外部 LayerScale `γ=1e-3`，使行为近似恒等但首步梯度不被截断；
7. quality head 读取 `[proposal_feature, e_shape]`，预测 IoU quality。

原计划首版接入最后两个 cascade heads。Phase-0 后的 C2 quality-only 实现进一步
收缩到**仅第 6 级头**：前五级不构建 quality 参数，保证新增损失和推理重排均可
单独归因；profile/坐标精炼若进入后续实验，再考虑第 5–6 级。

### 3.2 接入位置

- `ldmdet/core/single_head.py`
  - 增加可选 `canonical_refiner=None`
  - RoIAlign 后、DynamicConv 前计算形态 embedding
  - DynamicConv 后将 `γ·delta_feature` 加到 proposal feature
  - `_predict` 附带 quality logit；关闭模块时返回接口与当前实现严格兼容
- `ldmdet/core/head.py`
  - 支持 `canonical_refiner_start_head=4`
  - 仅 head 4/5 构建 refiner，不在六个头间共享参数
  - 训练时收集末级 quality 输出，推理时校准最终 score
- `experiments/mmdet_bridge/detector.py`
  - 从 registry 构建 refiner 配置
- `ldmdet/loss/criterion.py`（以实际 criterion 文件为准）
  - 对已匹配正样本以 stop-gradient IoU 作为 quality target
  - 首版使用 BCE/Varifocal 形式，权重从 0.25 起
- `experiments/configs/ldmdet/directions/capr/`
  - `capr_profile_only_24obj.py`
  - `capr_quality_only_24obj.py`
  - `capr_full_24obj.py`

## 4. 防止再次退化成常数分支

实现必须暴露并保存以下诊断：

- `foreground_entropy`、前景图空间方差
- 主轴各向异性 $\lambda_1/(\lambda_2+\epsilon)$
- `delta_feature` norm 与 LayerScale
- refiner 第一层/最后层梯度 norm
- 原图与旋转增强后的 canonical embedding cosine
- quality 与真实 IoU 的 Spearman 相关系数及 ECE

单元测试需要证明：

1. 输入旋转 90° 后主轴相应旋转，canonical profile 近似不变；
2. 输入翻转后正反平均严格不变；
3. 初始化时输出扰动很小，但第一次反传后前景层和 profile encoder 梯度均非零；
4. 禁用模块时 checkpoint 加载和输出完全向后兼容；
5. FP16/BF16 前向有限，主轴矩计算保持 FP32。

## 5. 分阶段实验矩阵

### Phase 0：先完成误差分解

实现 oracle classification/localization、AP50:5:95、per-head recall/AP 和 overlap-stratified AP。若主要缺口不是 AP85–95/相似类别，则暂停 CAPR，重新选择结构方向。

### Phase 1：机制最小验证（Dataset 2，seed 42）

| 实验 | profile | quality | 目的 |
|---|:---:|:---:|---|
| B0 | × | × | 相同代码/相同短微调基线 |
| C1 | ✓ | × | 形态规范化是否提供新信息 |
| C2 | × | ✓ | 高 IoU 排序是否为主增益源 |
| C3 | ✓ | ✓ | 完整模型是否互补 |
| C4 | ✓但不旋转 | × | 规范轴的必要性 |
| C5 | ✓但零初始化输出 | × | 复现并验证 M1 梯度瓶颈 |

从 A4/BoxChart 当前选定 checkpoint 微调 12 epoch，FP32，统一初始权重和推理随机种子。机制门槛：C1/C2 至少一个相对 B0 `mAP +0.002`，完整 C3 `mAP +0.003`，并在 AP85–95 或 G21/G22/Y 上出现预期同向变化；否则不进入长训。

### Phase 2：完整验证

- Dataset 2：seed 42/123/789，报告 mean±std、每阈值 AP、类别 AP、APs、延迟和参数量
- Dataset 1：相同冻结超参数从三个 A4 checkpoint 微调，验证低数据条件下是否仍有收益
- 通用性：若具备非染色体检测数据，选择一个含细长/小目标类别的通用数据子集；不把着丝粒专用特征写死在网络中

成立标准：三 seed 平均 mAP ≥ +0.003，至少 2/3 seed 为正，95% bootstrap CI 不明显跨越零；延迟增幅目标 <8%。

## 6. 实施顺序与停止规则

1. 先写 Phase 0 诊断，确认 classification/localization/overlap 的实际上限；
2. 实现独立 refiner 与单元测试，不先改 loss；
3. 跑 C1 profile-only，检查梯度和空间诊断；若再次呈常数，立即停止；
4. 实现 quality target 与 C2；
5. 仅当 C1/C2 至少一个过门槛时组合 C3；
6. seed 42 机制成立后再占用三 seed 训练资源。

CAPR 与 GACS 正交：CAPR 改善单次 NFE 的表征和排序，GACS 根据级联后验残差减少不必要的第二 NFE。最终可以分别消融，再报告组合的精度–速度前沿。
