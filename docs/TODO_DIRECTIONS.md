# 待做研究方向总览 (TODO Directions)

> 本文档梳理 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 所有进行中或待启动的研究方向。
> 这些方向部分有代码就绪、配置就绪或实验已在运行, 部分仅有理论框架。
> 每个方向附 **可靠数据源地址** (本地服务器路径 / SwanLab project / config 路径)。
> 更新时间: 2026-07-30 (LINEAGE 重新梳理: 修复 R3 状态矛盾—实际仅 seed42 完成, 非 3-seed; 修复 §六 D1 dim_d1_mask 伪造数据; 重编号消除 §十一 断层, 原 §十五~§十七→§十四~§十六; Head Distillation 交叉引用修正 →LINEAGE §七。原 2026-07-28: R1/R2 评审循环完成, 4 方向通过(LVD-RF/TRIP/BEAR/ISLR-RF), 10 方向淘汰→FALSIFIED §十五~§二十二; ReFlow 重试确认方法本质失败 → FALSIFIED §十四, 从本文档移除; 2026-07-27 校验+归档: D3→LINEAGE §六, D1→LINEAGE §十四, M1→FALSIFIED §二十六, SC-RF→FALSIFIED §二十三; 全部代号替换为描述性名称; Few-Shot FBM CrossAttn 中断@ep59)
>
> 📌 **关联文档**:
> - [docs/EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md) (主路线实验脉络, 已完成方向)
> - [docs/FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (已证伪方向归档)
> - [docs/EXPERIMENT_CATALOG.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (实验数据索引)
> - [docs/paper/paper_draft_CN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md) (论文草稿, §6 结论与未来工作)
> - [docs/paper/theory_analysis_RF_DPM.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md) (直线度/参数化/框更新交互 理论深化分析)
>
> ⚠ **状态约定**: 🔄 运行中 / ⛔ 待启动 / ✓ 已完成 / 🔴 已证伪
>
> 📋 **文档流转规则**: 方向完成后, 有效→迁入 EXPERIMENT_LINEAGE.md; 证伪→迁入 FALSIFIED_DIRECTIONS.md; 本文档仅保留 🔄进行中 + ⛔待启动 + 边际待验证方向。
>
> 📝 **命名约定** (2026-07-27): 本文档所有方向使用**描述性名称** (如 "x0/v Prediction 对照" 而非 "R3"), 与 EXPERIMENT_LINEAGE.md 风格一致。SwanLab project name、config 文件名、work_dirs 目录中的内部代号 (如 `r3_vpred` / `a4_dpm_pp` / `m1_morphology_aware`) 保留以兼容工程实现, 但章节标题和索引表不再出现代号。

## 〇、方向索引与状态汇总

| 方向 | 状态 | 优先级 | SwanLab Project |
|------|------|--------|-----------------|
| Few-Shot 跨数据集微调 | ⛔ **探索性** (不在当前论文范围, 6/7 源预训练就绪, FBM CrossAttn best 0.857@ep45 中断) | ~~高~~ | `few-shot-benchmark` |
| 跨数据集扩展 (OT Collapse 普遍性) | ⛔ **探索性** (纯理论, 不在当前论文范围) | ~~中~~ | — |
| 速度引导自适应 Renewal | ⛔ 待系统评估 (代码就绪) | 中 | `ldmdet-mainline-ablation-24obj` |
| Brenier 映射神经化 | ⛔ 未开展 (纯理论, TMI 投稿后) | 低 | — |
| 级联头角色分化 | ⛔ 待启动 (零代码改动) | 中-高 | (待创建) |
| LVD-RF (Lyapunov 速度方向正则) | ⛔ 待启动 (R2 通过, 7.5/10) | 高 (优先级 1) | (待创建) `ldmdet-mainline-ablation-24obj` |
| TRIP (Tikhonov-Morozov 反问题正则) | ⛔ 待启动 (R2 通过, 7.5/10) | 高 (优先级 2) | (待创建) `ldmdet-mainline-ablation-24obj` |
| BEAR (反向误差感知正则) | ⛔ 待启动 (R2 通过, 7.3/10) | 中 (优先级 3) | (待创建) `ldmdet-mainline-ablation-24obj` |
| ISLR-RF (输入空间 Lipschitz 正则) | ⛔ 待启动 (R2 通过, 7.4/10) | 中 (优先级 4) | (待创建) `ldmdet-mainline-ablation-24obj` |

## 一、Few-Shot 跨数据集微调

> 配置就绪, ⛔ 待启动 (等待 FBM CrossAttn 源预训练完成)。详见 [EXPERIMENT_LINEAGE.md §3.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

### 核心目标

- 24obj 源预训练 → chromo 目标微调 ($k=5, k=10$), 验证扩散检测器低数据迁移能力
- 扩散检测器在小样本迁移上竞争力 (与 DINO/RTMDet-L 对比)
- 论文 §4.3.1 已提及: Dataset 1 (训练集较小, 1,540 张) 上 KaryoFlow (0.753 mAP) 实际超过 RTMDet-L (0.742) 和 DINO R50 (0.737), 提示扩散范式在低数据小目标情形下尤其具有竞争力

### 实验设计

- 14 个目标微调配置 (7 模型 × 2 k 设置), 全部位于 `experiments/configs/few_shot/target_finetune/`
- 数据: $k=5$ (112 图, 120 标注), $k=10$ (222 图, 240 标注)
- 目标微调数据集: Chromosome20240904 (chromo, 旧数据集)
- 源预训练数据集: 24_chromosomes_object (24obj)

#### 实验证明目的

验证扩散检测器在低数据迁移上的竞争力, 对照 anchor-based (Cascade R-CNN, YOLOX-S) / transformer-based (DINO R50) / single-shot (RTMDet-L) / FBM 变体 (SimpleGate, CrossAttn) / LDMDet SOTA

#### 14 个目标微调配置列表

- `cascade_rcnn_r50_k5.py` / `cascade_rcnn_r50_k10.py`
  -- 模型: Cascade R-CNN R50 (anchor-based baseline)
- `dino_r50_k5.py` / `dino_r50_k10.py`
  -- 模型: DINO R50 (transformer-based, 多尺度可变形注意力)
- `ldmdet_fbm_crossattn_k5.py` / `ldmdet_fbm_crossattn_k10.py`
  -- 模型: LDMDet + Cross-Attention FBM (源预训练 🔄 运行中)
- `ldmdet_fbm_simplgate_k5.py` / `ldmdet_fbm_simplgate_k10.py`
  -- 模型: LDMDet + SimpleGate FBM (源预训练已停止, mAP=0.677)
- `ldmdet_sota_k5.py` / `ldmdet_sota_k10.py`
  -- 模型: LDMDet SOTA (源预训练已完成)
- `rtmdet_l_k5.py` / `rtmdet_l_k10.py`
  -- 模型: RTMDet-L (CSPNeXt-L, single-shot)
- `yolox_s_k5.py` / `yolox_s_k10.py`
  -- 模型: YOLOX-S (CSPDarkNet-S, single-shot)

### 源预训练状态

7 个源预训练模型 (24obj 数据集), 状态如下:

- LDMDet SOTA = 已完成 (best @ epoch 26)
  -- 本地: `work_dirs/few_shot/source_pretrain_ldmdet_sota_24obj/`
  -- config: `experiments/configs/few_shot/source_pretrain/ldmdet_sota_24obj.py`
- LDMDet FBM SimpleGate = 0.677 ⛔ 已停止 (epoch 5)
  -- 本地: `work_dirs/few_shot/source_pretrain_ldmdet_fbm_simplgate_24obj/20260701_203410/`
  -- SwanLab: `few-shot-benchmark` / run_id=`hyuiam5m` (best mAP=0.6770 @ epoch 5)
- LDMDet FBM CrossAttn = 0.810 🔄 运行中 (epoch 10/150)
  -- 本地: `work_dirs/few_shot/source_pretrain_ldmdet_fbm_crossattn_24obj/20260701_174434/`
  -- SwanLab: `few-shot-benchmark` / run_id=`9hj8pe4a` (best mAP=0.8100 @ epoch 10)
  -- 进展: ep1=0.000 → ep6=0.752 → ep9=0.787 → ep10=0.810 (持续上升)
- Cascade R-CNN R50 = 已完成 (best @ epoch 72)
  -- 本地: `work_dirs/few_shot/source_pretrain_cascade_rcnn_r50_24obj/`
- DINO R50 = 已完成 (best @ epoch 102)
  -- 本地: `work_dirs/few_shot/source_pretrain_dino_r50_24obj/`
- RTMDet-L = 已完成 (best @ epoch 116)
  -- 本地: `work_dirs/few_shot/source_pretrain_rtmdet_l_24obj/`
- YOLOX-S = 已完成 (best @ epoch 200)
  -- 本地: `work_dirs/few_shot/source_pretrain_yolox_s_24obj/`

### 当前状态

⛔ 尚未启动, 等待 FBM CrossAttn 源预训练完成 (当前 epoch 10/150)。

- SwanLab: `few-shot-benchmark` (源预训练, 3 个实验); 目标微调 project 待配置
- ⚠ 5 个非 FBM 实验仅保留 best checkpoint, 无训练日志 (需加载 checkpoint 评估或查 SwanLab)

### 预期结果

- 扩散检测器在小样本迁移上竞争力 (与 DINO/RTMDet-L 对比)
- 验证论文 §4.3.1 "Dataset 1 上 KaryoFlow 超过 RTMDet-L 和 DINO R50" 的低数据优势
- 若 FBM CrossAttn 源预训练成功 (>0.85), 可作为 LDMDet 变体参与对比

## 二、跨数据集扩展 (最高级目标, 理论推导)

> 纯理论推导, ⛔ 无实验验证。对应论文 §6 结论与未来工作。

### 核心目标

- 证明方法可扩展到有相似特性的其他图像 (不一定要真跑训练)
- 验证 OT Diversity Collapse 现象的普遍性 claim
- 论文 §6 已提及: "这一画像在医学影像中反复出现: 组织病理学中的细胞检测、乳腺 X 光和视网膜成像中的病灶检测、微生物菌落计数"

### 理论推导

- **OT Diversity Collapse 的 a-priori 诊断** (Table 2): 任何 $d \ll 100$ 且 $K \gg 10$ 的任务是候选
  -- 图像生成 ($d \sim 10^5$, $K=$ batch): $\Delta H / H \approx 0$ (OT 坍缩可忽略)
  -- 检测 COCO ($d=4$, $K \sim 7$): $\Delta H / H \approx 0.55$
  -- 检测 染色体 ($d=4$, $K \sim 46$): $\Delta H / H \approx 0.69$
- **x0/v Prediction 设置依赖性**: 低维 + shifted schedule → x0-prediction, 可指导其他低维任务参数化选择
- **Stochastic Coupling 双重收益**: 低数据情形下的精度 + 一般情形下的稳定性 ($4.6\times$ epoch-std 减少)

### 候选场景

- 细胞检测 (组织病理学): 多核 / $d=4$ / 小标注队列 → 最自然的下一步
- 病灶检测 (乳腺 X 光, 视网膜成像): 小目标 / 有限阳性案例
- 微生物菌落计数: 密集目标 / $d=4$
- 遥感密集检测: 高 $K$ / 低 $d$ 结构化预测

### 当前状态

- 纯理论推导, 无实验验证
- 论文 §6 结论已提及候选场景 (细胞检测 / 病灶检测 / 菌落计数)
- 论文承认局限: "经验验证限于染色体数据; 在 COCO 或细胞检测基准上验证将检验 OT 坍缩预测的普遍性"

### 未来工作

- 在至少一个非染色体高 $K$ 低 $d$ 基准上验证 (细胞检测是最自然的下一步)
- 将大幅强化普遍性 claim
- 理论通过 Table 2 提供 a-priori 诊断: 严重性 $\Delta H/H$ 预测 Stochastic Coupling 是否会有帮助

## 三、速度引导自适应 Renewal

> 已集成到 head.py, 待系统评估。config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_vgar_24obj.py`

### 核心设想

- 用 $\alpha(t) \cdot \hat{x}_0 + (1-\alpha(t)) \cdot z$ 替代纯噪声 renewal
- 数学公式: $\alpha(t) = 0.2 + 0.6 \cdot \text{sigmoid}(5 \cdot (0.5 - t))$
- $x_{\text{renewed}} = \alpha(t) \cdot x_0\_{\text{pred}} + (1-\alpha(t)) \cdot \text{randn}$
- 理论依据: $v_\theta$ 的 $x_0\_{\text{pred}}$ 编码了 "框应该去哪里", 纯随机 renewal 浪费了这个信息
- $\alpha$ 随时间步自适应: 早期更随机 (探索), 后期更确定 (利用)

### 当前状态

- 已集成到 `head.py` (代码就绪)
- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_vgar_24obj.py` (基于 `a4_dpm_pp_24obj.py`)
- 改动: `velocity_guided_renewal=True`
- SwanLab: `ldmdet-mainline-ablation-24obj` / `a4_vgar` (project 已配置)
- 基线: +DPM-Solver++ (mAP=0.863)

### 与 Box Renewal × Solver 交互关系

- 速度引导自适应 Renewal 缓解 Box Renewal × Solver 矛盾 (保留部分 $\hat{x}_0$ 信息), 但未完全消除
- $\alpha(t)$ 在 $t \to 0$ 时 $\to 0.8$ (`head.py:177-179`), 仍保留 20% 随机性
- Box Renewal × Solver 矛盾仅缓解未消除: 下一步 $\hat{x}_0^{(n+1)}$ 与 $\hat{x}_0^{(n)}$ 有部分相关性, 但非完全轨迹连续
- 理论预期: VGAR 下 $\eta_{\text{str}}$ 应介于 baseline (renewal on) 和 renewal off 之间

### 待做

- 系统评估速度引导自适应 Renewal 对 $\eta_{\text{str}}$ 和 mAP 的影响
- 与 Box Renewal × Solver 交互方案 A/B/C 对比, 验证速度引导自适应 Renewal 是否为更优修复方案
- 3 seeds (42/123/789) 重训, 与 +DPM-Solver++ baseline 对照
- 若速度引导自适应 Renewal 显著改善 $\eta_{\text{str}}$ 且 mAP 不退化, 可作为论文新方向纳入

## 四、Brenier 映射神经化 (突破方向, 纯理论)

> ⛔ 未开展 (纯理论)。基于 doubao AI 建议 + 最优传输理论。TMI 投稿后考虑。
> 核心设想: 用 ICNN 参数化 Brenier 势 φ, T*(z) = ∇φ(z) 直接给出从噪声到 bbox 的最优传输映射

### 核心理论

- **Brenier 定理**: 在平方距离代价下, 最优传输映射 T* = ∇φ 存在且唯一 (φ 为凸函数)
- **ICNN (Makkuva et al., 2020)**: 提供参数化凸函数的方法, 通过非负权重 + 单调激活保证凸性
- **4D bbox 空间**: 输入维度 d=4 (低维), ICNN 计算成本可控

### 与当前架构的冲突

- ⚠ **架构冲突**: 当前 cascade head (RoIAlign + DynamicConv) vs ICNN 参数化 φ, 架构完全不同
- ⚠ **训练范式冲突**: flow matching 损失 vs Wasserstein 距离损失, 需重训
- ⚠ **推理范式冲突**: DPM-Solver++ 多步积分 vs ∇φ 一步映射, 无法复用 solver 框架

### 分阶段计划

- **E.1 理论分析** (纯理论, ~2 天): 推导 4D Brenier 势形式, 分析 ICNN 表达能力, 纳入论文 §6 未来工作 (~0.2 页)
- **E.2 ICNN 原型** (代码, ~3 天): 实现 4D ICNN + ∇φ 自动微分 + Sinkhorn 损失, 单元测试凸性
- **E.3 训练验证** (高成本, ~4.5 天): ICNN 替代 cascade head, 3 seeds × 150 epochs

### 优先级

- **低** (突破性方向, 风险高, TMI 10 页限制下难以纳入)
- **建议**: 仅做 E.1 (理论分析), 纳入论文 §6 未来工作, 不做实验

### doubao 方向对照 (A/B/C/D/E)

| doubao 方向 | 核心建议 | 当前论文对应方向 | 当前状态 |
|------------|---------|----------------|---------|
| A. 检测专用 RF-DPM 联合推导 | per-dim solver | per-dim η_str 诊断 | ✓ 完成 (→ LINEAGE §九) |
| B. 速度感知网络结构 | 直接预测 v 而非 x0 | x0/v Prediction 对照 | ⚠ 单 seed 初步 (→ LINEAGE §八, seed 123/789 待补) |
| C. 时间条件深度融合 | step-aware embedding | step-aware embedding | ✓ 完成 (→ FALSIFIED §二十五) |
| D. 自适应阶次 DPM-Solver++ | t 大用低阶, t 小用高阶 | 自适应阶次 DPM-Solver++ | ✓ 完成 (→ LINEAGE §十) |
| E. Brenier 映射神经化 | ICNN 参数化 Brenier 势 | Brenier 映射神经化 (本节) | ⛔ 未开展 |

> ⚠ **关键发现**: doubao 方向 D 假设"t 小时需要高阶修正", 但实际诊断显示 η_3rd 在后期 step (t 小) 显著降低 (44.6→18.2, 降幅 59%), 即实际趋势与 doubao 假设相反。

---

## 五、级联头角色分化 — ⛔ 待启动 (中-高优先级)

> 详细方案: [STRUCTURAL_IMPROVEMENT_ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md)

### 设计

- 方案 A (损失权重衰减): 零代码改动, 仅改 criterion 配置
- Box Renewal × Solver 诊断支持: head0 修正最大 (reg std 0.94), 后级递减, 等权 deep_supervision 可能非最优
- 训练: 从 +DPM-Solver++ checkpoint 微调, 对比等权 vs 衰减

### 形态感知 RoI 编码器 v2 (M1-v2 改进方向, ⛔ 待启动)

M1 FP32 null result 已归档至 [FALSIFIED §二十六](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (best 0.862@ep19, Δ=-0.001 持平 +DPM-Solver++, h_conv/v_conv 均匀 → 设计问题)。

- **M1-v2 改进方向**:
  - 非零初始化 fuse (如小常数初始化 0.01, 打破梯度瓶颈)
  - 显式形态先验注入 (臂长比/面积作为输入, 而非依赖卷积发现)
  - 注意力机制替代方向卷积 (self-attention 自然捕获空间关系)

---

## 六、LVD-RF (Lyapunov Velocity Direction Regularization) — ⛔ 待启动 (优先级 1)

> R2 评审通过 (A↔B 两轮交互), FINAL 整合文档: [FEASIBLE_LVD_RF.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/FEASIBLE_LVD_RF.md)
> R2 评分: 7.5/10 (推荐, 有条件)

### 核心目标

通过在训练损失中增加**速度方向余弦正则项**, 约束速度场 $v_\theta$ 的方向一致性, 降低 $\eta_{\text{str}}$ (直线度诊断指标), 进而减少 DPM-Solver++ 截断误差。理论根基为 per-proposal 条件 Lyapunov 稳定性 (修订后), 并由 ProReflow (CVPR 2025) 的 "方向优于幅度" 实验结论支持。

核心优势: **零额外前向** (训练开销 <2%), 工程可行性最高; 速度方向正则轴与 TFR (幅度) / ISLR (雅可比) 正交, 可独立贡献收益。是 4 个 R2 通过方向中**工程可行性最高**的一个, 适合作为首选改进方向。

### 设计摘要

- **核心改动**: 在训练损失中增加速度方向余弦正则项
  $$L_{\text{lvd}} = \lambda \cdot \mathbb{E}\left[1 - \cos\left(v_\theta(x_t, t),\ v_\theta(x_t', t')\right)\right]$$
  默认采用 $\sin^2$ 形式 (梯度强 2×, 缓解梯度消失): $L_{\text{lvd}} = \lambda \cdot \mathbb{E}[\sin^2\alpha]$
- **理论支撑**:
  - 定理 2.3' (条件 Lyapunov 稳定性): 在假设 (A1) $\|v_\theta\| \leq V_{\max}$, (A2) 幅度 Lipschitz $L_v^{\text{mag}}$, (A3) $\hat{x}_0$ 非退化下, $\eta_{\text{str}} = O(\sqrt{\epsilon})$
  - ProReflow (CVPR 2025, arXiv:2503.04824) 实验确认 "方向优于幅度" (B 的 R1 对发表状态判断有误, WebSearch 三方证实: OpenReview + IEEE Xplore + 华东师大教师页面)
- **与已有方向正交**: 速度方向正则轴独特, 与 TFR (幅度正则) / ISLR (雅可比正则) 正交, 可独立或叠加使用
- **梯度消失缓解**: $\sin^2$ 默认 (cos_sim=0.99 时仍有 28% 梯度) + sqrt 自适应切换 fallback (cos_sim > 0.99 持续 1000 iter 触发)

### 实验计划

- **Phase 0 诊断** (0.5-1 天):
  - 验证 $\eta_{\text{str}}$ 行为 (确认 baseline 直线度问题)
  - 验证 ProReflow "方向优于幅度" 假设在检测场景的迁移性
  - 实测 $\|v_\theta\|(t)$ 曲线, 验证假设 (A1)(A2) 是否成立 (Blocking: 若不成立需补充幅度正则)
- **Phase 1 单 seed 验证** (3 天):
  - $\sin^2$ 形式默认, $\lambda \in \{0.001, 0.003, 0.01, 0.03, 0.1\}$ 网格搜索
  - 监控 cos_sim 分布, 若 cos_sim > 0.99 持续 1000 iter 触发自适应切换 sqrt 形式
  - xyxy 有效性检查 (x2 > x1, y2 > y1), 无效框跳过
- **Phase 2 3-seed 验证** (5 天 × 3 seeds):
  - 3 seeds (42/123/789), 最优 $\lambda^*$ 重训
  - 关键消融: LVD-2 (方向) vs TFR-bl (幅度) 对比, 验证方向正则跨域迁移性
  - 3-way 对比: $\sin^2$ 全程 vs sqrt 全程 vs 自动切换

### 预期增益

- mAP: +0.003~0.008 (Dataset 2)
- Y 染色体 AP: +0.005~0.015 (方向正则对稀有类更显著)
- $\eta_{\text{str}}$: 条件性 $O(\sqrt{\epsilon})$ 下降 (依赖假设 A1/A2 成立)
- 训练开销: **<2%** (零额外前向, 工程优势显著)

### 风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| 条件性保证传递性 (A1/A2 不由 LVD-RF 保证) | 高 | Phase 1 实测 $\|v_\theta\|(t)$, 若不成立补充幅度正则 (与 TFR/VCR 叠加) |
| 梯度消失 (cos_sim → 1 时) | 高 | $\sin^2$ 默认 (梯度 2×) + sqrt 自适应切换 fallback |
| ProReflow 跨域迁移性未验证 (图像生成 → 检测) | 中 | Phase 2 LVD-2 vs TFR-bl 关键消融 |
| per-proposal 独立性假设不完全成立 (共享 backbone) | 低 | 条件独立性仍支持分析, 重叠染色体干扰为二阶效应 |

### SwanLab & 配置

- SwanLab Project: (待创建) `ldmdet-mainline-ablation-24obj`
- 配置: `experiments/configs/ldmdet/mainline_24obj/lvd_rf_24obj.py` (待创建)

---

## 七、TRIP (Tikhonov-Morozov 反问题正则化) — ⛔ 待启动 (优先级 2)

> R2 评审通过 (A↔B 两轮交互), FINAL 整合文档: [FEASIBLE_TRIP.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/FEASIBLE_TRIP.md)
> R2 评分: 7.5/10 (谨慎推荐)

### 核心目标

通过**替换回归目标** (而非加正则项), 将训练目标从 GT $x_0$ 改为贝叶斯 MAP 收缩解 $\tilde{x}_0(t)$, 显式建模 SNR 退化 (大 $t$ 段信号被噪声淹没)。理论框架为 "SNR 退化 + 贝叶斯 MAP", $\lambda = t^2$ 来自噪声方差/先验方差比 (Tarantola 2005 标准)。

核心优势: **最独特** (目标替换而非加正则项, 与所有损失项方案代数正交); 贝叶斯 MAP 框架数学严格成立; $d=4$ + $K \approx 46$ + 24 类全匹配。**但**: 需 Phase 1 验证 baseline 是否已 MMSE 最优 (Blocking 前提); 新颖性较原 "不适定反问题 + Morozov" 框架下降 (与 label smoothing 概念相近)。

### 设计摘要

- **核心改动**: 替换回归目标
  $$x_0 \to \tilde{x}_0(t) = (1 - s(t))\,x_0 + s(t)\,\mu_p$$
  其中 $s_{\text{MAP}}(t) = \frac{t^2}{(1-t)^2 \sigma_p^2 + t^2} \in [0, 1]$ 为贝叶斯 MAP 收缩系数
- **理论支撑**:
  - 引理 2.2 (SNR 退化): $\text{SNR}(t) = (1-t)^2 \sigma_p^2 / t^2 \to 0$ as $t \to 1$
  - 贝叶斯 MAP 解: $x_0^{\text{MAP}}(t) = [(1-t)^2 I + t^2 \Sigma_p^{-1}]^{-1}[(1-t) x_t + t^2 \Sigma_p^{-1} \mu_p]$
  - $\lambda = t^2$ 是数学导出 (非超参), 来自噪声方差/先验方差比
  - 期望形式边界: $s_{\text{MAP}}(0) = 0$ (目标=GT), $s_{\text{MAP}}(1) = 1$ (目标=$\mu_p$); 小 $t$ 段 $s \approx t^2/\sigma_p^2 = O(t^2)$ (比 Morozov 的 $O(t)$ 更快衰减, 对 mAP 末 step 精度更友好)
- **与已有方向正交**: 目标替换 vs 损失项增加, 与 TFR/BEAR/LVD-RF/ISLR-RF 代数正交
- **Morozov 备选**: 仅作经验调参旋钮 (可能在 小数据集 上更稳健, 但无理论保证), 主方案必须用 MAP ($\lambda = t^2$)

### 实验计划

- **Phase 0 诊断** (0.5-1 天):
  - **E1.5 (Blocking)**: baseline MMSE 最优性验证 — 测量大 $t$ 段 ($t \in [0.7, 1.0]$) 网络输出是否 ≈ $\mu_p$ (MMSE 最优)。若已 MMSE 最优, TRIP 收益有限, 需重新定位或放弃
  - E1.2: 验证 $\eta_{\text{str}}$ 在大 $t$ 段行为 (是否 "GT 记忆抖动"), 校准 $\eta_{\text{str}}$ 下降预期
- **Phase 1 单 seed 验证** (3 天):
  - 主方案 MAP ($\lambda = t^2$), 类条件先验 $\mu_p^c$ 估计
  - 关键消融: 类条件 vs 全局先验 (验证类条件对 Y AP 的增益)
  - 修正 (2.5) 符号 typo: $\|(1-t)(x-x_t)\|^2 \to \|(1-t)x - x_t\|^2$
  - 向量化 `_compute_trip_target` (避免 `for b, c` Python 循环)
- **Phase 2 3-seed 验证** (5 天 × 3 seeds):
  - 3 seeds (42/123/789), 最优配置重训
  - 关键消融: TRIP ($t$-自适应 $s(t)$) vs 固定 $\alpha$ label smoothing, 验证 $t$-自适应性收益
  - MAP vs Morozov 对比 (E2.3)

### 预期增益

- mAP: +0.002~0.008 (Dataset 2, **需 Phase 1 E1.5 验证 baseline 非 MMSE 最优**)
- Y 染色体 AP: +0.002~0.010
- $\eta_{\text{str}}$: -10~25%
- 训练开销: **~5%** (类条件先验估计, 无额外前向)

### 风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| Baseline 已 MMSE 最优 → TRIP 收益有限 | 高 (Blocking) | Phase 1 E1.5 优先验证, 若成立需重新定位为 "训练稳定加速" |
| 期望形式与 label smoothing 概念相近 (新颖性下降) | 中 | Phase 2 消融 TRIP vs 固定 $\alpha$ label smoothing, 量化 $t$-自适应性收益 |
| 大 $t$ 段训练信号冲突 (类条件目标 vs 类无关输入) | 中 | 影响有限 (差异是 $\mu_p^c - \mu_p$ 类偏移, 损失中为常数项, 不影响梯度方向) |
| Morozov 备选方案理论地位不清 | 低 | 明确 Morozov 仅作经验调参旋钮, 主方案必须用 MAP ($\lambda = t^2$) |

### SwanLab & 配置

- SwanLab Project: (待创建) `ldmdet-mainline-ablation-24obj`
- 配置: `experiments/configs/ldmdet/mainline_24obj/trip_24obj.py` (待创建)

---

## 八、BEAR (Backward-Error-Aware Regularization) — ⛔ 待启动 (优先级 3)

> R2 评审通过 (A↔B 两轮交互), FINAL 整合文档: [FEASIBLE_BEAR.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/FEASIBLE_BEAR.md)
> R2 评分: 7.3/10 (谨慎推荐)

### 核心目标

通过在训练损失中增加**二阶均差正则项**, 显式惩罚 DPM-Solver++ 截断误差的二阶分量。理论根基为修正方程 (Modified Equation) 框架, 直接连接多步法反向误差分析 (BEA, Hairer & Wanner Ch. XV Theorem XV.3.1), 是 TFR (一阶均差正则) 的严格推广。

核心优势: 叙事优雅 (TFR 严格推广, 修正方程框架直接连接 solver 截断误差); $\psi$ 闭式表达式数学验证正确; $\phi_2$ 映射修正完全正确。**但**: 训练开销 +60~70% (3 次前向); 与 EXER-RF 数学对象相同 (冗余性未根本解决); "TFR 严格推广" 叙事价值在联合使用时减弱 (退化为强制常数)。

### 设计摘要

- **核心改动**: 在训练损失中增加二阶均差正则项
  $$L_{\text{bear}} = \lambda \cdot \mathbb{E}\left[|\psi(t)| \cdot \|D_2\|^2\right]$$
  其中 $D_2$ 是三步二阶均差 (需 3 次前向: $v_\theta(x_{t_0}, t_0)$, $v_\theta(x_{t_1}, t_1)$, $v_\theta(x_{t_2}, t_2)$)
- **权重 $\psi$ 闭式表达式** (R2 数学验证正确):
  $$\psi = \frac{t_{n+1}^2 - t_n^2}{2} - 2t_n(t_{n+1} - t_n) + t_n^2 \ln\frac{t_{n+1}}{t_n}$$
  (数值验证: $t_n=0.5, t_{n+1}=0.25$ 时 $\psi = -0.0170$, 与代码 `rectified_flow.py` L215-217 一致)
- **理论支撑**:
  - 修正方程框架: DPM-Solver++ 截断误差 $\propto \|D_2\|$
  - 定理 2.1 (BEA, Hairer & Wanner Ch. XV): 多步法反向误差分析
  - 假设 H1: 沿任意轨迹的 $D_2$ Lipschitz 性 (经验假设, 需 Phase 0 验证, 判据 $\rho > 0.7$)
- **与 TFR 关系**: TFR 是 BEAR 的特例 (一阶 vs 二阶均差), 联合使用退化为强制 $\hat{x}_0$ 常数
- **MEC-RF fallback**: 当 BEAR 3× 开销不可接受时, 使用 MEC-RF (1.3× JVP) 作为高效替代
- **空间一致性**: 在 raw cxcywh 空间计算 $D_2$ (与 solver 的 `x0_history` 一致)

### 实验计划

- **Phase 0 诊断** (0.5-1 天):
  - Lipschitz 假设 H1 验证: 测量 $\rho = \text{corr}(D_2^{\text{train}}, D_2^{\text{infer}})$, 判据 $\rho > 0.7$
  - $\psi$ 闭式表达式数值验证 (与代码 `rectified_flow.py` L215-217 一致)
  - $C_1 \approx 0.04$ 数值估计确认 ($C_2$ 仍依赖 Lipschitz 常数 $L$, 需诊断)
- **Phase 1 单 seed 验证** (3 天):
  - $\lambda \in \{0.001, 0.003, 0.01, 0.03, 0.1\}$ 网格搜索
  - 在 raw cxcywh 空间计算 $D_2$ (与 solver 的 `x0_history` 一致)
- **Phase 2 3-seed 验证** (5 天 × 3 seeds):
  - 3 seeds (42/123/789), 最优 $\lambda^*$ 重训
  - 关键消融: $|\psi|$ (时间步依赖) vs $\lambda_{\text{ext}}=7$ (常数, EXER-RF) vs 均匀权重
  - BEAR vs MEC-RF (1.3× JVP fallback) 收益/开销权衡

### 预期增益

- mAP: +0.003~0.008 (Dataset 2)
- $\eta_{\text{3rd}}$ (三阶直线度): 显著下降 (BEAR 主要跟踪 $\eta_{\text{3rd}}$)
- 训练开销: **+60~70%** (3 次前向, 完整二阶导数)

### 风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| 训练开销高 (+60~70%, 3 次前向) | 高 | MEC-RF fallback (1.3× JVP), 当 3× 开销不可接受时使用 |
| 与 EXER-RF 数学对象相同 (冗余性) | 中 | Phase 2 消融 $|\psi|$ vs $\lambda_{\text{ext}}$ vs 均匀, 量化权重设计差异 |
| TFR 严格推广叙事价值减弱 (联合使用退化为 TFR) | 中 | BEAR 独立价值在 "单独使用时比 TFR 更精细" |
| 训练-推理轨迹一致性 gap | 中 | 假设 H1 (Lipschitz), Phase 0 验证 $\rho > 0.7$ |
| $C_2$ 常数未显式 | 低 | Phase 0 诊断估计 |

### SwanLab & 配置

- SwanLab Project: (待创建) `ldmdet-mainline-ablation-24obj`
- 配置: `experiments/configs/ldmdet/mainline_24obj/bear_24obj.py` (待创建)

---

## 九、ISLR-RF (Input-Space Lipschitz Regularization) — ⛔ 待启动 (优先级 4)

> R2 评审通过 (A↔B 两轮交互), FINAL 整合文档: [FEASIBLE_ISLR_RF.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/FEASIBLE_ISLR_RF.md)
> R2 评分: 7.4/10 (谨慎推荐)

### 核心目标

通过在训练损失中增加**输入雅可比 Frobenius 正则项**, 约束网络 $f$ 对输入 $x_t$ 的 Lipschitz 常数, 降低泛化误差。理论根基为 Sokolić (2017, IEEE TSP) 的 Rademacher 复杂度界 $\hat{\mathcal{R}}_n(\mathcal{F}) \leq O(B_x \sqrt{M}/\sqrt{n})$, 其中 $M = \mathbb{E}[\|J_f\|_F^2]$ (WebSearch 验证: Sokolić 真实存在, Wu & Li 2024 arXiv:2412.12449 明确建立 Jacobian 正则化与 Rademacher 复杂度的界)。

核心优势: 理论核心正确 ($J_f \to 0$ 与理想 RF 一致); Sokolić Rademacher 界经验证正确; $d=4$ 甜区论证合理 (Finlay-Oberman 2019 ICLR 2020 工程先例)。**但**: 训练开销 +80~200% (4 VJP 二阶微分), 实施最难; 超参数 proliferation ($\lambda_0, p$ 两个超参); Rademacher 界线性收紧 (泛化 gap 改善 ~29% 而非 50%)。

### 设计摘要

- **核心改动**: 在训练损失中增加输入雅可比 Frobenius 正则项
  $$L_{\text{islr}} = \lambda \cdot \mathbb{E}\left[t^p \cdot \|J_f\|_F^2\right]$$
  其中 $J_f = \partial f_\theta(x_t, t) / \partial x_t$ 是网络对输入 $x_t$ 的雅可比, $t^p$ 为 $t$-dependent 加权 ($p=2$ 默认)
- **理论支撑**:
  - Sokolić (2017) Rademacher 复杂度界: $\hat{\mathcal{R}}_n(\mathcal{F}) \leq O(B_x \sqrt{M}/\sqrt{n})$, $M = \mathbb{E}[\|J_f\|_F^2]$ ($\sqrt{M}$ 正确, 因 Lipschitz 常数 $\leq \sqrt{M}$)
  - 命题 1.8' (修正): 理想 $J_f$ 在 $t \approx 0$ 时为 $I$ (网络已知 GT), 在 $t \approx 1$ 时为 $0$ (网络从 $x_t$ 推断)
  - $t$-dependent 加权: $\lambda(t) = \lambda_0 \cdot t^p$, $t$ 大 (噪声端) 强正则, $t$ 小 (数据端) 弱正则
  - Grönwall 指数改善 ~31% (对应 seed std 下降 ~15%), $t \in [0.25, 1]$ 范围
- **$K_{\text{islr}}=16$ 子采样**: 从 $K \approx 46$ 中子采样 16 个 proposals (比例 ~35%), 方差降低 $\sqrt{16} = 4\times$ (原 $\sqrt{46} \approx 6.8\times$ 削弱)
- **Fallback**: Finlay-Oberman (2019, ICLR 2020) 或 Hutchinson 1-sample 估计 (方差 $O(1/d)$ 量级, 取决于 $J$ 的奇异值分布)

### 实验计划

- **Phase 0 诊断** (0.5-1 天):
  - $K_{\text{islr}}=16$ 子采样微基准: 验证 4 VJP 开销 (+80~200%) 和方差降低 (4×)
  - $t$-dependent 加权验证: 确认 $\lambda(t) = \lambda_0 t^p$ 在 $t \in [0.25, 1]$ 的作用范围 (注意 $t=0.25$ 时 $\lambda = 0.0625\lambda_0$, 几乎不正则)
  - Grönwall 指数改善 ~31% 数值确认
- **Phase 1 单 seed 验证** (3 天):
  - 固定 $p=2$, 仅调 $\lambda_0 \in \{0.001, 0.003, 0.01, 0.03, 0.1\}$ (避免 2D 网格搜索)
  - 监控 $\|J_f\|_F$ 分布, 验证过度正则化风险
- **Phase 2 3-seed 验证** (5 天 × 3 seeds):
  - 3 seeds (42/123/789), 最优 $\lambda_0^*$ 重训
  - 在最优 $\lambda_0^*$ 附近做 $p$ 的一维消融 ($p \in \{1, 2, 4\}$)
  - 关键消融: 4 VJP (全量) vs Hutchinson (1-sample) vs Finlay-Oberman fallback

### 预期增益

- mAP: +0.002~0.006 (Dataset 2)
- seed std: -10~20% (Grönwall 指数改善 ~31%)
- 训练开销: **+80~200%** (4 VJP + 额外反传, 实施最难)

### 风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| 训练开销极高 (+80~200%, 4 VJP) | 高 | $K_{\text{islr}}=16$ 子采样 + Hutchinson fallback (1-sample, 方差 $O(1/d)$) |
| 超参数 proliferation ($\lambda_0, p$ 两个超参) | 中 | Phase 1 固定 $p=2$ 仅调 $\lambda_0$; Phase 2 在最优 $\lambda_0^*$ 附近做 $p$ 一维消融 |
| 过度正则化 ($J_f \to 0$ 在 $t \approx 0$ 不期望) | 中 | $t$-dependent 加权 $\lambda(t) = \lambda_0 t^p$, $t$ 小弱正则 |
| Rademacher 界线性 vs Bartlett 平方 (收紧效应弱) | 低 | 诚实承认泛化 gap 改善 ~29% (非 50%), 小数据集收益论点削弱 |
| $K_{\text{islr}}=16$ 削弱方差降低论证 (4× vs 6.8×) | 低 | 4× 方差降低仍有效, 更新 §4.2 论证 |

### SwanLab & 配置

- SwanLab Project: (待创建) `ldmdet-mainline-ablation-24obj`
- 配置: `experiments/configs/ldmdet/mainline_24obj/islr_rf_24obj.py` (待创建)

---

## 附: SwanLab Project 映射 (待做方向相关)

| SwanLab Project | 方向 | 状态 | URL Pattern |
|-----------------|------|------|-------------|
| `few-shot-benchmark` | Few-Shot 源预训练 | 🔄 1 中断 (best 0.857@ep45), 6 已完成 | `https://swanlab.cn/@einspanner/few-shot-benchmark/runs/<run_id>` |
| `ldmdet-mainline-ablation-24obj` | 速度引导自适应 Renewal (a4_vgar) | ⛔ 待启动 | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| (待创建) | 级联头角色分化 | ⛔ 待启动 | 详见 [STRUCTURAL_IMPROVEMENT_ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md) |
| (待创建) `ldmdet-mainline-ablation-24obj` | LVD-RF (Lyapunov 速度方向正则, 优先级 1) | ⛔ 待启动 (R2 通过 7.5/10) | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| (待创建) `ldmdet-mainline-ablation-24obj` | TRIP (Tikhonov-Morozov 反问题正则, 优先级 2) | ⛔ 待启动 (R2 通过 7.5/10) | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| (待创建) `ldmdet-mainline-ablation-24obj` | BEAR (反向误差感知正则, 优先级 3) | ⛔ 待启动 (R2 通过 7.3/10) | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| (待创建) `ldmdet-mainline-ablation-24obj` | ISLR-RF (输入空间 Lipschitz 正则, 优先级 4) | ⛔ 待启动 (R2 通过 7.4/10) | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |

> SwanLab 用户名: `einspanner` (登录态见 `/home/linkst/.swanlab/.netrc`, api_key 已配置)
> 目标微调 project (Few-Shot 14 个配置) 待 FBM CrossAttn 源预训练完成后配置

<!-- 文档结束。
     更新策略: 当方向状态变化 (如训练启动 / 完成 / 证伪), 更新对应章节的 ⛔/🔄/✓/🔴 标记和 SwanLab run_id。
     方向完成后: 有效→迁入 EXPERIMENT_LINEAGE.md; 证伪→迁入 FALSIFIED_DIRECTIONS.md; 本文档仅保留 🔄进行中 + ⛔待启动。
     2026-07-28 更新: R1/R2 评审循环完成, 4 方向通过 (LVD-RF/TRIP/BEAR/ISLR-RF, R2 评分 7.3~7.5/10), 10 方向淘汰→FALSIFIED §十五~§二十二; 新增 §六~§九。
     2026-07-27 更新: 删除已归档方向 S1 (→LINEAGE §七), 方向 A (→LINEAGE §九), 方向 C (→FALSIFIED §二十五), 方向 D (→LINEAGE §十), Head Distillation (→LINEAGE §七); ReFlow 重试确认方法本质失败 → FALSIFIED §十四。 -->
