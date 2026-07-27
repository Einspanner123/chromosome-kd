# 待做研究方向总览 (TODO Directions)

> 本文档梳理 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 所有进行中或待启动的研究方向。
> 这些方向部分有代码就绪、配置就绪或实验已在运行, 部分仅有理论框架。
> 每个方向附 **可靠数据源地址** (本地服务器路径 / SwanLab project / config 路径)。
> 更新时间: 2026-07-28 (ReFlow 重试确认方法本质失败 → FALSIFIED §十四, 从本文档移除; 2026-07-27 校验+归档: R3 3-seed完成→LINEAGE §八, D3→LINEAGE §六, D1→LINEAGE §十五, M1→LINEAGE §十五, SC-RF→LINEAGE §十五; 全部代号替换为描述性名称; Few-Shot FBM CrossAttn 中断@ep59)
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
- 基线: A4 DPM-Solver++ (mAP=0.863)

### 与 Box Renewal × Solver 交互关系

- 速度引导自适应 Renewal 缓解 Box Renewal × Solver 矛盾 (保留部分 $\hat{x}_0$ 信息), 但未完全消除
- $\alpha(t)$ 在 $t \to 0$ 时 $\to 0.8$ (`head.py:177-179`), 仍保留 20% 随机性
- Box Renewal × Solver 矛盾仅缓解未消除: 下一步 $\hat{x}_0^{(n+1)}$ 与 $\hat{x}_0^{(n)}$ 有部分相关性, 但非完全轨迹连续
- 理论预期: VGAR 下 $\eta_{\text{str}}$ 应介于 baseline (renewal on) 和 renewal off 之间

### 待做

- 系统评估速度引导自适应 Renewal 对 $\eta_{\text{str}}$ 和 mAP 的影响
- 与 Box Renewal × Solver 交互方案 A/B/C 对比, 验证速度引导自适应 Renewal 是否为更优修复方案
- 3 seeds (42/123/789) 重训, 与 A4 baseline 对照
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
| B. 速度感知网络结构 | 直接预测 v 而非 x0 | x0/v Prediction 对照 | ✓ 3-seed 完成 (→ LINEAGE §八) |
| C. 时间条件深度融合 | step-aware embedding | step-aware embedding | ✓ 完成 (→ LINEAGE §十一) |
| D. 自适应阶次 DPM-Solver++ | t 大用低阶, t 小用高阶 | 自适应阶次 DPM-Solver++ | ✓ 完成 (→ LINEAGE §十) |
| E. Brenier 映射神经化 | ICNN 参数化 Brenier 势 | Brenier 映射神经化 (本节) | ⛔ 未开展 |

> ⚠ **关键发现**: doubao 方向 D 假设"t 小时需要高阶修正", 但实际诊断显示 η_3rd 在后期 step (t 小) 显著降低 (44.6→18.2, 降幅 59%), 即实际趋势与 doubao 假设相反。

---

## 五、级联头角色分化 — ⛔ 待启动 (中-高优先级)

> 详细方案: [STRUCTURAL_IMPROVEMENT_ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md)

### 设计

- 方案 A (损失权重衰减): 零代码改动, 仅改 criterion 配置
- Box Renewal × Solver 诊断支持: head0 修正最大 (reg std 0.94), 后级递减, 等权 deep_supervision 可能非最优
- 训练: 从 A4 checkpoint 微调, 对比等权 vs 衰减

### 形态感知 RoI 编码器 v2 (M1-v2 改进方向, ⛔ 待启动)

M1 FP32 null result 已归档至 [LINEAGE §十五](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md) (best 0.862@ep19, Δ=-0.001 持平 A4, h_conv/v_conv 均匀 → 设计问题)。

- **M1-v2 改进方向**:
  - 非零初始化 fuse (如小常数初始化 0.01, 打破梯度瓶颈)
  - 显式形态先验注入 (臂长比/面积作为输入, 而非依赖卷积发现)
  - 注意力机制替代方向卷积 (self-attention 自然捕获空间关系)

---
## 附: SwanLab Project 映射 (待做方向相关)

| SwanLab Project | 方向 | 状态 | URL Pattern |
|-----------------|------|------|-------------|
| `few-shot-benchmark` | Few-Shot 源预训练 | 🔄 1 中断 (best 0.857@ep45), 6 已完成 | `https://swanlab.cn/@einspanner/few-shot-benchmark/runs/<run_id>` |
| `ldmdet-mainline-ablation-24obj` | 速度引导自适应 Renewal (a4_vgar) | ⛔ 待启动 | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| (待创建) | 级联头角色分化 | ⛔ 待启动 | 详见 [STRUCTURAL_IMPROVEMENT_ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md) |

> SwanLab 用户名: `einspanner` (登录态见 `/home/linkst/.swanlab/.netrc`, api_key 已配置)
> 目标微调 project (Few-Shot 14 个配置) 待 FBM CrossAttn 源预训练完成后配置

<!-- 文档结束。
     更新策略: 当方向状态变化 (如训练启动 / 完成 / 证伪), 更新对应章节的 ⛔/🔄/✓/🔴 标记和 SwanLab run_id。
     方向完成后: 有效→迁入 EXPERIMENT_LINEAGE.md; 证伪→迁入 FALSIFIED_DIRECTIONS.md; 本文档仅保留 🔄进行中 + ⛔待启动。
     2026-07-27 更新: 删除已归档方向 S1 (→LINEAGE §七), 方向 A (→LINEAGE §九), 方向 C (→LINEAGE §十一), 方向 D (→LINEAGE §十), Head Distillation (→LINEAGE §十五); ReFlow 重试确认方法本质失败 → FALSIFIED §十四。 -->
