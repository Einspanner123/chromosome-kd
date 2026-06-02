# LDMDet 投稿可行性评估与补充路线

> 基准日: 2026-05-28 | 严格基于事实，无主观先验判断
> 硬件: A6000 48GB + A5000 24GB

---

## 一、理论逻辑性评估

### 1.1 严格的数学命题（逻辑自洽，无已知缺陷）

| 命题 | 数学基础 | 状态 |
|------|---------|------|
| **Prop 1.1** 直线路径最小化传输代价 | Cauchy-Schwarz 不等式 | ✅ 严格，无争议 |
| **Prop 1.2** 检测损失时间敏感度 δx₀ = -t·δv | 链式法则 + v_θ=(x_t-x₀_pred)/t | ✅ 修正后推导正确 |
| **Prop 1.3** AdaLN-Zero 初始化保证 | 代码审计确认 Block 2 无 alpha gate | ✅ 修正后描述准确 |
| **DPM-Solver++ RF 推导** | 半线性 ODE 分解 + 积分因子 | ✅ 数学正确，离线验证通过 |

### 1.2 需实验支撑的命题（逻辑方向正确，但非严格定理）

| 命题 | 逻辑基础 | 局限 |
|------|---------|------|
| **Prop 1.4** 多样性-传输效率权衡 | 方向性关系（非确定性方程） | 仅提供回归模型 β₀-β₁·C_trans+β₂·D_idx-β₃·B_match，无闭式最优解 |
| **Decomp 2.2** 总误差分解 | ε_total = ε_discretization + ε_generalization | ε_generalization 的界是启发式复杂度项，**非已证明的泛化界** |
| **维度假说** R_idx = logK/(logK+d·log(1/τ)) | 经验指标，非定理 | 文档自身标注为"方向性论证" |

### 1.3 机制假说（需进一步验证）

| 命题 | 基础 | 局限 |
|------|------|------|
| **Prop D.1** Reflow 梯度冲突 | ρ=-0.104 实测 | 推导中对 dv_θ/dθ 做了标量简化（"严格意义上不成立"——原文自述） |
| **Hyp E.1** Stochastic OT+TRD 冲突 | 机制分析 | 无直接梯度测量验证 |
| **Hyp E.2** CAT-OT Voronoi 边界冲突 | 几何分析 + 代码级论证 | 解释了 collapse (eval=0) 但无直接梯度测量 |
| **Scale-Conditioned 三定理** | OT 等价 + 变分原理 + 信息论 | 依赖 OT 最优性假定、FM 变分解释等前提，前提本身未在文档中严格证明 |

### 1.4 总体理论评价

- **自洽性**: 经过 13 处错误修正（三校），理论框架内部自洽
- **诚实性**: 每个命题标注了证据等级（严格命题/可检验命题/机制假说），自我批判程度高于大多数已发表论文
- **弱点**: 泛化界（Decomp 2.2）和维度假说（R_idx）无法宣称严格理论贡献；梯度冲突推导（Prop D.1）在审稿人面前可能被质疑

---

## 二、实验证据覆盖度

### 2.1 有实验支撑的声明

| 声明 | 实验证据 | 可信度 |
|------|---------|--------|
| RF > DDPM | ldmdet_rf 0.733 vs baseline 0.725 | ⚠️ DDPM 的 `_ddim_step` 有负索引 bug，DDPM baseline 数值不可靠 |
| Shifted Schedule > Uniform | +1.4% | ✅ 无已知 bug |
| AdaLN-Zero > Scale-Shift | +0.3%（vs 同期非 adaln baseline +1.1%） | ✅ 无已知 bug |
| Hard OT < Random | 0.735 vs 0.751 | ✅ 趋势稳健（不受单 seed 波动影响） |
| Stochastic > Argmax | 0.751 vs 0.748 | ⚠️ 差值 0.003，单 seed 波动已达 0.018 |
| Sinkhorn ε 扫描 in [0.5,5] | 平坦曲线 → CAM 命题 | ⚠️ argmax 下的 ε 扫描各点均为单 seed |
| Reflow Epoch1 最优后退化 | 所有 Reflow 实验一致呈现 | ✅ 趋势稳健 |
| TRD/CAT 组合效应 | 各组合均低于单路径最佳 | ⚠️ velocity target 符号错误影响所有相关数值 |
| DPM-Solver++ > Heun (离线) | 0.755 离线验证 | ✅ 需确认推理一致性 |

### 2.2 缺少实验支撑的声明

| 声明 | 缺口 | 严重程度 |
|------|------|---------|
| 通用检测改进（非仅染色体） | **COCO/LVIS 验证未做** | 🔴 致命（对 CVPR/ICCV 而言） |
| 推理速度优势 | **SOTA 模型 FPS 未测** | 🔴 重大（扩散检测的核心卖点是速度-精度权衡） |
| SOTA mAP=0.753 | **单 seed，另一 seed=0.735** | 🔴 重大（差值 1.8% 超过大多数声称改进） |
| Stochastic OT 可复现 | TF32 被确认为系统性偏差源 | 🟡 根因已找到，但 5-seed 统计待做 |
| 与标准检测器对比 | 未在染色体数据集跑 Faster R-CNN/DINO/YOLO | 🟡 缺少外部基线 |
| Per-class AP | 未报告 | 🟡 对染色体应用重要（小染色体 vs 大染色体） |
| Test set 评估 | test set 存在但从未使用 | 🟡 所有结果均为 validation set |
| velocity loss / TRD / CAT / Reflow | 代码 bug 修复后需重跑 | 🔴 所有含 velocity loss 的数值不可信 |

### 2.3 数据集与训练规模

#### 数据集 1: Chromosome20240904_NoAug_NoResize_coco（主力实验数据集）

| 项 | 实际值 |
|----|--------|
| 训练集 | 1,540 张图片，72,023 个标注 |
| 验证集 | 440 张 |
| 测试集 | 220 张（存在但未使用） |
| 类别 | 24 类（非标准 C 组顺序: C10,C11,C12,C6,C7,C8,C9） |

#### 数据集 2: 24_chromosomes_object（基准测试数据集）★

| 项 | 实际值 |
|----|--------|
| 训练集 | **3,500 张**图片，160,888 个标注 |
| 验证集 | **500 张**，22,984 个标注 |
| 测试集 | **1,000 张**，45,980 个标注 |
| 总图片数 | **5,000 张** |
| 类别 | 24 类（标准 C 组顺序: C6,C7,C8,C9,C10,C11,C12） |
| 来源 | `D:\Taichung_chromosomes\`（原始 XML 路径） |

#### ⚠️ 数据集 2 与 ChromosomeNet 的关系

XML 文件中的原始路径为 `D:\Taichung_chromosomes\chromosome_original\`。ChromosomeNet 论文 (Kuo et al., IEEE OJEMB 2025) 的数据来源为 **Taichung Veterans General Hospital（台中荣民总医院）**，同样是 5,000 张中期细胞图像 — 来源: https://pubmed.ncbi.nlm.nih.gov/39906268/

**判断：24_chromosomes_object 与 ChromosomeNet 使用的极可能是同一数据集（或来自同一医院的不同批次）。**

证据：
| 线索 | 24obj 数据集 | ChromosomeNet |
|------|------------|---------------|
| 来源路径 | `Taichung_chromosomes` | Taichung Veterans General Hospital |
| 总图片数 | 5,000 | 5,000 |
| 类别数 | 24 | 24 |
| 标注格式 | XML (Pascal VOC) → 转为 COCO JSON | 未知 |
| 每图染色体数 | ~46 | 未明确报告（但典型中期细胞为 46） |

**如果确认是同一数据集，这意味着：**

1. ✅ **可以直接与 ChromosomeNet 的 99.60% mAP50 对比** — 我们只需在同一数据集上报告 mAP50
2. ✅ **论文的说服力大幅增强** — "在 ChromosomeNet 同一数据集上，我们的扩散检测器在 COCO-style mAP 上达到 X，而 ChromosomeNet 报告 mAP50 为 99.60%"
3. ⚠️ **指标不可直接比较** — mAP50 与 mAP (IoU 0.50:0.95) 是不同的指标，需同时报告两者
4. 🔴 **需确认 ChromosomeNet 的 train/val/test 划分** — 如果划分方式不同，对比不完全公平

#### 24obj Benchmark 配置（已就绪，待运行）

| 模型 | 配置文件 | 状态 |
|------|---------|------|
| Cascade R-CNN R50 | `benchmark_24obj/cascade_rcnn_r50.py` | ❌ 未跑 |
| DINO R50 | `benchmark_24obj/dino_r50.py` | ❌ 未跑 |
| RTMDet-L | `benchmark_24obj/rtmdet_l.py` | ❌ 未跑 |
| YOLOX-S | `benchmark_24obj/yolox_s.py` | ❌ 未跑 |
| DiffusionDet (DDPM) | `benchmark_diffusiondet_24obj.py` | ❌ 未跑 |
| LDMDet SOTA | `benchmark_24obj/ldmdet_rf_heun_adaln_stochot_eps5.py` | ❌ 未跑 |

#### 单卡训练 batch_size

| GPU | 原数据集 | 24obj 数据集 |
|-----|---------|-------------|
| A6000 48GB | bs=4-6 | bs=4-6 |
| A5000 24GB | bs=2 | bs=2 |
| 训练 epoch | ~60-120 | ~60-150 |

---

## 三、创新性与已发表工作对比

### 3.1 最接近的竞争论文

**DeFloMat: Detection with Flow Matching for Stable and Efficient Generative Object Localization**
- 发表: arXiv 2512.22406, 2025年12月 — https://arxiv.org/abs/2512.22406
- 核心: 将 DiffusionDet 的 DDPM 替换为 Rectified Flow（基于 Conditional OT）
- 结果: 3步推理，43.32% AP10:50（MRE 医学数据集；⚠️ AP10:50 = IoU 0.10~0.50 均值，远宽松于标准 AP50）
- **威胁**: 这是当前最接近的工作——他们已先于我们发表了"Flow Matching 用于目标检测"
- **差异**: DeFloMat 未探索 OT 耦合策略选择、未分析多样性坍缩、未使用 Stochastic Coupling、未尝试单步推理、未做染色体应用

### 3.2 各贡献维度的新颖性判断

| 贡献 | 最接近工作 | 新颖性判断 |
|------|-----------|-----------|
| RF 替代 DDPM 做检测 | DeFloMat (Dec 2025) | 🔴 **非首发**。DeFloMat 领先 ~5个月 |
| OT 耦合用于检测训练 | 无已知论文 | 🟢 **新**。未发现任何论文对 DiffusionDet 做 OT coupling |
| OT 多样性坍缩 (低维) | 逆问题中的 variance collapse (Mar 2026) — https://arxiv.org/abs/2603.14135 | 🟢 **新颖框架**。对方未归因于 OT 确定性/维度 |
| Stochastic Coupling | IDBM/Schrödinger Bridge (生成方向) — https://arxiv.org/abs/2304.00917 | 🟢 **检测场景首次**。生成领域有类似概念 |
| CAM 命题 | 无已知论文 | 🟢 **新**。argmax 削弱 ε 调控的观察未见报道 |
| 四因素框架 | 无已知统一框架 | 🟢 **新**。但"框架"类贡献需要被广泛引用才成立 |
| 1-2步扩散检测 | DeFloMat 3步，无 1步工作 | 🟢 **开放问题**，但 Reflow 当前结果未超过 DeFloMat 的 3步 |
| 染色体扩散检测 | 所有染色体工作用 YOLO/Faster R-CNN | 🟢 **首次**。但染色体社区使用标准检测器已达 99%+ mAP50 |
| Flow Matching 检测 + Sinkhorn OT | 无已知论文 | 🟢 **新**。OTCS (NeurIPS 2023) 做了 OT+扩散，但是图像到图像翻译 |
| Large Sinkhorn Couplings 用于 Flow | On Fitting Flow Models with Large Sinkhorn Couplings (2025) — https://arxiv.org/abs/2506.05526 | 🟡 同时期独立工作，方向不同（生成 vs 检测）|

### 3.3 核心新颖性总结

> 如果 DeFloMat 不存在，核心贡献是"首次将 Flow Matching 用于检测"。但 DeFloMat 存在，因此必须将论文重心从"FM for detection"转移到"**OT coupling 策略选择 + 多样性坍缩理论 + Stochastic Coupling 解决方案 + 染色体核型分析应用**"。

### 3.4 与 DeFloMat 的差异化策略

| DeFloMat 做了的 | 我们的额外贡献 |
|----------------|---------------|
| RF/CFM 替换 DDPM | OT 耦合策略系统对比 (Nearest/Sinkhorn argmax/Sinkhorn stochastic/Group Hierarchical) |
| 3步推理 (AP10:50=43.32%) | OT 多样性坍缩理论 + CAM 命题 |
| MRE 医学数据 | Stochastic Coupling 作为确定性 OT 的修复方案 |
| — | Reflow 1步推理 + 梯度冲突分析 |
| — | 四因素统一框架 |
| — | DPM-Solver++ 加速 |
| — | 染色体核型分析 + KCEC 探索 |

---

### 3.5 DeFloMat 发表状态核查

**当前状态 (2026-05-28): 仅 arXiv 预印本，未被任何期刊/会议接收**

核查结果:
- **无 OpenReview 记录**: 在所有主要会议 (ICLR, CVPR, NeurIPS, ECCV) 的 OpenReview 中搜索 "DeFloMat" 均无结果
- **无 GitHub 仓库**: 未找到公开代码
- **无期刊发表**: 未在任何期刊检索到
- **发表时间线分析**:
  - arXiv 上传: 2025-12-26
  - ICLR 2026 deadline: 2025-09-24 → 上传时 ICLR 2026 投稿已截止
  - CVPR 2026 deadline: 2025-11-13 → 上传时 CVPR 2026 投稿已截止
  - ECCV 2026 deadline: 2026-03-05 → **可能在审中**，decision 2026-06-17 — https://eccv.ecva.net/
  - AAAI 2027 deadline: 2026-07-28 → 可投 — https://aaai.org/conference/aaai/aaai-27/

**解读**:

1. **最可能的情况**: DeFloMat 投了 ECCV 2026（投稿前上传 arXiv 在 CS 领域普遍接受）。如果是这样，6月中旬就会有结果。
2. **次可能**: 论文被 CVPR 2026 拒后上传 arXiv，目前未重新投稿，或在投期刊。
3. **不太可能**: 单纯占坑不上传（浪费 5 个月）。

**对我们的影响**:

| 维度 | 影响 |
|------|------|
| 首发权 | ✅ **我们仍有机会成为第一个被同行评审接收的 "FM for Detection" 论文** |
| 引用义务 | 🔴 **必须引用** DeFloMat，无论其是否被接收 |
| 差异化压力 | 🟡 如果 DeFloMat 被 ECCV 2026 接收（6月出结果），我们的差异化需要更明确 |
| 时间窗口 | 🟢 如果 DeFloMat 也被拒，我们有机会在 CVPR 2027 "首次正式发表" |

**DeFloMat 的可能弱点（推测，未看到评审意见）**:

从公开信息推断，DeFloMat 最可能被拒的原因:
1. **单一医学数据集** (MRE Crohn's Disease) — 与我们的染色体问题相同，通用性未验证
2. **仅替换 DDPM→FM** — 方法学贡献有限，审稿人可能认为"直接把 DiffusionDet 的 DDPM 换成 FM" 不够新颖
3. **非标准评估指标** — 使用 AP10:50（IoU 0.10~0.50 均值）而非标准 AP50 或 COCO mAP，审稿人可能质疑指标选择有 cherry-picking 嫌疑
4. **无理论分析** — 没有解释为什么 FM 比 DDPM 更适合检测
5. **仅与 DiffusionDet 对比** — 缺少与其他检测范式 (DETR, YOLO, Faster R-CNN) 的对比
6. **无代码** — 影响可复现性评分

> **关键教训**: DeFloMat 可能的被拒原因恰恰说明了单纯的 "replace DDPM with FM" 不够。我们的 OT 耦合策略、多样性坍缩理论、Stochastic Coupling 恰好弥补了这些缺陷——这些构成了我们相对于 DeFloMat 的核心增量。

---

## 四、适用范围分析

### 4.1 已验证有效的范围

- 数据集: 染色体核型分析 (1,540 训练图, 24类, K≈46 实例/图)
- 模型: ResNet-50 + AdaLN-Zero + Shifted Schedule + Heun 求解器
- 推理步数: 4步（最佳 0.753），2步性价比最优 (0.740，与 4步差距 0.003)
- 检测空间: ℝ⁴ 边界框

### 4.2 未验证的范围

- **通用目标检测** (COCO, LVIS): 零验证
- **更强 backbone** (Swin, ConvNeXt): ConvNeXtV2-Tiny 实验中退化 (0.736 vs ResNet-50 0.753)
- **更大规模数据** (>10万张): 零验证
- **不同检测空间维度**: d=4 的结论是否泛化到 d>4 未知

### 4.3 维度假说的预测（均未验证）

- R_idx 高 (d 小, K 大): 多样性坍缩风险高 → 染色体 (d=4, K≈46, R_idx≈0.47)
- R_idx 中 (d 中等, K 中等): 需要验证 → COCO (d=4, K≈7.3, R_idx≈0.18)
- R_idx 低 (d 大, K 小): 传输效率可能主导 → 图像生成 (d=64×64)
- **这些预测均为假说，一条都未经实验验证**

---

## 五、接受概率评估

### 5.1 CVPR 2027

| 因素 | 评估 | 方向 |
|------|------|------|
| Diffusion + Detection 仍热 | CVPR 2025 有 Generalized Diffusion Detector, ReDiffDet 等多篇；NeurIPS 2025 有 Promptable 3-D Localization (隐空间扩散做 3D 检测) — https://cvpr.thecvf.com/Conferences/2025 | 🟢 方向匹配 |
| Flow Matching 热度上升 | ICLR 2025 有多篇 FM 理论和应用 (Poster/Oral) — https://openreview.net/group?id=ICLR.cc/2025/Conference | 🟢 技术路线匹配 |
| 要求 COCO 验证 | CVPR 检测论文几乎 100% 报告 COCO 结果 | 🔴 **当前零 COCO 验证是致命缺陷** |
| 要求 SOTA 对比 | 染色体数据集非公开 benchmark，无法与社区对比 | 🔴 **外部 baseline 缺失严重** |
| 医学/专用数据集论文 | CVPR 接受专用数据集论文但需有通用 insight | 🟡 四因素框架声称通用但未在通用数据验证 |
| **综合**: COCO 验证是入场券，当前无 COCO → 大概率 desk reject。加上 COCO 后，理论贡献 + 染色体应用可构成有竞争力的投稿 | |

### 5.2 ICLR 2027

| 因素 | 评估 | 方向 |
|------|------|------|
| 偏好理论深度 | OT 多样性坍缩 + 四因素框架 + CAM 命题 | 🟢 理论贡献匹配 |
| 理论严密性要求 | Decomp 2.2 泛化界是启发式；Prop D.1 梯度推导有简化 | 🔴 **审稿人会指出这些不是严格定理** |
| 可复现性要求 | Stochastic OT 单 seed 方差 1.8% | 🔴 **可复现性不足是 ICLR 高频拒稿理由** |
| 实验规模 | 60+ 实验但均在单个私有数据集 | 🟡 需要在讨论中说服审稿人 |
| **综合**: 适合 ICLR —— 如果修复了理论文档中的"假定理"表述并补充 5-seed 统计。不加 COCO 也可能接受（ICLR 对专用数据集包容度高于 CVPR） | |

### 5.3 AAAI 2027

| 因素 | 评估 | 方向 |
|------|------|------|
| 染色体应用+AI | AAAI 对 AI+Science 方向友好 | 🟢 应用匹配 |
| 实验要求 | 相对 CVPR/ICLR 稍低 | 🟢 |
| 理论深度 | AAAI 不如 ICLR 看重理论 | 🟡 理论贡献可能被低估 |
| **综合**: 在没有 COCO 的情况下可能是最快的 CCF-A 选项 | |

### 5.4 TPAMI

| 因素 | 评估 | 方向 |
|------|------|------|
| 实验量 | 60+ 实验，如果补齐相当于一本小书 | 🟢 期刊偏好完整叙事 |
| 理论深度 | 如果将所有假说降级为"可检验命题"则合适 | 🟢 |
| 审稿周期 | 6-12月，时间充裕可从容补齐所有缺口 | 🟢 |
| ChromosomeNet 对比 | 已有 99.6% mAP50 的染色体检测 | 🔴 需直接对比或承认差距 |
| **综合**: 如果愿意等 12-18 个月，TPAMI 是最佳选项（无 deadline 压力 + 完整 story） | |

### 5.5 近期顶会录用趋势参考

**CVPR 2025 检测方向趋势**:
- Generalized Diffusion Detector: 用扩散模型做域泛化检测 — https://arxiv.org/abs/2503.02101
- ReDiffDet: 旋转等变扩散模型做旋转目标检测 — https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_ReDiffDet_Rotation-equivariant_Diffusion_Model_for_Oriented_Object_Detection_CVPR_2025_paper.html
- 扩散模型+检测是持续热点，但竞争激烈

**NeurIPS 2025 相关录用**:
- ReCon: 扩散数据增强做检测 (NeurIPS 2025 Spotlight) — https://openreview.net/forum?id=2zjH76SmiF
- Neptune-X: 多模态条件生成做海事检测 (Spotlight) — https://nips.cc/virtual/2025/poster/116049
- Promptable 3-D Localization: 隐空间扩散做 3D 检测 — https://proceedings.neurips.cc/paper_files/paper/2025/file/f71b1f37df59d34924f61e9fce05a35a-Paper-Conference.pdf

**ICLR 2025 Flow Matching 趋势**:
- Flow Matching 理论和应用是当前热点
- Faster Inference of Flow Models via Improved Data-Noise Coupling (ICLR 2025 Poster) — https://openreview.net/forum?id=rsGPrJDIhh
- Flow Matching with General Discrete Paths (ICLR 2025 Oral) — https://openreview.net/forum?id=tcvMzR2NrP
- Generator Matching (ICLR 2025 Oral) — https://openreview.net/forum?id=RuP17cJtZo

---

### 5.6 接受概率排序（当前状态 → 补齐后）

| Venue | 当前概率 | 补齐 COCO 后 | 补齐全部 P0 后 |
|-------|---------|-------------|---------------|
| AAAI 2027 | 30-40% | — | 50-60% |
| ICLR 2027 | 25-35% | 40-50% | 50-60% |
| CVPR 2027 | 10-20% | 35-45% | 45-55% |
| TPAMI | 30-40% | 50-60% | 60-70% |
| Pattern Recognition | 40-50% | 60-70% | 70-80% |
| Medical Image Analysis | 35-45% | — | 50-65% |

> 概率基于: 当前代码 bug 未修复、无 COCO 验证、无多 seed 统计、单数据集评估的状态。**所有概率均为估值，不代表实际结果。**

---

## 六、补充清单

### 6.1 理论补充

| 编号 | 内容 | 优先级 | 预估工时 |
|------|------|--------|---------|
| **T1** | 降低梯度冲突推导的数学声明：将 Prop D.1 从"定理"降为"机制假说 + 实验测量" | P0 | 0.5天 |
| **T2** | 降低泛化界的数学声明：将 Decomp 2.2 从"界"降为"解释性分解" | P0 | 0.5天 |
| **T3** | 维度假说形式化：明确 R_idx 是"经验相关指标"非"预测性定理" | P0 | 0.5天 |
| **T4** | 补充 COCO 维度假说预测：基于 R_idx 公式预测 COCO (d=4, K≈7.3) 的预期行为 | P1 | 1天 |
| **T5** | Scale-Conditioned 定理前提审查：检查 OT 最优性假定在离散检测场景下是否成立 | P1 | 2天 |

### 6.2 实验补充（P0 — 论文必需）

| 编号 | 内容 | 硬件需求 | 预估时间 |
|------|------|---------|---------|
| **E1** | 修复 velocity target 符号 bug：`v_target = x_noises - x_starts`，修复所有调用处 | — | 1天 |
| **E2** | 重跑 velocity/TRD/CAT/Reflow 受影响实验：trd_only, trd_full, velocity, cat_only, Reflow v2-v6, ITD, stochastic+TRD/CAT 组合 | A6000 | 2-3周 (连续跑) |
| **E3** | Stochastic OT 5-seed 统计：reproduce eps=5 的 5 seed mean±std；同样做 eps=1,2 | A6000 + A5000 | 1周 (5 seed 并行) |
| **E4** | Test set 最终评估：用 best checkpoint 在 held-out test set (220张) 上跑一次 | A6000 | 0.5天 |
| **E5** | 外部 baseline：在 24obj 数据集上运行已配置好的 Cascade R-CNN R50, DINO R50, RTMDet-L, YOLOX-S, DiffusionDet (DDPM), LDMDet SOTA | A6000 | 1-2周 (6个模型) |
| **E6** | SOTA 模型 FPS benchmark：RF-AdaLN-StochOT 在 1/2/4/8 步的 FPS + 参数量 + FLOPs | A6000 | 0.5天 |

### 6.3 实验补充（P1 — 强化论文）

| 编号 | 内容 | 硬件需求 | 预估时间 |
|------|------|---------|---------|
| **E7** | COCO 验证：在 COCO 2017 train(118K)/val(5K) 上训练并评估主模型 | A6000 (48GB) | 3-4周 (全量) / 2-3天 (mini-train 5K) |
| **E8** | Per-class AP 分析：染色体 24 类分别报告，分析小染色体 vs 大染色体差异 | — | 0.5天 |
| **E9** | 更多 backbone：Swin-T 在染色体数据集 | A6000 | 1周 |
| **E10** | 纯 curvature CAT 实现：v_θ(t+dt)-v_θ(t) 惩罚替代 x₀ consistency | A6000 | 1天实现 + 3天训练 |
| **E11** | Ablation: 逐模块消融表（已有大部分数据，需补格式） | — | 1天 |

### 6.4 指标补充

| 指标 | 当前状态 | 需求 |
|------|---------|------|
| mAP / mAP50 / mAP75 | ✅ 有 (但仅 validation set) | 补充 test set |
| mAP_s / mAP_m / mAP_l | ✅ 有 | — |
| Per-class AP | ❌ 无 | 补充 |
| FPS | ❌ 仅旧 DDPM baseline | 补充 SOTA 模型 |
| FLOPs / Params | ❌ 无 | 补充 |
| AR (Average Recall) | ❌ 无 | 可选补充 |
| 5-seed mean±std | ❌ 几乎全无 | 补充关键声明 |

---

## 七、硬件约束下的时间估算

| 硬件 | 显存 | 能力 | 用途 |
|------|------|------|------|
| A6000 | 48GB | 单卡 bs=4-6 | 主力训练 + COCO 大数据集 |
| A5000 | 24GB | 单卡 bs=2 | 并行跑 multi-seed / 消融 |

### 关键路径 (假设 A6000 + A5000 双卡)

```
Week 1-2:      E1 代码修复 + E2 velocity/TRD/CAT/Reflow 重跑 (A6000)
Week 3:        E3 5-seed 统计 (A6000 + A5000 并行)
Week 4:        E4 test set + E5 外部 baseline (A5000) + E6 FPS (A6000)
Week 5-8:      E7 COCO 训练 (A6000 独占)
Week 9:        论文撰写
```

### COCO 训练说明

- COCO 2017 train 118K 张，验证 5K 张
- A6000 bs=4 预估: 1 epoch ~2-3天，需 ~36 epochs → **14-21天**
- 加速方案: COCO mini-train（随机抽样 5K-10K 张），2-3天出概念验证结果
- 如果时间不足: 仅做 COCO mini-train 作为概念验证，论文中以"preliminary COCO results"呈现

---

## 八、各 Venue 对应的最低可行包

### AAAI 2027 (8月投，最紧)

```
必需: E1, E2, E3 (至少 3 seed), E4, E5, E6, T1, T2, T3
可跳过: E7 COCO (AAAI 对专用数据集包容度更高)
时间: ~5周 (6月初→7月中) → 刚好赶上
风险: 高（时间极紧，代码 bug 修复后可能发现新问题）
```

### ICLR 2027 (9月投)

```
必需: E1-E6 全部, T1-T4, 纯 curvature CAT 可选
建议: E7 COCO 概念验证 (mini-train 5K)
时间: ~8周 (6月初→8月初) → 充足
```

### CVPR 2027 (11月投)

```
必需: E1-E9 全部, T1-T5
时间: ~16周 (6月初→10月中) → 充裕
```

### TPAMI（随时投）

```
必需: E1-E11 全部, T1-T5, 论文完整撰写
时间: 无 deadline 压力，6-12个月从容准备
```

---

## 九、建议决策顺序

```
1. 先修代码 bug (E1)，重跑关键实验 (E2) — 这是所有路线的前置条件
2. 同时做 5-seed 统计 (E3) + external baseline (E5) + FPS (E6)
3. 观察重跑结果:
   a. 如果 5-seed mean 稳定在 0.75+ → 论文故事稳固
   b. 如果 5-seed mean < 0.74 → 需要重新评估贡献强度
4. 根据剩余时间选择:
   - 7月中完成 → 投 AAAI 2027 (不用 COCO)
   - 8月中完成 → 投 ICLR 2027 (加 COCO mini-train)
   - 10月中完成 → 投 CVPR 2027 (加完整 COCO)
   - 不赶 deadline → 投 TPAMI (从容做完整)
```

---

## 附录: 文献检索关键发现

### 直接竞争

| 论文 | 发表时间 | 与我们的关系 |
|------|---------|-------------|
| **DeFloMat** (arXiv 2512.22406) | 2025-12 | RF+CFM 替换 DDPM 做检测，3步推理，43.32% AP10:50（非标准 AP50）。先于我们 ~5个月发表。**必须引用并差异化** — https://arxiv.org/abs/2512.22406 |
| **ReCon** (NeurIPS 2025 Spotlight) | 2025-12 | 扩散数据增强做检测，不同方向 — https://openreview.net/forum?id=2zjH76SmiF |
| **Promptable 3-D Localization** (NeurIPS 2025) | 2025-12 | 隐空间扩散做 3D 检测 — https://proceedings.neurips.cc/paper_files/paper/2025/file/f71b1f37df59d34924f61e9fce05a35a-Paper-Conference.pdf |
| **OTCS** (NeurIPS 2023) | 2023-12 | OT 耦合 + 扩散做图像翻译，非检测 — https://openreview.net/forum?id=9Muli2zoFn |
| **Faster Inference of Flow Models via Improved Data-Noise Coupling** (ICLR 2025 Poster) | 2025-04 | Flow Matching 的数据-噪声耦合优化，方向相关但非检测 — https://openreview.net/forum?id=rsGPrJDIhh |

### 染色体检测竞争

| 论文 | 发表时间 | 指标 |
|------|---------|------|
| **ChromosomeNet** (IEEE OJEMB 2025) | 2025 | 99.60% mAP50, 5,000 张图 — https://pubmed.ncbi.nlm.nih.gov/39906268/ |
| **Aycromo** (arXiv 2604.24685) | 2026-04 | 99.40% mAP50, YOLOv11 — https://arxiv.org/abs/2604.24685 |
| **Hybrid framework** (Scientific Reports 2026) | 2026 | 98.03% detection acc (mAP@0.5=0.985), YOLOv8+ResNet50 — https://www.nature.com/srep/ (来源: 生物通 2026-05-19 报道 https://m.ebiotrade.com/newsf/2026-5/20260519000634725.htm) |

> 注意: 以上染色体论文均使用标准检测器 (YOLO/Faster R-CNN)，非扩散方法，报告 mAP@50 而非 COCO-style mAP@50:95。我们的 mAP50 0.94-0.95 低于这些工作的 98-99%，但 mAP@50:95 与 mAP@50 不可直接比较。

---

*评估日期: 2026-05-28 | 基于 MASTER_TIMELINE.md, THEORY_FRAMEWORK.md, THEORY_WHY_FAILED.md, 60+ 实验日志, 以及外部文献检索*
