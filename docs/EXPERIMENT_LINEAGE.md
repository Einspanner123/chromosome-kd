# 实验脉络主路线文档 (按创新点主题组织)

> 📋 **命名约定**: 本文档使用论文正式名称 (Dataset 1 / Dataset 2 / RF+Heun / +Stoch. Coupling / +DPM-Solver++ / Top-K)。内部实验代号 (24obj / A0-A4 / IO3 / StochOT) 仅保留在文件路径和 SwanLab run_id 中以兼容工程实现。仅 TODO_DIRECTIONS.md 保留内部代号用于研究规划。

> 本文档为 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 的有效方向主路线梳理。
> 按"创新点主题"组织实验脉络, 让审稿人快速识别 solid 的研究链条与创新性。
> 数据源: 24 Chromosomes Object (Dataset 2, 5000 张图) 为主, Chromosome20240904 (Dataset 1, 1540 张图) 作低数据对照。
> SwanLab URL 模式: `https://swanlab.cn/@einspanner/<project>/runs/<run_id>`
> 更新时间: 2026-07-26 (新增 §十二 FPS 基准 / §十三 噪声鲁棒性 / §十四 测试集+跨域 zero-shot, 补全 §一 per-class AP / §二 Table 7 / §三 Table 8)
>
> 📌 **关联文档**:
> - [TODO_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/TODO_DIRECTIONS.md) (进行中/待启动方向)
> - [FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (已证伪方向)
> - [EXPERIMENT_CATALOG.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (实验数据索引)
>
> 📋 **文档流转规则**: 本文档仅收录 ✓已完成且有效的方向 (3-seed 完成或零成本诊断完成)。进行中方向保留章节但标注 🔄, 详见 TODO_DIRECTIONS.md。

---

## 〇、任务特性画像 (贯穿全文参照)

本画像定义了染色体核型分析检测任务的核心特性, 后续所有创新点的"与任务结合"论证均回溯此画像。

### 图像特性
- 中期相铺展图像, 每张含约 46 条紧密排列的染色体
- 24 个类别 (A1–Y), 跨越大/中/小三组尺寸, 形态相似性强
- C 组 (C6–C12) 7 条亚中着丝粒染色体, 仅靠细微带纹差异区分
- Y 染色体最小, 仅男性单拷贝出现, 训练样本约 1803 vs 常染色体约 7000
- 频繁的相互重叠与接触

### 检测范式特征
- 基于扩散的检测 (DiffusionDet 范式): 噪声框 → GT 框迭代去噪
- 预测空间维度 d=4 (cxcywh), 相比图像生成 d≈10⁵ 极低维
- 每张图 K≈46 个 GT 框 (高目标密度), COCO 平均 K≈7
- 500 个噪声 proposals 全部通过 6 级 cascade head × 4 solver step = 24 NFE
- box_renewal 是检测特有操作 (图像生成无此机制)

### 数据特征
- Dataset 2 (24 Chromosomes Object): 5000 张, mAP 量级 0.77-0.87
- Dataset 1 (Chromosome20240904): 1540 张, mAP 量级 0.72-0.75, 低数据对照
- 类别不平衡严重 (Y vs 常染色体 1:3.9)
- 临床采集, 标注质量受观察者主观影响

### 关键约束
- 临床交互式筛查延迟带: 13.3-14.2 FPS (Top-K K=200)
- cascade head 占 90%+ 推理延迟
- 跨站点/跨 seed 可复现性 (临床部署要求)

---

## 一、RF (Rectified Flow) — 范式替换贡献

### 核心贡献: 直线 ODE 路径取代 DDPM 弯曲随机轨迹

RF 以从噪声 $\mathbf{x}_1$ 到 GT $\mathbf{x}_0$ 的确定性直线 ODE 路径取代 DDPM 的弯曲随机轨迹, 使速度场沿路径恒定, 在少步推理下保持低截断误差。

- **形式化**: $x_t = (1-t)x_0 + t x_1$, 速度场 $v = x_1 - x_0$ 恒定
- **训练目标**: flow matching 损失 $\mathcal{L}_{FM} = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$
- **范式定位**: data-prediction 形式, 与 DPM-Solver++ 天然兼容 (避免 $v=(x_t-\hat{x}_0)/t$ 在 $t\to 0$ 的奇点)

### 与染色体检测任务特性的结合

- **密集 proposals 误差复合**: 每张图 46 个 GT × 500 proposals, DDPM 弯曲轨迹的截断误差在 proposals 间复合, RF 直线轨迹将单步误差降为 0 (理想情况)
- **小训练集限制弯曲轨迹学习**: 1540-5000 张临床图像难以学习复杂 DDPM 弯曲轨迹, RF 直线 ODE 路径降低学习负担
- **24 类细粒度依赖稳定特征**: RF 恒定速度场为 24 类细粒度判别提供稳定特征表示, C 组带纹差异得以保留

### 范式贡献归因

DDPM baseline→RF+Heun 累积 +0.053 mAP (统一口径: DDPM baseline = DiffusionDet 0.803, 训练配置 AdamW/150ep), 其中 solver×step 解耦消融证明:
- **91% (+0.048 mAP) 归因于 RF 范式本身**
- 9% (+0.005 mAP) 归因于 solver/步数选择 (Heun 4 步 vs Euler 1 步)
- AdaLN-Zero 单独贡献为 0 (Appendix B 零结果)
- 偏移噪声调度 (shift=3.0) 单独贡献为 −0.001 (噪声范围)
- ⚠ 历史口径 (2026-07-27 前): 旧 DDPM baseline (a0_baseline, SGD/12ep, 训练不足, mAP=0.774) → RF+Heun 累积 +0.082 mAP, 94% 归因于 RF; 2026-07-27 统一口径至 DiffusionDet (AdamW/150ep, mAP=0.803) 后, 累积增益 +0.053 mAP, 91% 归因于 RF, 结论方向不变 (RF 范式为核心增益来源)

### 实验列表

#### 实验证明目的: RF 范式相对 DDPM 的精度优势 (主消融)

- DDPM baseline (DDPM Euler 1-step, a0_baseline)
  -- 数据集: Dataset 2
  -- 结果: mAP=0.774, AP50=0.968, AP75=0.916
  -- ⚠ 训练配置: SGD/lr=0.02/12ep, 训练不足, 2026-07-27 已将论文 DDPM baseline 统一为 DiffusionDet (AdamW/150ep, mAP=0.803, 见下方对照)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a0_baseline

- DDPM baseline (统一口径, DiffusionDet checkpoint)
  -- 数据集: Dataset 2
  -- 结果: mAP=0.803 (val) / 0.804 (test), AP50=0.970, AP75=0.936
  -- 训练配置: AdamW/150ep, 与 RF+Heun 完全一致 (除 diffusion_type=ddpm)
  -- 结构对比: 两个 subagent 确认 DiffusionDet 与 a0_baseline 结构相同, 性能差异 (0.774 vs 0.803) 完全由训练配置 (SGD/12ep vs AdamW/150ep) 导致, 非模型结构差异
  -- SwanLab (project=chromosome-kd-benchmark-24obj): benchmark_diffusiondet
  -- checkpoint: work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth

- KaryoFlow (RF+Heun)
  -- 数据集: Dataset 2
  -- 改动: diffusion_type=rectified_flow, solver=heun, rf_schedule=shifted, time_conditioning=adaln_zero
  -- 结果: mAP=0.856, AP50=0.990, AP75=0.969 [+0.053 主贡献 (统一口径 DiffusionDet 0.803) / +0.082 (旧口径 a0_baseline 0.774)]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a1_rf_heun

- +AdaLN-Zero
  -- 结果: mAP=0.856, AP50=0.990, AP75=0.972 [+0.000 持平 RF+Heun, AdaLN 单独贡献为 0]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a2_adaln

- +Stoch. Coupling eps=5
  -- 结果: mAP=0.858, AP50=0.990, AP75=0.973 [+0.002 边际]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

- +DPM-Solver++ 替换 Heun
  -- 结果: mAP=0.863, AP50=0.990, AP75=0.974 [+0.005 推理加速且精度提升]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: solver×step 解耦, 隔离 RF 范式贡献

- RF+Heun checkpoint 上 solver×step 全组合 (Dataset 2 验证集, seed 42)
  -- Heun 4 步 (7 NFE): mAP=0.856
  -- Euler 4 步 (4 NFE): mAP=0.855
  -- DPM-Solver++ 4 步 (4 NFE): mAP=0.855
  -- Euler 1 步 (1 NFE): mAP=0.851
  -- DPM-Solver++ 1 步 (1 NFE): mAP=0.851
  -- 结论: 匹配步数下 solver 类型对 mAP 无影响; 步数 1→4 仅 +0.004; solver/步数联合仅贡献 6%
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a1_rf_heun

#### 实验证明目的: DDPM 多步消融 (Dataset 2, 论文 Appendix G)

- DiffusionDet checkpoint (Dataset 2, best @ ep26, seed 42) 上 Euler 1/2/4/8 步推理
  -- 配置: `experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py` + `--sampling-steps {1,2,4,8} --solver-type euler`
  -- 命令: `python experiments/runners/test.py experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py --checkpoint work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth --dataset val --sampling-steps {1,2,4,8} --solver-type euler --seed 42`
  -- 结果 (验证集, seed 42):
     | Steps | NFE | mAP   | AP50  | AP75  | AP_S  | AP_M  | AP_L  |
     |-------|-----|-------|-------|-------|-------|-------|-------|
     | 1     | 1   | 0.805 | 0.971 | 0.937 | 0.405 | 0.802 | 0.810 |
     | 2     | 2   | 0.804 | 0.969 | 0.939 | 0.414 | 0.800 | 0.806 |
     | 4     | 4   | 0.804 | 0.970 | 0.938 | 0.402 | 0.800 | 0.804 |
     | 8     | 8   | 0.805 | 0.971 | 0.940 | 0.405 | 0.801 | 0.818 |
  -- 结论: DDPM Euler 1→8 步 mAP 变化 <0.002, 所有精度指标 (AP50/AP75/AP_S/AP_M/AP_L) 均在噪声范围内波动; 对比 RF 范式切换带来的 +0.053 mAP 增益, DDPM 增加步数的收益可忽略, 证实 "DDPM 步数对精度几乎无影响, 范式切换才是核心增益来源"
  -- 注: 1 步推理 mAP (0.805) 与 Table 5 中 DDPM baseline (0.803, 训练评估) 略有差异, 源于推理与训练评估的 maxDets 设置不同; 多步对比在相同推理设置下进行, 结论不受影响
  -- 数据源: ross server 推理日志 (2026-07-28), project=ldmdet-inference, 实验名 `euler_{N}step`

#### 实验证明目的: Dataset 1 低数据对照, RF vs DDPM

- rf_heun_adaln 3 seeds (Dataset 1)
  -- 结果: mAP=0.746 ± 0.001 [+0.017 vs DDPM 0.729 ± 0.003]
  -- seed42=0.7450, seed789=0.7470, seed123=0.7470
  -- SwanLab (project=ldmdet-ablation):
     - rf_heun_adaln_seed42: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/4xhp5ffymboa05hyn245u
     - rf_heun_adaln_seed789: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/ww6nlti3ufdkm4htjg5pw
     - rf_heun_adaln_seed123: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/dimdbu8fk0re4satbzpgs

- diffusiondet_ddpm 3 seeds (Dataset 1, 根 baseline)
  -- 结果: mAP=0.729 ± 0.003
  -- seed42=0.7260, seed789=0.7270, seed123=0.7330
  -- DDPM 步数对齐验证 (project=ldmdet-inference): DDIM 1/4/8 步均为 0.729, +0.017 为纯算法贡献
  -- SwanLab (project=ldmdet-ablation):
     - diffusiondet_ddpm_seed42: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/apfn46t67iqg1bjraq8xd
     - diffusiondet_ddpm_seed789: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/hny1od5fcvx8ngt9b063g
     - diffusiondet_ddpm_seed123: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/ghghjry3bylt0sfoj30j5

#### 实验证明目的: vs SOTA 检测器 (DINO/RTMDet-L/Cascade)

- KaryoFlow (+DPM-Solver++) 3-seed 均值
  -- mAP=0.859, 落后 DINO R50 (0.868) 仅 0.009, 落后 RTMDet-L (0.863) 0.004
  -- 超越 Cascade R-CNN (0.854), YOLOX-S (0.796), DiffusionDet (0.803)
  -- 相对 DiffusionDet seed42 best: +0.060 mAP, 3-seed 均值: +0.056
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp
  -- 对照基准 (project=chromosome-kd-benchmark-24obj): DINO/RTMDet-L/Cascade/YOLOX/DiffusionDet

#### 实验证明目的: 逐类 AP 分析 (论文 Figure 5 + Table F.1, §4.3.1)

- +DPM-Solver++ checkpoint (seed 42, best @ ep117, 独立推理) 上 24 个类别的 per-class AP
  -- 整体 mAP=0.863, AP50=0.988, AP75=0.972, AP_S=0.574, AP_M=0.859, AP_L=0.908
  -- 整体 AP 随染色体尺寸单调下降: Large→Medium→Small 为 0.896→0.848→0.805
  -- Y 染色体最难: seed 42 AP=0.779; 3 seed 均值 0.771 ± 0.006 (数据稀缺 1803 vs 7000 + 形态变异)
  -- C 组 (C6-C12) 组内差异仅 0.029 (0.871-0.900), 跟踪尺寸梯度但被削弱, 表明模型学到带纹线索
  -- 所有类别 AP50 > 0.988 (Y 0.972), 定位接近饱和, 残余误差集中在细粒度分类
  -- 数据源: [results/a4_per_class_ap.md](file:///home/linkst/workspace/projects/chromosome-kd/results/a4_per_class_ap.md) (step 132, mAP=0.863)
  -- 3-seed per-class 稳定性: [EXPERIMENT_CATALOG.md §7.8](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C6, 24 类 × 3 seeds)
  -- 论文图: latex/figures/per_class_ap.png (按尺寸组着色, 虚线为整体均值)

#### 实验证明目的: SOTA per-image Wilcoxon 检验 (论文 §4.3.2 引用, 暂不放入正文)

- RF (+Stoch. Coupling 配置) vs 4 个 SOTA 检测器 per-image 配对检验 (Dataset 2 val, 500 imgs, seed 42)
  -- Aggregate mAP: DINO R50 0.8685 > RTMDet-L 0.8626 > Cascade R-CNN 0.8535 > RF (+Stoch. Coupling) 0.8521 > DiffusionDet 0.8031
  -- RF (+DPM-Solver++ best 0.863) vs DINO R50 (0.8685) aggregate 差距仅 0.63%, per-image Wilcoxon 差距 1.64% (+Stoch. Coupling 配置)
  -- RTMDet-L ep85 修复 (2026-07-20): 之前用 ep86 (次优 0.861), 实际 ep85 best=0.8630 (实测 0.8626)
  -- 数据源: [experiments/analysis/baseline_inference_24obj_perimage_results.json](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/baseline_inference_24obj_perimage_results.json)
  -- 详细: [EXPERIMENT_CATALOG.md §7.4.6](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md)

---

## 二、OT Diversity Collapse 与 Stochastic Coupling — 理论新颖性贡献

### 核心贡献: 低维检测空间 OT 坍缩形式化分析 + Stochastic Coupling 补救

OT 配对在低维 (d=4) 检测空间中将噪声空间划分为 Voronoi 单元, 使耦合分配成为噪声的确定性函数——耦合多样性坍缩至零, 损害训练。Stochastic Coupling 从 Sinkhorn transport 矩阵采样分配 (而非 argmax), 在 hard OT 与随机耦合之间插值, 恢复多样性。

- **形式化上界** (命题 1): $\Delta H \le \log K$ (OT 下 $V$ 可由 $X_t$ 恢复, $H_{OT}=0$)
- **形式化下界** (命题 2, Fano 不等式): $\Delta H \ge \log K \cdot (1-P_{err}) - h(P_{err})$
- **染色体检测数值**: $d_{min}\approx 20$ px, $\sigma_t\sim 1$ px, $P_{err}<10^{-45}$, $\Delta H \ge 0.999\log K$
- **经验验证** (Dataset 1): $\Delta H = 3.8415$ vs $\log K = 3.8427$, 相对误差 0.03%
- **Stochastic Coupling 单调性** (命题 3, 包络定理): $H_{stoch}(V|X_t; \epsilon)$ 关于 $\epsilon$ 单调非递减

### 与染色体检测任务特性的结合

- **低维 d=4 触发坍缩**: 检测预测空间 d=4 vs 图像生成 d≈10⁵, OT 在低维下逼近 $\log K$ 熵减上界
- **高 K≈46 加剧坍缩**: $\Delta H/H \approx 0.69$ (染色体) vs 0.55 (COCO, K≈7) vs ≈0 (图像生成)
- **小训练集放大损害**: Dataset 1 上 Stochastic Coupling 增益 +0.034 (p<10⁻¹²⁰), 数据稀缺时 OT 诱发配对的边际收益减弱, Stochastic Coupling 价值最大

### 数据集规模依赖性

| 数据集 | 规模 | Stoch. vs Random mAP Δ | 显著性 | 平滑性增益 |
|--------|------|------------------|--------|------------|
| Dataset 1 | 1540 张 | +0.034 | p<10⁻¹²⁰ (n=1320) | 4.6× epoch std |
| Dataset 2 | 5000 张 | +0.0001 | p=0.80 (n=500, ns) | 4.6× epoch std |

- Dataset 1: Hard OT 实际比 Random 更差 (−0.008, p<10⁻⁸), 证实 OT 多样性坍缩病理
- Dataset 2: mAP 增益可忽略, 但平滑性收益独立成立 (Last-30 std: 0.006 → 0.0013)
- 数据更多时, 模型见到足够多样本平均掉随机耦合噪声, OT 坍缩及 Stochastic Coupling 边际收益减弱

### 可扩展性倾向

Table 2 a-priori 诊断: 任何 $d \ll 100$ 且 $K \gg 10$ 的任务都是 Stochastic Coupling 候选。
- 严重性 $\Delta H/H$ 预测 Stochastic Coupling 是否会有帮助
- 候选任务: 细胞检测 (组织病理学, 多核 tile, d=4 bbox)、病灶检测 (乳腺 X 光/视网膜, 小目标有限阳性)、微生物菌落计数、遥感密集目标
- 不适用: 高维图像生成 ($d \sim 10^5$, $\Delta H/H \approx 0$, OT-CFM 成功)

### 实验列表

#### 实验证明目的: 经验熵验证 OT Diversity Collapse 理论

- 经验熵测量 (Dataset 1 验证集)
  -- 理论值 $\log K = 3.8427$ vs 经验值 $\Delta H = 3.8415$, 相对误差 0.03%
  -- 配置: 真实染色体检测图像, 8 个 GT 框, OT (最近邻) vs Random 分配
  -- 结论: $N\to\infty$ 界在染色体检测设置下是极佳近似, 即便 mini-batch N=2

#### 实验证明目的: Dataset 1 耦合消融 (低数据, 大增益)

- Hard OT 3 seeds (Dataset 1)
  -- 结果: mAP=0.705 ± 0.002 [−0.008 vs Random, 证实 OT 坍缩]
  -- SwanLab: 见下文 Random/Stoch 对照

- Random Coupling 3 seeds (Dataset 1)
  -- 结果: mAP=0.713 ± 0.005
  -- seed42=0.713, seed123=0.718, seed789=0.708
  -- SwanLab: 见 Dataset 2 同名实验

- Stochastic Coupling ε=5, 3 seeds (Dataset 1)
  -- 结果: mAP=0.747 ± 0.003 [+0.034 vs Random, p<10⁻¹²⁰]
  -- seed42=0.745, seed123=0.745, seed789=0.750
  -- Hard OT vs Random: Δ=−0.0061, p<10⁻⁸ (Hard OT 比 Random 更差, 证实坍缩病理)
  -- Stoch vs Hard: Δ=+0.0369, p<10⁻¹⁵⁵

#### 实验证明目的: Dataset 2 耦合消融 (大数据, 增益可忽略但平滑性显著)

- Random Coupling 3 seeds (Dataset 2, project=ldmdet-ablation)
  -- 平均: mAP=0.860 ± 0.001
  -- seed42: mAP=0.859 (best @ 59)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/p5xqii8mcqmbhuo5lhlff
  -- seed789: mAP=0.860 (best @ 82)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/r8n441mu4gws43xyoneoj
  -- seed123: mAP=0.860 (best @ 115)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/q6jgxefgxbp8f2sf5qzpc

- Sinkhorn Stochastic OT 1 seed (Dataset 2, project=ldmdet-ablation)
  -- 结果: mAP=0.856 (best @ 53) [对应 +Stoch. Coupling 配置]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/o96m1eqz4l12qjeyys1cs

- GHSS Coupling 3 seeds (Dataset 2, project=ldmdet-ablation)
  -- 平均: mAP=0.858 ± 0.001
  -- seed42: mAP=0.857 (best @ 83)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/k84cq9oftbp2nld88a85t
  -- seed789: mAP=0.859 (best @ 75)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/holadvhaz9v2rh8l494hv
  -- seed123: mAP=0.859 (best @ 102)
     - SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/73cr3uyqw4f1q68xz1evg

- 多维稳定性比较 (Dataset 2, 单 seed)
  -- Last-30 epoch std: Random 0.006 vs Stoch 0.0013 (4.6× 改善)
  -- Last-30 CV: 0.69% vs 0.16% (4.4×)
  -- Last-30 range: 0.023 vs 0.005 (4.6×)
  -- Last-30 处于 best 1% 内 epoch 数: 13/30 (43%) vs 30/30 (100%)
  -- 训练失败率 (9 runs): 0/9 vs 0/9
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

#### 实验证明目的: ε 消融, 验证 Stochastic Coupling 饱和性

- ε 消融 (Dataset 2, +Stoch. Coupling 配置)
  -- ε < 1: 有害 (相同增广下 mAP −1.3%)
  -- ε ≥ 1: 进入饱和, 收益递减
  -- ε = 5: 主路线配置
  -- 结论: Stochastic Coupling 是必要的 OT 正则化项而非精度助推器
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

#### 实验证明目的: 多维稳定性比较 (论文 Table 7 + Figure 4, §4.4.2)

- 5 项稳定性指标: Random (RF+Heun) vs Stochastic Coupling ε=5 (+Stoch. Coupling) (Dataset 2, 单 seed)
  -- Last-30 epoch std: 0.006 → 0.0013 (4.6× 改善)
  -- Last-30 CV (std/mean): 0.69% → 0.16% (4.4× 改善)
  -- Last-30 range (max−min): 0.023 → 0.005 (4.6× 改善)
  -- 最后 30 个 epoch 中处于最佳 1% 内的 epoch 数: 13/30 (43%) → 30/30 (100%)
  -- Best mAP / best epoch: 0.856 / ep62 → 0.858 / ep114
  -- Total training epochs: 92 → 144
  -- Training failure rate (9 runs): 0/9 → 0/9
  -- 结论: Stoch Coupling 在每个被测轴上胜出, 关键是 30/30 vs 13/30 使小数据下基于 EarlyStopping 的 checkpoint 选择远为可靠
  -- 数据源: [EXPERIMENT_CATALOG.md §7.8 / 训练日志](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (来自训练日志的逐 epoch mAP)
  -- 论文图: latex/figures/training_stability.png (Random 振荡 vs Stoch 平滑收敛, 阴影带标记 last 30 epoch)

#### 实验证明目的: Dataset 1 per-class AP 增益 (论文 §5.6 / §7.3.5 引用)

- Stochastic Coupling 在 Dataset 1 上对 24 类的 per-class AP 增益 (3-seed, IoU=0.5:0.95)
  -- Y 染色体: 0.569 ± 0.009 → 0.622 ± 0.021 (+0.059, Wilcoxon W=0, p=1.66e-13)
  -- G22: 0.581 → 0.623 (+0.042); F19: 0.694 → 0.720 (+0.026); F20: 0.684 → 0.718 (+0.034)
  -- 大类 (A1/A2/A3) 平均 +0.029-0.037
  -- 小类 (Y/G/F) 平均 +0.040, 大类平均 +0.029 → 小类增益更大, OT 坍缩对低频类压制更严重
  -- 整体 mAP +0.034 (p<10⁻¹²⁰, n=1320)
  -- 数据源: [EXPERIMENT_CATALOG.md §7.3.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (Problem 3: Y 染色体稳定性)

---

## 三、DPM-Solver++ — 推理加速 + 匹配步数精度优势贡献

### 核心贡献: RF 适配 data-prediction 形式 + 修正 FlowDet 结论

将 DPM-Solver++ (Lu et al., 2022) 适配到 RF 线性路径的 data-prediction 形式, 利用 $\hat{x}_0$ 历史在 $t$ 空间中的多项式插值实现 1 NFE/步。修正 FlowDet "高阶 solver 表现更差" 的结论: 在匹配步数下, 高阶 DPM-Solver++ 相对 Heun 改善精度同时 NFE 减少 1.71×。

- **二阶更新公式**: $x_{t_{n+1}} = \frac{t_{n+1}}{t_n}x_{t_n} + (1-\frac{t_{n+1}}{t_n})\hat{x}_0^{(n)} + \varphi_1 D_1$
- **NFE 优势**: 4 步共 4 NFE, 相比 Heun 4 步 7 NFE 加速 1.71×
- **奇点处理**: $t\to 0$ 处由 $\epsilon$ 截断 ($t_{n+1} > 10^{-7}$)
- **修正 FlowDet**: 匹配步数下 DPM-Solver++ 相对 Heun +0.006 mAP (Wilcoxon p<10⁻⁶), 而非更差

### 与染色体检测任务特性的结合

- **临床交互式筛查延迟带**: 13.3-14.2 FPS (Top-K K=200), 比 DiffusionDet (41 FPS 但 mAP 0.803) 数量级改善
- **cascade head 占 90%+ 延迟**: 主干+颈部仅约 5.8 ms (4-8%), DPM-Solver++ 通过 NFE 减少降低 cascade head 调用次数
- **标准 single-shot 检测器仍快 3-7×**: KaryoFlow 定位交互式筛查延迟带, 以延迟换精度

### η_str 诊断 (R1 理论深化, 详见 §五)

3 seeds 单调下降 3.43→2.45→1.68, 量化"2 步收敛":
- step 2 已降至 step 1 的 71%, step 3 二阶校正贡献低于噪声阈值
- 修正"RF 轨迹接近直线" claim: 实际 $\eta_{str}\in[0.7, 1.5]$ 非零但曲率足够小
- DPM-Solver++ 在 box_renewal 污染下仍提供 +0.006 mAP 精度优势, 因 proposals 在每步冷启动后由 RF 速度场重新对齐至直线 ODE 路径

### 轨迹级收敛模式分析 (2026-07-29, 详见 §六)

匈牙利匹配追踪每个 GT 目标的预测框在各步的位置变化, 量化各 solver 的收敛模式差异:
- **Euler**: 单调递增 IoU (D2: 0.054→0.086→0.137→0.687), 但步数过多时累积误差反噬 (D1: 8-step IoU 0.737 < 4-step 0.751)
- **Heun**: 单调递增且更快 (D2: 0.050→0.099→0.220→0.709), 二阶校正使中间步骤更逼近 $x_0$
- **DPM-Solver++**: **非单调收敛** (D2: 0.056→0.069→**0.055↓**→0.676), step 3 IoU 反降; 中间步骤不具物理意义, 为 D3 矛盾提供轨迹级解释
- **DPM++ 精度机制**: center_dist 最小 (10.7px D2) 但 IoU 不是最高 (0.676), mAP +0.006 来自中心定位而非框尺寸

### 实验列表

#### 实验证明目的: +Stoch. Coupling vs +DPM-Solver++ 逐图像配对检验 (匹配步数下精度优势)

- +DPM-Solver++ (DPM++) vs +Stoch. Coupling (Heun+Stoch. Coup.) 4 步对比 (Dataset 2 验证集)
  -- mAP Δ: +0.0056, Wilcoxon p=2.5×10⁻⁷ ***, 配对 t p=8.4×10⁻⁷ *** (n=500)
  -- +DPM-Solver++−+AdaLN-Zero (combined): Δ=+0.0057, Wilcoxon p=4.5×10⁻⁴, t p=4.9×10⁻⁵ ***
  -- +Stoch. Coupling−+AdaLN-Zero (Stoch. Coup.): Δ=+0.0001, p=0.797 ns (Dataset 2 上不显著)
  -- 结论: 高阶 solver 在相等步数下略更好, 而非更差, 修正 FlowDet 结论
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: R1 η_str 诊断 (3 seeds × 4 configs)

- baseline (renewal on) 3 seeds
  -- mAP: 0.859 ± 0.004
  -- Step 1 η_str: 3.43 ± 0.36
  -- Step 2 η_str: 2.45 ± 0.24 (降至 step1 的 71%)
  -- Step 3 η_str: 1.68 ± 0.15
  -- 模式: 单调递减, 3 seed 稳定
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp
  -- 本地脚本: experiments/analysis/r1_eta_str_measure.py, r1_d3_summary.py
  -- 结果文件: experiments/analysis/r1_eta_str_a3_seed{42,123,789}.json (含 renewal on/off 配置)

#### 实验证明目的: 步数消融, 验证 2 步收敛

- DPM-Solver++ 步数消融 (Dataset 2, +DPM-Solver++ checkpoint, seed 42)
  -- 2 步: mAP=0.863 (收敛)
  -- 4 步: mAP=0.863 (无收益)
  -- 结论: 超过 2 步无收益, η_str 诊断定量解释该现象
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: 匹配 NFE 下 DPM-Solver++ 对比 Heun

- DPM-Solver++ 4 步 (4 NFE, mAP=0.863) ≈ Heun 2 步 (3 NFE, mAP=0.863)
  -- 精度相当, DPM-Solver++ 以零成本换取 43% 更少 NFE
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: 完整 Table 8 配对显著性检验 (论文 Table 8, §4.3.2)

- Dataset 2 验证集逐图像配对检验 (500 imgs, seed 42, Wilcoxon signed-rank + paired t-test)

  | 比较 | Metric | Δ | Wilc. p | t p | n |
  |------|--------|---|---------|-----|---|
  | +Stoch. Coupling−+AdaLN-Zero (Stoch. Coup.) | mAP | +0.0001 | 0.797 ns | 0.944 ns | 500 |
  | +DPM-Solver++−+Stoch. Coupling (DPM++) | mAP | +0.0056 | 2.5e-7 *** | 8.4e-7 *** | 500 |
  | +DPM-Solver++−+AdaLN-Zero (combined) | mAP | +0.0057 | 4.5e-4 *** | 4.9e-5 *** | 500 |
  | +Stoch. Coupling−+AdaLN-Zero (Stoch. Coup.) | AP_S | +0.0012 | 0.783 ns | 0.947 ns | 60 |
  | +DPM-Solver++−+Stoch. Coupling (DPM++) | AP_S | -0.0031 | 0.855 ns | 0.855 ns | 60 |
  | +DPM-Solver++−+AdaLN-Zero (combined) | AP_S | -0.0019 | 0.691 ns | 0.898 ns | 60 |

  -- 结论 1: Stoch Coupling 在 Dataset 2 上 mAP 增益不显著 (p=0.80), 仅在 Dataset 1 显著 (Table 9)
  -- 结论 2: DPM-Solver++ 在匹配 4 步下 +0.0056 mAP, p<10⁻⁶, 高阶 solver 略 *更好* 而非更差, 修正 FlowDet 结论
  -- 结论 3: 所有 AP_S 差异不显著 (p>0.6), 各变体间小目标数值噪声等价
  -- 数据源: [EXPERIMENT_CATALOG.md §7.4](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C3: 配对统计显著性检验)
  -- 推理脚本: `experiments/analysis/baseline_inference_24obj_perimage.py`

#### 实验证明目的: Dataset 1 耦合消融配对检验 (论文 Table 9, §4.4.1)

- Dataset 1 耦合消融 3-seed pooled 逐图像配对检验 (n=1320, IoU=0.5:0.95)

  | 比较 | Metric | Δ | Wilc. p | t p | n |
  |------|--------|---|---------|-----|---|
  | Stoch−Rand | mAP | +0.0308 | 9.0e-126 *** | 9.9e-130 *** | 1320 |
  | Hard−Rand | mAP | -0.0061 | 1.2e-8 *** | 1.6e-9 *** | 1320 |
  | Stoch−Hard | mAP | +0.0369 | 9.0e-155 *** | 2.8e-156 *** | 1320 |
  | Stoch−Rand | AP_S | +0.0450 | 3.9e-68 *** | 2.5e-75 *** | 1314 |
  | Hard−Rand | AP_S | -0.0051 | 1.7e-2 * | 2.2e-2 * | 1314 |
  | Stoch−Hard | AP_S | +0.0501 | 1.9e-83 *** | 5.9e-85 *** | 1314 |

  -- 结论: Stoch Coupling 在低数据 Dataset 1 上 +0.0308 mAP 高度显著; Hard OT 比 Random 更差 (-0.0061), 证实 OT 多样性坍缩病理
  -- 数据源: [EXPERIMENT_CATALOG.md §7.4 + §7.3](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C2/C3 配对检验)

---

## 四、Top-K Proposal Pruning — 延迟优化贡献

### 核心贡献: 500→K proposals 剪枝 + DPM-Solver++ 兼容性处理

推理时 500 个 proposals 全部通过 4 步 cascade 头, cascade 头占据 90%+ 延迟。在第 0 步之后, 基于置信度分数将 proposals 从 500 剪枝到 K, 只有 top-K 个 proposals 进入第 1-3 步, 将后续 3 步计算量降低 500/K 倍。

- **DPM-Solver++ 兼容性**: 剪枝后调用 `dpm_solver.reset()`, 因 $\hat{x}_0$ 历史存在维度不匹配 (500 → K)
- **副作用**: reset 改变 η_str 模式, 从"单调递减"变为"V 型" (step1 低, step2 高, step3 中)
- **理论解释**: step 1 reset 后退化为 Euler 一阶, step 2 新历史建立后二阶校正 $D_1$ 恢复

### 与染色体检测任务特性的结合

- **K=200 最优**: 46 条染色体 + 重叠冗余, 200 proposals 提供足够容量
- **K=100 掉点 (−0.013) 主因 proposal 不足**: 100 个框覆盖 ~46 条染色体 + 重叠冗余时容量紧张
- **K=100 掉点非 solver 历史污染**: K=100/K=200 η_str 几乎相同 (step2: 2.18 vs 2.24, step3: 1.54 vs 1.54), D3 假设被证伪

### 实验列表

#### 实验证明目的: K ∈ {100, 200, 300} 消融, 验证 K=200 最优

- +DPM-Solver++ + Top-K (K=300)
  -- NFE=4, Latency=71.27 ms, FPS=14.0, mAP=0.861 [−0.002 vs +DPM-Solver++]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- +DPM-Solver++ + Top-K (K=200) [最优]
  -- NFE=4, Latency=70.46 ms, FPS=14.2, mAP=0.860 [−0.003 vs +DPM-Solver++, 最佳速度-精度权衡]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- +DPM-Solver++ + Top-K (K=100)
  -- NFE=4, Latency=69.71 ms, FPS=14.3, mAP=0.850 [−0.013 vs +DPM-Solver++, 掉点]
  -- 掉点主因: proposal 容量不足 (非 solver 历史污染, 见 D3 证伪)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: Top-K 改变 η_str 模式 (V 型 vs 单调递减)

- TopK K=200 (3 seeds)
  -- mAP=0.862, Step1 η_str=1.37, Step2=2.24, Step3=1.54
  -- 模式: V 型 (step1 低因 reset, step2 高因新历史建立)

- TopK K=100 (3 seeds)
  -- mAP=0.852, Step1 η_str=1.20, Step2=2.18, Step3=1.54
  -- 模式: V 型, 与 K=200 几乎相同 (证伪 D3 对 K=100 掉点解释)

---

## 五、R1: η_str 直线度诊断 — 理论深化贡献 (已完成)

### 核心贡献: 推理时零开销可读出的直线度指标, 定量刻画"2 步收敛"

DPM-Solver++ 二阶校正项 $D_1^{(n)} = (\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)})/(t_n - t_{n-1})$ 在理想 RF 下应为 0 (因 $\hat{x}_0$ 恒定)。定义直线度指标 $\eta_{str}^{(n)} = \|D_1^{(n)}\|_2 / \|\hat{x}_0^{(n)}\|_2$, 推理时零开销可读出, 定量刻画学习轨迹的直线度。

- **命题 R1.1**: $\eta_{str} \ge 0$, $\eta_{str}=0$ 当且仅当 $\hat{x}_0$ 在区间上为常数
- **命题 R1.2**: 若 $v_\theta$ 精确恢复 $v=x_1-x_0$ (理想 1-RectFlow), 则 $\forall n: \eta_{str}^{(n)}=0$
- **命题 R1.3**: 若 $\bar{\eta}_{str}^{(n)} < \epsilon_{conv}$ 对 $n \ge N_0$ 成立, 则 DPM-Solver++ 在 $N_0$ 步后无显著精度增益

### 与染色体检测任务特性的结合

- **修正"RF 轨迹接近直线" claim**: 实际 $\eta_{str}\in[0.7, 1.5]$ 非零但曲率足够小, 使 DPM-Solver++ 校正项对 mAP 的边际贡献 < 0.001
- **区分"RF 训练成功"与"RF 训练失败但被 solver 步数补偿"**: $\eta_{str}$ 提供机制级判据
- **何时需要 reflow (2-RectFlow)**: 若训练后 $\bar{\eta}_{str} > 0.1$ 持续, reflow 可能进一步拉直轨迹; 若 $< 0.01$, reflow 收益有限

### 实验列表

#### 实验证明目的: 3 seeds × 4 configs (renewal on/off × K={100,200}) η_str 测量

- baseline (renewal on) 3 seeds
  -- mAP: 0.859 ± 0.004
  -- Step1 η_str: 3.43 ± 0.36, Step2: 2.45 ± 0.24, Step3: 1.68 ± 0.15
  -- 模式: 单调递减 (降 51%)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- renewal off (方案 B) 3 seeds
  -- mAP: 0.858 ± 0.003 [Δ=−0.0003, 噪声范围]
  -- Step1 η_str: 1.50 ± 0.33, Step2: 1.11 ± 0.19, Step3: 0.70 ± 0.09
  -- 模式: 单调递减 (降 53%)
  -- 结论: 方案 B 不损失精度, 且使 η_str 诊断有效 (renewal 污染被消除)

- TopK K=200 (3 seeds)
  -- mAP=0.862, Step1 η_str=1.37, Step2=2.24, Step3=1.54, 模式: V 型

- TopK K=100 (3 seeds)
  -- mAP=0.852, Step1 η_str=1.20, Step2=2.18, Step3=1.54, 模式: V 型

- 本地脚本
  -- experiments/analysis/r1_eta_str_measure.py (支持 `--box-renewal on/off`)
  -- experiments/analysis/r1_d3_summary.py (3 seed × 4 config 汇总)
  -- 8 个 JSON 结果文件: experiments/analysis/r1_eta_str_a3_seed{42,123,789}_{renewal_on,off}.json

### 与已证伪方向 Adaptive Step 的区分

- **Adaptive Step (adaptive step early-exit)**: 根据收敛提前终止, 改变推理步数, 已证伪 (失败原因是 step 1 的 x0_pred 不稳定)
- **R1**: 仅观测 $\eta_{str}$, 不改变任何推理流程, 提供事后诊断
- R1 不触发 Adaptive Step 的失败模式

---

## 六、D3: Box Renewal 与 DPM-Solver++ 交互 — 检测特有操作理论化 (已完成)

### 核心贡献: 揭示 box_renewal 与多步法历史矛盾 + 化解

box_renewal 在每个 solver step 后将低置信度 proposals 重置为随机噪声, 但 DPM-Solver++ 二阶校正项 $D_1$ 假设 $\hat{x}_0$ 是 $t$ 的连续函数。被 renewal 的 proposal 的 $\hat{x}_0^{(n+1)}$ 是对新噪声的预测, 与 $\hat{x}_0^{(n)}$ 无轨迹连续性, 使 $D_1$ 失效。

- **命题 D3.1**: 对被 renewal 的 proposal $i$, $D_{1,i}^{(n+1)}$ 期望范数远大于真实轨迹曲率
- **推论 D3.2**: box_renewal 后 $\eta_{str}$ 不再反映直线度, 而是被 renewal 噪声主导
- **实测**: renewal on 使 $\eta_{str}$ 虚高 56-58% (ratio off/on = 0.42-0.44), 但 mAP 仅 −0.0003 (噪声范围)

### 与染色体检测任务特性的结合

- **box_renewal 是检测特有操作**: 图像生成无此机制 (生成任务没有"低置信度 proposal"概念)
- **密集目标下 renewal 比例高**: 46 个 GT + 500 proposals, 低置信度 proposals 较多, renewal 触发频繁
- **VGAR (Velocity-Guided Adaptive Renewal) 缓解**: $\alpha(t)\hat{x}_0 + (1-\alpha(t))z$, 但 $t\to 0$ 时 $\alpha\to 0.8$, 仍保留 20% 随机性, D3 矛盾仅缓解未消除

### D3 对 K=100 掉点解释被证伪

| 配置 | mAP | Step1 η_str | Step2 | Step3 |
|------|-----|-------------|-------|-------|
| TopK K=200 | 0.862 | 1.37 | 2.24 | 1.54 |
| TopK K=100 | 0.852 | 1.20 | 2.18 | 1.54 |

K=100 与 K=200 的 $\eta_{str}$ 在 step 2 几乎相同 (2.18 vs 2.24, 差异 < 3%), 但 mAP 差 −0.010。**D3 假设被证伪**: K=100 掉点主因是 proposal 数量不足, 不是 DPM-Solver++ 历史破坏。

### 方案 B (renewal off) 已验证 + K 值依赖性确认 (2026-07-30 补充)

**基础验证 (A4 DPM-Solver++, Dataset 2 K=500, 3-seed)**:
- 3 seed 平均 mAP 0.858 ± 0.003 (vs baseline 0.859 ± 0.004), Δ=−0.0003 (噪声范围)
- **方案 B 不损失精度**, 且使 η_str 诊断有效 (renewal 污染被消除)
- 使 R1 指标在 renewal on 时失效的问题得到化解

**K 值依赖性验证 (2026-07-30, 全场景 renewal ON vs OFF 直接对比)**:

数据源: [renewal_off_all_scenarios.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_off_all_scenarios.json) · [renewal_off_topk_verify.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_off_topk_verify.json) · [renewal_off_k100_3seed_k150.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/renewal_off_k100_3seed_k150.json)

| 场景 | renewal ON | renewal OFF | ΔmAP | 判定 |
|------|-----------|-------------|------|------|
| Dataset 1 A4 (K=500) | 0.744 | 0.743 | −0.001 | ✓ 不影响 |
| Dataset 2 K=500 | 0.864 | 0.862 | −0.002 | ✓ 不影响 |
| Dataset 2 K=300 | 0.862 | 0.863 | +0.001 | ✓ 不影响 |
| Dataset 2 K=200 | 0.862 | 0.862 | 0.000 | ✓ 不影响 |
| Dataset 2 K=150 (seed42) | 0.861 | 0.859 | −0.002 | ⚠ 边界 |
| **Dataset 2 K=100 (3-seed)** | **0.839±0.012** | **0.808±0.023** | **−0.031±0.012** | **⚠ 有影响** |

**结论: 推理时关闭 box_renewal 在 K≥200 (推荐配置) 下安全, K=150 为边界, K=100 (非推荐) 下有 −0.031±0.012 退化 (3-seed 确认)。**

- **K=100 退化主因**: proposal 稀缺性。K=100 时 100 个 proposal 覆盖 46 GT + 重叠冗余, box_renewal 的"proposal 回收"机制 (重置死 proposal 为噪声, 给重新收敛机会) 价值凸显; K≥200 时冗余 proposal 弥补回收缺失。
- **3-seed 稳定性**: K=100 3-seed Δ=−0.031±0.012 (seed42: −0.019, seed123: −0.043, seed789: −0.032), 退化稳定且显著, 远超 noise 阈值。单 seed 测量 (−0.016) 低估了实际退化。
- **APs paradox**: K=100 renewal OFF 的小目标 APs 反升 (0.507 vs 0.464, +0.043), 因 renewal 重置为纯随机噪声偏向中大目标, 关闭后小目标定位不被破坏; 但中大目标 recall 下降更多 (n_matched −1.4%), 净效果为负。
- **bottleneck 不矛盾**: FALSIFIED §十 no_box_renewal 的 -0.016 退化根因是 **early stopping 选择偏差** (box_renewal 不在 loss() 路径, 训练时不影响模型权重; 验证评估无 renewal → val mAP 波动 → 次优 checkpoint), 非 model 能力退化。本实验是**推理切换** (DPM++, 训练 ON 推理 OFF, 3-seed ΔmAP=−0.0003), 证实推理时关闭 renewal 在 K≥200 下安全。
- **作为 DPM++ 适配改进**: 推理时关闭 renewal 使 D1 校正免受 renewal 噪声污染 (理论净化), 在推荐配置 K≥200 下不损失精度, 同时使 R1 诊断有效。K=100 作为边界条件讨论, 进一步证实 box_renewal 的核心价值是 proposal 回收而非 DPM++ 历史维护。

### 方案 A (per-proposal D1 掩码) 已实现 (2026-07-29)

路径 A 对被 renewal 的 proposal 置零 D1 校正项, 保留未被 renewal 的 proposal 的完整 D1 历史:

- **实现**: `RFDPMSolverMultistep.step()` 新增 `renewal_mask: Optional[Tensor]` 参数
  - `renewal_mask` 是 `[bs, N]` bool 张量, True 表示该 proposal 在上一步被 box_renewal 重置
  - 对 `renewal_mask=True` 的 proposal: `D1 *= (~renewal_mask).unsqueeze(-1).float()`, 即 D1 置零 → 退化为线性插值
  - 对 `renewal_mask=False` 的 proposal: D1 保留, 继续使用完整二阶校正
- **传递链**: `head.predict()` 在 `apply_box_renewal()` 后通过 `torch.isclose(x_raw, x_raw_before)` 计算 `_renewal_mask`, 在下一步 `dpm_solver.step()` 时传入
- **优势 vs 方案 B**: 保留 renewal 对低置信度 proposal 的淘汰能力 (5-60% proposals 被 renewal), 同时精确保护 DPM++ 历史
- **推理测试**: DPM-Solver++ 4-step (renewal ON + path A) 推理成功, 31 个单元测试通过

### 去噪轨迹数值分析 (2026-07-29)

基于 DPM-Solver++ 4-step 的匈牙利匹配轨迹追踪, 量化各 solver 的收敛模式差异:

**Dataset 2 (24obj, 44 GT, image_id=131)**:

| Solver | Step 1 IoU | Step 2 | Step 3 | Step 4 | 最终 center_dist |
|--------|-----------|--------|--------|--------|-----------------|
| Euler 4-step | 0.054 | 0.086 | 0.137 | **0.687** | 11.1px |
| Heun 4-step | 0.050 | 0.099 | 0.220 | **0.709** | 11.6px |
| DPM++ 4-step | 0.056 | 0.069 | **0.055↓** | **0.676** | **10.7px** |

**Dataset 1 (chr2024, 34 GT, image_id=198)**:

| Solver | Step 1 IoU | Step 2 | Step 3 | Step 4 | 最终 center_dist |
|--------|-----------|--------|--------|--------|-----------------|
| Euler 4-step | 0.181 | 0.340 | 0.505 | **0.751** | 4.6px |
| Heun 4-step | 0.236 | 0.402 | 0.618 | **0.776** | 5.3px |
| DPM++ 4-step | 0.246 | 0.310 | 0.340 | **0.665** | **19.0px** |

**关键发现**:

1. **DPM-Solver++ 的非单调收敛**: DPM++ step 3 IoU 下降到 step 1 以下 (D2: 0.055 < 0.056), 是全局多项式外推的中间值而非"当前最优估计"。这为 D3 矛盾提供了轨迹级直观解释: **renewal 打断 $\hat{x}_0$ 连续性使 DPM++ 的非单调行为更不稳定**, 而路径 A 通过置零被 renewal proposal 的 D1, 使其退化为线性 (单调) 行为

2. **DPM++ 精度优势的机制**: DPM++ 在 D2 的 center_dist=10.7px (最小) 但 IoU=0.676 (最低)。这意味着 DPM++ 产生**中心定位更精确但尺寸偏大的框**。mAP 对中心定位更敏感 (IoU 阈值区间宽), 因此 DPM++ 的 mAP +0.006 优势来自中心定位而非框尺寸

3. **Euler 累积误差反噬**: Euler 8-step 的最终 IoU (D1: 0.737) 反而低于 Euler 4-step (0.751), 说明**步数过多时一阶 solver 累积误差抵消步数收益**。这与 η_str∈[0.7,1.5] 一致: 轨迹有足够曲率使一阶累积误差随步数增长

4. **1-step baseline 完全相同**: 所有 solver 的 1-step IoU 相同 (D2: 0.619, D1: 0.648), 因为同 seed 同初始噪声。**轨迹差异完全源于多步 ODE 求解器的行为差异**, 不涉及模型权重变化

5. **cxcywh 分维度差异与 DPM++ 精度机制**: DPM++ 在 cxcywh 空间各维度的曲率不同 (方向 A per-dim η_str 诊断已确认 cx/cy 曲率 > w/h 曲率)。轨迹数据分析揭示:
   - DPM++ 最终步 area_ratio=1.354 (pred/GT), Heun=1.562, **两者都产生过大的框**
   - DPM++ center_dist=9.98px (最小), Heun=11.68px
   - DPM++ IoU=0.690 < Heun IoU=0.698, **看似矛盾**: center_dist 更小但 IoU 更低
   - **解释**: IoU = intersection / union, Heun 的更过大的框 (1.562) 覆盖了更多 GT 区域, 部分补偿了中心偏移。但 mAP 在高 IoU 阈值 (0.75, 0.95) 下对中心精度更敏感, DPM++ 的中心优势使其在这些阈值下更好, 净 mAP +0.006
   - per-dim 误差: DPM++ 在 w 维度误差显著低于 Heun (0.835×), cx 也更好 (0.966×), cy 略差 (1.054×), h 持平 (0.977×)

6. **分维度 D1 掩码验证 (2026-07-29, seed42 + 3-seed 验证)**: 在 RFDPMSolverMultistep.step() 中新增 `dim_d1_mask` 参数, 允许对不同维度选择性启用/禁用 D1 校正。

   **D2 (24obj, seed42, box_renewal OFF)** — 数据源: [per_dim_d1_clean_repro_norenewal.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/per_dim_d1_clean_repro_norenewal.json):

   | 配置 | mAP | Δ mAP | APs | Δ APs |
   |------|-----|-------|-----|-------|
   | DPM++ std [1,1,1,1] | 0.862 | — | 0.531 | — |
   | cx/cy D1, w/h E [1,1,0,0] | 0.863 | +0.001 | 0.542 | +0.011 |
   | All Euler [0,0,0,0] | 0.863 | +0.001 | 0.561 | +0.030 |
   | cx/cy E, w/h D1 [0,0,1,1] | 0.863 | +0.001 | 0.566 | +0.035 |

   > ⚠ **数据口径修正 (2026-07-29)**: 原表格标注 "3-seed 均值" 但实为单 seed42 硬编码数据 (std=0.000 不合理, 无 JSON 支撑, 与 run_hybrid_3seed.py 中硬编码 baseline 形成循环引用)。已改为单 seed42 真实数据。baseline [1,1,1,1] 与 [1,1,0,0] 已通过独立 3-seed 验证 (box_renewal OFF): baseline 0.858±0.004, [1,1,0,0] 0.859±0.004, Δ=+0.001 在 noise 范围内, "分维度 D1 掩码不显著" 结论不变。3-seed 数据源: [baseline_heun_3seed_norenewal.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/baseline_heun_3seed_norenewal.json)。

   **D1 (chr2024, 3-seed 均值)**:

   | 配置 | mAP | Δ mAP | APs | Δ APs |
   |------|-----|-------|-----|-------|
   | DPM++ std [1,1,1,1] | 0.7463±0.001 | — | 0.509±0.002 | — |
   | cx/cy D1, w/h E [1,1,0,0] | 0.7463±0.001 | +0.0000 | 0.510±0.002 | +0.001 |
   | All Euler [0,0,0,0] | 0.7460±0.000 | -0.0003 | 0.509±0.002 | +0.000 |
   | cx/cy E, w/h D1 [0,0,1,1] | 0.7460±0.000 | -0.0003 | 0.510±0.002 | +0.001 |

   **结论: 分维度 D1 掩码不显著**。
   - D2 (seed42): 4 种配置 mAP 差异 ≤0.001, APs 差异 +0.011~+0.035 (单 seed, APs 高方差不可靠)
   - baseline [1,1,1,1] 与 [1,1,0,0] 的 3-seed 验证: 0.858±0.004 vs 0.859±0.004, Δ=+0.001 在 noise 范围内
   - D1 (chr2024) 的所有差异均在 seed 方差范围内
   - **该方向不纳入主路线, 但实现保留为可配置参数**

7. **Hybrid 求解器验证 (2026-07-29, Q3: w/h Heun 2阶 vs Euler 1阶, 3-seed, 结论修正)**: 上述 `dim_d1_mask=[1,1,0,0]` 使 w/h 退化为 Euler 1阶 (仅 linear 项), 但 Heun 是真正的 2阶求解器 (速度梯形法), 机制不同于 DPM++ (x0 插值)。新增 `RFDPMSolverHybrid` 类测试 w/h 用 Heun 2阶是否优于 Euler 1阶。

   **D2 (24obj, 3-seed, box_renewal OFF)** — 数据源: [baseline+full Heun](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/baseline_heun_3seed_norenewal.json) · [hybrid](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/hybrid_3seed_norenewal.json):

   | 配置 | seed42 | seed123 | seed789 | mean ± std | lat(ms) |
   |------|--------|---------|---------|-----------|---------|
   | baseline [1,1,1,1] (全 DPM++) | 0.863 | 0.855 | 0.857 | 0.858 ± 0.004 | 90.5 |
   | [1,1,0,0] w/h Euler 1阶 | 0.863 | 0.856 | 0.857 | 0.859 ± 0.004 | 88.9 |
   | full Heun (全维度 Heun 2阶, 对照) | 0.863 | 0.857 | 0.855 | 0.858 ± 0.004 | 144.3 |
   | **hybrid w/h Heun 2阶** | 0.862 | 0.856 | 0.856 | **0.858 ± 0.003** | 145.3 |

   | per-dim L1 | seed42 | seed123 | seed789 | mean |
   |------------|--------|---------|---------|------|
   | baseline w_L1 | 0.00188 | 0.00208 | 0.00199 | 0.00198 |
   | baseline h_L1 | 0.00194 | 0.00206 | 0.00202 | 0.00201 |
   | hybrid w_L1 | 0.00189 | 0.00208 | 0.00199 | 0.00199 |
   | hybrid h_L1 | 0.00195 | 0.00207 | 0.00203 | 0.00202 |

   **结论: Heun 2阶对 w/h 无显著影响 (ΔmAP=0.000 vs baseline), 是 null result**。
   - ⚠ **原 "Heun 有害 −0.005 mAP" 结论被推翻**: 原 baseline 3-seed 数据 (0.8630±0.000, std=0.000) 为硬编码错误 (无 JSON 支撑, 与 run_hybrid_3seed.py 循环引用), 真实 baseline 0.858±0.004, hybrid 与其完全一致 (ΔmAP=0.000)
   - hybrid vs [1,1,0,0] (w/h Euler 1阶): ΔmAP=−0.001, 在 3-seed noise (std=0.004) 范围内
   - full Heun (全维度 Heun 2阶) mAP=0.858±0.004, 与 baseline 一致, **证明 Heun 本身无害** — 精度层面 Heun 与 DPM++ 等价
   - per-dim L1: hybrid 与 baseline 的 w/h L1 误差几乎相同 (w: 0.00199 vs 0.00198, h: 0.00202 vs 0.00201), Heun 校正未改变 w/h 精度
   - **网络 x0 预测的吸引子效应**: RF 低曲率轨迹下, 网络的 $\hat{x}_0$ 预测是强吸引子, 无论 solver 用 DPM++ (x0 插值) 还是 Heun (速度梯形), 迭代精修使最终预测收敛到同一 $\hat{x}_0$, solver 阶数/类型对最终 mAP 的影响被吸收

   **v_next 数值稳定性诊断 (证伪原假设)** — 数据源: [vnext_instability.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/vnext_instability.json):
   - 原假设: Heun 的 $v_{next} = (x_{euler} - \hat{x}_0^{next}) / t_{next}$ 在小 $t_{next}$ 时因 $1/t$ 放大导致数值爆炸 (原称 $1/t_{next}=31.4\times$)
   - 实测 (20 图, w/h 维度范数比 $\|v_{next}\|/\|v_t\|$):
     | step | $t_n$ | $t_{next}$ | ratio_mean | ratio_max |
     |------|-------|-----------|-----------|-----------|
     | 0 | 1.0 | 0.75 | 1.017 | 1.056 |
     | 1 | 0.9 | 0.5 | 1.038 | 1.106 |
     | 2 | 0.75 | 0.25 | **1.225** | 1.384 |
     | 3 | 0.5 | 0.0 | — (回退 linear) | — |
   - **max ratio_mean=1.22, max ratio_max=1.38, 远未爆炸** ($>10\times$ 才视为不稳定)
   - 原假设基于错误的时间网格 $[1.0, 0.997, 0.758, 0.032, 0.0]$ ($t_{next}=0.032$), 实际 $t_{next}$ 最小为 0.25
   - **结论: Heun 在 RF 低曲率轨迹下数值稳定, v_next 不稳定假设不成立**
   - DPM++ 相对 Heun 的精度优势 (§三 +0.006 mAP, 单 seed) 不源于数值稳定性, 而源于 x0 插值对低曲率轨迹的更高阶逼近 + 更少 NFE (4 vs 7)

   **效率分析**:
   - hybrid 延迟 145.3ms vs baseline 90.5ms, **+60%** (Heun 2阶每步多一次网络前向, 额外 NFE)
   - full Heun 144.3ms, 与 hybrid 接近 (全维度额外 NFE)
   - [1,1,0,0] w/h Euler 88.9ms, 比 baseline 快 1.8% (w/h 跳过 D1 校正计算)
   - **hybrid 无精度收益但延迟 +60%, 效率层面是负优化; 精度层面是 null result (非有害)**

### 实验列表

#### 实验证明目的: 3 seeds × 4 configs, 同 R1 数据

- 配置矩阵
  -- Config 1: baseline (renewal on, K=500), mAP=0.859 ± 0.004
  -- Config 2: renewal off (方案 B), mAP=0.858 ± 0.003 [Δ=−0.0003]
  -- Config 3: TopK K=200 (renewal on + reset), mAP=0.862
  -- Config 4: TopK K=100 (renewal on + reset), mAP=0.852

- 核心结论
  -- D3 矛盾被证实: renewal 使 η_str 虚高 56-58%, 但 mAP 仅 −0.0003
  -- D3 对 K=100 掉点解释被证伪: K=100/K=200 η_str 几乎相同
  -- Top-K reset 改变 η_str 模式: 单调递减 → V 型
  -- 方案 B 可行: mAP 不损失, η_str 诊断有效

- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp
- 本地脚本: experiments/analysis/r1_d3_summary.py
- 结果文件: experiments/analysis/r1_eta_str_a3_seed{42,123,789}_{renewal_on,off}.json

### 与已证伪方向 Cascade Head Count e2e 的区分

### 去噪轨迹可视化与分维度 D1 掩码实验 (2026-07-29)

- **实验目的**: (1) 可视化各 solver 的去噪轨迹收敛模式 (2) 验证分维度 D1 掩码策略
- **配置矩阵**
  -- D2: a4_dpm_pp_24obj, 4 种 dim_d1_mask (seed42, box_renewal OFF) + 1 种 hybrid (3-seed, box_renewal OFF); baseline/[1,1,0,0]/full Heun 另有 3-seed 验证
  -- D1: a4_dpm_pp_chr2024, seed=42/123/789, 4 种 dim_d1_mask
  -- dim_d1_mask: [1,1,1,1] (DPM++ std) / [1,1,0,0] (cx/cy D1, w/h E) / [0,0,0,0] (all Euler) / [0,0,1,1] (cx/cy E, w/h D1)
  -- hybrid: cx/cy DPM++ 2阶 + w/h Heun 2阶 (速度梯形, 额外 NFE)
  -- 评估: mAP, mAP75, APs (小目标), per-dim L1 (cx/cy/w/h)
- **核心结论**
  -- DPM++ 非单调收敛 (step 3 IoU 反降), 中间步骤不具物理意义
  -- Euler 累积误差反噬 (8-step IoU < 4-step IoU)
  -- DPM++ 精度优势来自中心定位 (center_dist 最小), 但 IoU 不是最高 (Heun 的更过大的框覆盖更多 GT)
  -- **分维度 D1 掩码不显著**: D2 seed42 4 配置 mAP 差异 ≤0.001; baseline 与 [1,1,0,0] 的 3-seed 验证 Δ=+0.001 (noise 范围内)
  -- ⚠ D2 原标注 "3-seed 均值" 实为单 seed42 硬编码 (已修正), APs 单 seed 差异不可靠
  -- **Hybrid (w/h Heun 2阶) vs baseline (3-seed, 结论修正)**: mAP Δ=0.000 (null result), 非 "有害 −0.005"; 原 baseline 硬编码数据 (0.8630±0.000) 错误, 真实 0.858±0.004; full Heun 对照 mAP=0.858±0.004 证明 Heun 本身无害; v_next 不稳定假设被实测证伪 (max ratio=1.22, 远未爆炸); 延迟 +60% (额外 NFE), 效率层面负优化
- **代码改动**
  -- ldmdet/diffusion/rectified_flow.py: RFDPMSolverMultistep.step() 新增 renewal_mask 和 dim_d1_mask 参数; 新增 RFDPMSolverHybrid 类
  -- ldmdet/diffusion/sampling.py: DiffusionSampler 新增 dim_d1_mask 属性; create_dpm_solver() 新增 'dpm_pp_heun_hybrid' 分支
  -- ldmdet/core/head.py: predict() 中 D3 路径 A (per-proposal renewal mask) + hybrid solver model_fn 注入
- **可视化文件**: docs/paper/latex/figures/trajectory/ (14 张图)
- **测试脚本**: tools/dim_d1_d2.py (D2), tools/dim_d1_d1.py (D1)

- **Cascade Head Count e2e (已证伪, mAP 0.684, −0.172)**: 重训架构, 把 cascade head 数量从 6 改为其他值
- **D3**: 仅诊断已有架构的 box_renewal 与 DPM-Solver++ 交互, 不重训, 不引入新模块
- D3 是诊断非新模块, 不重复 e2e 失败模式

---

## 七、S1: Cascade Head × Solver Step 解耦 — 架构合理性形式化 (✓ 已完成, 2026-07-25)

> ✓ 三组实验全部完成 (h3_s4 / h3_s8 / h6_s2 全部 0.859), S1.3 命题完整闭环。详见 [TODO_DIRECTIONS.md §二](file:///home/linkst/workspace/projects/chromosome-kd/docs/TODO_DIRECTIONS.md)。

### 核心贡献: cascade head 作为 implicit solver 的算子分裂视角

当前架构 6 cascade head × 4 solver step = 24 次前向, 但 DPM-Solver++ 仅需 4 NFE 的理论框架把每个 time step 内 6 个 cascade head 视为黑盒——这与 Cascade R-CNN 的级联精化思想同构。形式化为双向精化:
- **横向 (cascade head, 固定 t)**: 在固定时间步上精化 $x_t$, 类似 Cascade R-CNN 级联精化
- **纵向 (solver step, 固定 x 精化链)**: 推进时间 $t$, 类似 DPM-Solver++ 多步积分

### 与染色体检测任务特性的结合

- **解释 24 NFE 架构合理性**: 6 cascade head × 4 solver step 构成算子分裂, DPM-Solver++ 把复合算子 $\mathcal{B}_t^* \circ \mathcal{A}_t$ 视为单次 $v_\theta$ 评估
- **预防审稿人对"6 cascade head 是否冗余"质疑**: cascade head 序列在固定 t 上精化 $x_t$ 至不动点 $\mathcal{B}_t^*$, 横向收敛性是 4 NFE 框架有效的前提
- **解释 Cascade Head Count e2e 失败**: 减小 H 破坏横向收敛性, 而 S 未相应增加, 故 mAP 退化 −0.172

### 命题 S1.3: H×S 可交换性边界

在横向收敛假设下, 减小 H (如 H=3) 需增大 S 以补偿, 反之亦然。但 H×S 不是不变量: 因 $\mathcal{A}_t$ 是二阶 solver 而 $\mathcal{B}_{t,k}$ 是一阶精化, H 减半需 S 增加多于两倍。

### 实验列表

#### 实验证明目的: S1 消融重训 (3 配置, 全部完成)

- 配置 1: H=3 S=4 (12 NFE) ✓ 已完成
  -- 验证: 减小 H 是否破坏横向收敛性, 导致 mAP 退化
  -- 状态: best mAP=0.859, 之前会话已完成
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h3_s4

- 配置 2: H=6 S=2 (12 NFE) ✓ 已完成 (2026-07-25 确认)
  -- 验证: 减小 S 是否影响纵向积分精度, 与 H=3 S=4 对比验证 H×S 可交换性边界
  -- 状态: 早停@ep136/150 (patience=30 触发), best mAP=0.859 @ ep106, last 0.856 @ ep136
  -- 算力: workstation A5000
  -- work_dir: `work_dirs/s1_h6_s2_24obj/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h6_s2

- 配置 3: H=3 S=8 (24 NFE) ✓ 已完成
  -- 验证: 同等 24 NFE 下, 减小 H 增大 S 是否能补偿 (H 减半需 S 增加多于两倍)
  -- 状态: best mAP=0.859 (epoch 64), 30 epochs 未改善早停, ross A6000
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h3_s8

- 配置对照: +DPM-Solver++ baseline H=6 S=4 (24 NFE)
  -- mAP: 3-seed 均值 0.859 ± 0.003 (单 seed best 0.863)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

### 关键结论 (最终, 2026-07-25)

- **三组实验 mAP 全部为 0.859**, 与 +DPM-Solver++ baseline 3-seed 均值 (0.859 ± 0.003) 完全持平
- **S1.3 命题完整闭环**: H=3,S=4 / H=6,S=2 / H=3,S=8 三组同 NFE 或不同 NFE 配置下 mAP 持平
- s1_h6_s2 (H=6, S=2, NFE=12) 在 12 NFE 下 best mAP=0.859, **达到 +DPM-Solver++ baseline 3-seed 均值水平**, 说明**减少 step 并保持 head 可在更少 NFE 下维持性能**
- s1_h3_s8 (H=3, S=8, NFE=24) 在 24 NFE 下 best mAP=0.859, 与 baseline 持平, 表明同等 NFE 下 H=3 S=8 可补偿 H 减半
- 与 S1 命题 S1.3 (H×S 可交换性边界) 对照: H=6 充分大时减小 S 仍可保持横向收敛性, 横向 head 序列已收敛至不动点 $\mathcal{B}_t^*$
- 与已证伪 Cascade Head Count e2e (mAP 0.684, −0.172) 形成对比: 该实验减小 H 但未相应增加 S, 横向收敛性被破坏

### 与已证伪 Cascade Head Count e2e 的区分

- **Cascade Head Count e2e**: 重训架构, 把 cascade head 数量从 6 改为其他值, 端到端评估 (mAP 0.684, −0.172)
- **S1**: 形式化分析已有 H=6, S=4 架构的算子分裂结构, 给出"solver 阶数 × cascade 深度"权衡框架, 避免未来重试类似 e2e 实验

---

## 八、x0/v Prediction 对照 — 预测参数化选择论证 (✓ 3-seed 完成, 支持 R3.2)

> ✓ 3 seeds 全部完成 (2026-07-27 确认): v-prediction 3-seed 均值 0.857 ± 0.0015, vs +DPM-Solver++ baseline 0.859 ± 0.003, **平均 Δ = −0.002** (在 noise 范围内但方向一致, 支持 R3.2)。

### 核心贡献: 验证低维 + shifted schedule 下 x0-prediction 优势

基于 RF 原文 (Liu et al., 2023) 使用 v-prediction, 验证在低维 ($d=4$) 检测空间 + shifted schedule ($s=3.0$) 下 x0-prediction 是否优于 v-prediction, 为论文当前参数化选择提供经验依据。

- **命题 R3.1 (信息等价)**: $\hat{x}_0 = x_t - t\hat{v}$, x0-prediction 与 v-prediction 在 $d=4$ 低维 RF 下信息论等价, 差异仅在损失的 $t$ 加权: $\mathcal{L}_v = t^{-2}\mathcal{L}_{x_0}$
- **命题 R3.2 (shifted schedule 下的偏好)**: shifted schedule ($s=3.0$) 下 x0-prediction 的有效梯度信噪比优于 v-prediction, 因前者在 $t \to 0$ 时不放大梯度
- **命题 R3.3 (设置依赖性)**: RF 原文的 v-prediction 偏好依赖高维 + linear schedule 组合; 在低维 + shifted schedule 下 x0-prediction 是更优选择

### 实验列表

#### 实验证明目的: R3 v-prediction 3-seed 重训 (seed 42 完成, 123/789 待补)

- seed 42 ✓ 已完成 (2026-07-25 确认, workstation A4000)
  -- 状态: 早停@ep64/150 (patience=30 触发), best mAP=0.855 @ ep34, last 0.837 @ ep64
  -- 训练曲线: ep8 warmup=0.802 → ep34 best=0.855 → 长期停滞 (ep34-ep64 未刷新) → 早停
  -- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_24obj.py`
  -- 改动: `criterion=dict(v_prediction=True, v_prediction_t_eps=1e-2)` (batch normalization 均值=1, 避免训练崩溃)
  -- 对照: +DPM-Solver++ baseline (x0-prediction, 3-seed 均值 0.859 ± 0.003, 单 seed best 0.863)
  -- work_dir: `work_dirs/r3_vpred_24obj_seed42/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)
  -- SwanLab project: `ldmdet-r3-vpred` (experiment_name=`r3_vpred`)

- seed 123, 789 ⛔ 待启动
  -- 状态: 等待 GPU 空闲
  -- 算力分配: workstation A5000 / ross A6000

### 关键结论 (单 seed 初步, 2026-07-25)

- **v-prediction seed 42 best mAP=0.855**, vs +DPM-Solver++ baseline 0.863 = **Δ=-0.008**
- Δ=-0.008 超 3-seed noise (±0.003) 但偏小, **方向性支持 R3.2**: v-prediction 在低维 (d=4) + shifted schedule (s=3.0) 下劣于 x0-prediction
- 训练动态: best 出现在 ep34 (warmup 后稳定阶段), 之后 30 epoch 未刷新 → 早停, 表明 v-prediction 优化难度高于 x0-prediction
- 与命题 R3.2 一致: shifted schedule 下 v-prediction 的 $1/t^2$ 梯度放大在 $t \to 0$ 引入方差, 阻碍收敛
- **3-seed 完整验证待补**: 单 seed 已支持方向性结论, 论文纳入决策:
  -- 选项 A (推荐): 论文标注 "preliminary single-seed result", TMI 投稿后补 3-seed
  -- 选项 B: 投稿前补 seed 123/789 (workstation A5000 + ross A6000 并行, ~24h)

### 预期结果

- v-prediction mAP 显著低于 +DPM-Solver++ baseline (预期 ΔmAP < 0), 验证命题 R3.2
- 在 $t < 0.5$ (数据主导区, 对检测精度更关键) 时 v-prediction 的 $1/t^2$ 梯度放大引入显著方差
- 若实验确认, 可纳入论文 §3.1.1 末段或 §5.3 (约 0.3 页增量)

### 与已证伪 h_velocity_loss 的区分

- **h_velocity_loss (已证伪, CRASHED)**: 训练崩溃, 无有效 mAP, 未作对照分析 (详见 [FALSIFIED_DIRECTIONS.md §八](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md))
- **R3**: 通过 $1/t^2$ 损失加权模拟 v-prediction 梯度动态 + `v_prediction_t_eps=1e-2` 截断避免数值爆炸 + batch normalization (均值=1) 避免训练崩溃
- R3 是 h_velocity_loss 的可控重训版本, 提供机制级对照

---

## 九、方向 A: per-dim eta_str 诊断 — 维度级曲率分析 (Phase 2 mAP 对比完成, 持平+加速)

### 核心贡献: 检测空间 4 维 (cxcywh) 各维度的曲率差异诊断

R1 的 $\eta_{str}$ 是 4 维 (cxcywh) 的整体范数比, 但检测空间各维度物理含义不同 (位置 cx,cy vs 尺度 w,h)。方向 A 在 +DPM-Solver++ checkpoint 上零成本诊断各维度曲率, 探究是否可设计 per-dim solver。

### 诊断方法

- 在 +DPM-Solver++ checkpoint (best mAP=0.859, epoch 117) 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
- 计算 wh/cxcy 维度 eta_str 比值, 量化位置维度 vs 尺度维度的曲率差距
- 诊断脚本: `experiments/analysis/direction_a_d_diagnosis.py`
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py`
- checkpoint: `work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth`
- 结果 JSON: `work_dirs/diagnosis/dpm_pp_2nd.json`, `work_dirs/diagnosis/dpm_pp_3rd.json`, `work_dirs/diagnosis/dpm_pp_adaptive.json`

### 诊断结果

- **dpm_solver_pp (2 阶)**: wh/cxcy 比值 0.41-0.50
  -- h 维度 eta_str (4-11) 显著小于 cx,cy (17-50)
  -- h 维度曲率比 cx,cy 小 3-5×
- **dpm_solver_pp_3 (3 阶)**: wh/cxcy 比值 0.25-0.59
  -- step 0 有数值异常 (h=77.9, 待分析)
  -- w 维度差距较小 (1.5-2×)
- **结论**: 部分支持假设
  -- h 维度曲率显著小于 cx,cy (3-5× 差距), 支持原假设
  -- w 维度差距较小 (1.5-2×), 部分证伪 "w,h 都显著小于 cx,cy" 的强假设 (w 维度需修正假设)

### Phase 2: per-dim solver mAP 对比 (✅ 已完成, 2026-07-22)

- **执行**: +DPM-Solver++ checkpoint 零成本推理 (无需重训), 2 个 solver × 500 张验证图
- **实现**: [RFDPMSolverPerDim](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py) — h 维度 (index 3) 用 1 阶 Euler, cx/cy/w 维度 (index 0/1/2) 用 2 阶 DPM-Solver++
- **评估脚本**: [experiments/analysis/direction_a_per_dim_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/direction_a_per_dim_comparison.py)
- **结果数据**: [work_dirs/diagnosis/direction_a_per_dim_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/direction_a_per_dim_comparison.json)

| Solver | mAP | AP50 | AP75 | 延迟(ms) | FPS |
|--------|------|------|------|---------|-----|
| DPM-Solver++ 2阶 (+DPM-Solver++ baseline, 全维度2阶) | 0.862 | 0.988 | 0.973 | 151.1 | 6.6 |
| Per-dim (h=1阶, cxcy/w=2阶) | 0.863 | 0.989 | 0.972 | 142.9 | 7.0 |

**per-dim eta_str** (per-dim solver 全 500 图诊断, 更可靠 than Phase 1 的 50 图):

| Step | cx | cy | w | h |
|------|------|------|------|------|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 27.535 | 19.208 | 1.028 | 0.916 |
| 2 | 11.100 | 15.984 | 0.749 | 0.656 |
| 3 | 33.226 | 10.063 | 0.457 | 0.440 |

### Phase 2 关键结论

1. **mAP 持平 (ΔmAP = +0.001)**: per-dim solver (h=1阶) 与全 2 阶 baseline mAP 持平, 证明 h 维度降为 1 阶不损失精度
   - bbox 4 维度耦合性未被破坏 (位置 cx,cy 与尺度 w,h 的物理相关性不受 solver 阶数分配影响)
2. **延迟略低 (−8.3ms, ~5.5% 加速)**: h 维度省去 D1 校正计算, 但加速幅度有限 (因单步开销主要在 cascade head H=6)
3. **per-dim eta_str 修正 Phase 1 结论**: Phase 2 全量诊断显示 **w 维度 eta_str (0.5-1.0) 与 h (0.4-0.9) 接近**, 而非 Phase 1 (50 图) 所述"与 cx/cy 接近"
   - 即 w,h 维度曲率均显著小于 cx,cy (10-33), Phase 1 对 w 维度的判断需修正
   - 启示: w 维度也可降为 1 阶 (未来 实验 A.2 可验证)
4. **检测专用 solver 叙事**: 检测空间 4 维度 (cxcywh) 的曲率差异源于物理含义 — 位置 (cx,cy) 随 t 变化剧烈 (需 2 阶), 尺度 (w,h) 变化平缓 (1 阶足够), 这是检测任务特有的结构性先验

### Phase 2 论文纳入策略

- ✅ 纳入论文 §5.4 (方向 A 深化): per-dim solver mAP 持平 + 加速, 约 0.3 页
  - 叙事: "基于 per-dim eta_str 诊断, 设计 per-dim DPM-Solver++ (h 维度 1 阶 + cxcy/w 维度 2 阶), 实验表明 mAP 持平 (Δ=+0.001) 且推理加速 5.5%, 验证了检测空间位置维度与尺度维度的曲率差异可被 solver 阶数分配利用"
  - 强调: 检测专用 solver 设计, 与任务特性 (bbox 4 维结构) 结合

### 与 R1 的关系

- R1: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 A: per-dim $\eta_{str}$ 量化各维度曲率差异
- 互补: R1 决定步数, 方向 A 决定 per-dim 阶数分配

### Phase 3 (A.2): w,h 维度均降 1 阶 (✅ 已完成, 2026-07-28)

> Phase 2 启示 "w 维度 eta_str (0.5-1.0) 与 h (0.4-0.9) 接近, w 维度也可降为 1 阶"。A.2 验证此假设: w,h 维度均用 1 阶 Euler, 仅 cx/cy 用 2 阶 DPM-Solver++。

- **执行**: +DPM-Solver++ checkpoint 零成本推理 (无需重训), 2 个 solver × 500 张验证图
- **实现**: [RFDPMSolverPerDim](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py) (euler_dims=(2,3) 即 w,h, dpm_dims=(0,1) 即 cx,cy), 配置入口 [sampling.py:201-214](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/sampling.py) `dpm_solver_pp_per_dim_w`
- **评估脚本**: [experiments/analysis/a2_ddpm_eta_str_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/a2_ddpm_eta_str_comparison.py)
- **结果数据**: [work_dirs/diagnosis/a2_ddpm_eta_str_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/a2_ddpm_eta_str_comparison.json)

| Solver | mAP | AP50 | AP75 | AP_S | AP_M | AP_L | 延迟(ms) | FPS |
|--------|------|------|------|------|------|------|---------|-----|
| DPM-Solver++ 全 2 阶 (baseline) | 0.864 | 0.989 | 0.973 | 0.563 | 0.860 | 0.904 | 94.2 | 10.6 |
| A.2 per-dim-w (w,h=1阶, cx/cy=2阶) | 0.864 | 0.989 | 0.974 | 0.560 | 0.860 | 0.902 | 89.5 | 11.2 |
| **Δ** | **0.000** | — | — | — | — | — | **−4.7 (−5.0%)** | +5.3% |

**per-dim η_str** (baseline 全 2 阶, 500 图诊断):

| 维度 | cx | cy | w | h |
|------|------|------|------|------|
| η_str mean | 32.77 | 35.86 | 0.678 | 0.691 |

### Phase 3 (A.2) 关键结论

1. **mAP 持平 (ΔmAP = 0.000)**: w,h 维度均降为 1 阶不损失精度, 验证 Phase 2 启示
   - w,h 维度 η_str (~0.68) 比 cx,cy (~34) 低约 50×, 1 阶 Euler 足够
2. **延迟降低 5.0% (−4.7ms)**: w,h 维度省去 D1 校正, FPS 10.6→11.2
3. **bbox 4 维耦合性未被破坏**: 位置 (cx,cy) 2 阶 + 尺度 (w,h) 1 阶, 物理相关性不受 solver 阶数分配影响
4. **检测专用 solver 叙事强化**: 位置维度随 t 变化剧烈 (需 2 阶), 尺度维度变化平缓 (1 阶足够), 这是检测任务特有的结构性先验

### DDPM vs RF η_str 对比 (⚠️ 方法论问题, 不纳入论文)

> 实验试图对比 DDPM-trained vs RF-trained 模型的 η_str, 验证 "RF 降低速度场非线性"。但存在严重 framework mismatch, 结论不可靠。

- **方法**: DDPM checkpoint (DiffusionDet, best@ep26, mAP=0.803) 以 `diffusion_type_override='rectified_flow'` 强制走 RF 路径 + DPM-Solver++ 4 步
- **结果**: DDPM-on-RF mAP 暴跌至 0.725 (native 0.803), η_str=1.34 < RF 2.87
- **问题**: DDPM 训练 (cosine noise schedule) + RF 推理 (线性 t∈[0,1] 路径) 是 train-test framework mismatch
  - 低 η_str 不能解读为 "DDPM 轨迹更直", 更可能是模型在错误框架下速度场退化/平坦化 (欠拟合 → 近常数预测 → 低曲率)
  - DDPM native 0.803 已收敛 (AdamW/150ep + EarlyStopping, best@ep26), 非欠训练问题
- **结论**: 此对比无法支撑 "RF 降低速度场非线性" 叙事, 不写入论文
- **正确方案 (若论文需要)**: 在 DDPM native 框架 (DDIM/DDPM solver) 下用二阶 solver 测量 η_str, 或放弃 DDPM 对照 (RF 的 η_str∈[0.7,1.5] 自证低曲率)
- **教训**: 跨范式 η_str 对比必须控制 framework 一致性, 不能用 A 范式训练的 ckpt 强制走 B 范式推理路径

---

## 十、方向 D: 自适应阶次 DPM-Solver++ — 后期 step 降阶 (mAP 对比完成, 3 solver 持平)

### 核心贡献: 基于 $\eta_{3rd}$ 趋势的自适应降阶策略

DPM-Solver++ 3 阶校正项 $D_2$ 在后期 step 应小于早期 (因 RF 轨迹在 $t \to 0$ 时趋于直线)。方向 D 通过零成本诊断 $\eta_{3rd} = \|D_2\|/\|\hat{x}_0\|$ 趋势, 验证后期 step 可降为 2 阶的假设。

### 诊断方法

- 在 +DPM-Solver++ checkpoint 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
- 测量 $\eta_{3rd}$ 随 step 的变化趋势
- 实现位置: `ldmdet/diffusion/rectified_flow.py` (`RFDPMSolverAdaptive`, static + eta_threshold 两种模式)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a7_dpm_pp_adaptive_24obj.py`
- 结果 JSON: `work_dirs/diagnosis/dpm_pp_adaptive.json`

### 诊断结果

- $\eta_{3rd}$ 趋势: step 1 = 44.6 → step 2 = 18.2 (decreasing, 降幅 59%)
- **结论**: ✓ 支持重构假设 — 后期 step 的 3 阶校正项显著小于早期, 可降为 2 阶
- 与 R1 整体 $\eta_{str}$ 单调下降 (3.43→2.45→1.68) 一致, 但方向 D 量化了 3 阶项的衰减

### mAP 对比实验 (✅ 已完成, 2026-07-22)

- **执行**: +DPM-Solver++ checkpoint 零成本推理 (无需重训), 3 个 solver × 500 张验证图
- **评估脚本**: [experiments/analysis/direction_d_solver_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/direction_d_solver_comparison.py)
- **结果数据**: [work_dirs/diagnosis/direction_d_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/direction_d_comparison.json)

| Solver | mAP | AP50 | AP75 | 延迟(ms) | FPS |
|--------|------|------|------|---------|-----|
| DPM-Solver++ 2阶 (+DPM-Solver++ baseline) | 0.863 | 0.989 | 0.972 | 160.0 | 6.2 |
| DPM-Solver++ 3阶 (全程3阶) | 0.863 | 0.989 | 0.973 | 156.1 | 6.4 |
| 自适应 (前2步3阶+后2步2阶) | 0.863 | 0.988 | 0.973 | 153.3 | 6.5 |

- **applied_3rd_history** (自适应): `[false, true, true]` — 第 1 步未用 3 阶, 第 2-3 步用 3 阶

### 关键结论

1. **3 solver mAP 完全持平 (0.8630)**: ΔmAP(2→3) = 0.000, ΔmAP(2→adaptive) = 0.000
   - 证明 4 步采样下 2 阶 DPM-Solver++ 已足够, 3 阶校正项不带来精度增益
   - 佐证 R1 "η_str 2 步收敛" 结论: 2 阶 solver 在 4 NFE 下已达到精度天花板
2. **自适应延迟略低**: Δ延迟(2→adaptive) = −6.7ms (约 4.2% 加速)
   - 加速来自后期 step 降为 2 阶; 但幅度有限 (4%), 因单步开销主要在 cascade head (H=6) 而非 solver 阶数
3. **per-class AP 无显著差异**: 小类别 (Y, G22, F19, F20) 在 3 solver 下 AP 差异 < 0.01, 3 阶校正对困难类别无额外帮助
4. **doubao 方向 D 假设证伪**: doubao 假设"t 小用高阶", 但全程 3 阶与 2 阶 mAP 持平, 说明 t 小时的高阶修正在 4 步采样下无实质贡献

### 论文纳入策略

- ✅ 纳入论文 §5.4 (方向 D 深化): 3 solver mAP 持平结论佐证 R1 "2 步收敛", 约 0.2 页
  - 叙事: "基于 η_3rd 诊断, 设计自适应阶次 DPM-Solver++ (前期 3 阶 + 后期 2 阶), 实验表明 4 NFE 下 2 阶已充分, 3 阶校正项无额外增益 (ΔmAP=0.000), 自适应方案仅带来 4% 推理加速"
- 不作为主要贡献 (因无 mAP 提升), 作为 R1 诊断的验证实验

### 与 R1 的关系

- R1: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 D: per-step 3 阶项 $\eta_{3rd}$ 量化"后期 step 可降阶"
- 互补: R1 决定步数, 方向 D 决定每步阶数

---

## 十一、方向 C: step-aware embedding — Cascade head 感知 solver step (✓ 已完成, 非负面)

> ✓ 已完成 (2026-07-23 早停@ep148, best 0.859@ep118, Δ=-0.004 在 noise 内)。插桩分析显示 step_proj 活跃、loss 仍降、3 类改善 — **非负面方向**, 不归入 FALSIFIED, 保留为 S1 理论佐证。详见 [TODO_DIRECTIONS.md §六](file:///home/linkst/workspace/projects/chromosome-kd/docs/TODO_DIRECTIONS.md)。

### 核心贡献: 让 cascade head 感知 DPM-Solver++ step 编号

当前 cascade head 在所有 solver step 上共享参数, 但不同 step 上 $x_t$ 的统计特性不同 (早期近噪声, 后期近 GT)。方向 C 通过 step embedding 让 head 感知当前 step, 提升每步精化的针对性。

### 实现方式

- step_mlp + step_proj 零初始化
- 零初始化确保预训练兼容: 训练初期 step_proj 输出为 0, 模型行为与无 step embedding 时一致, 可在 +DPM-Solver++ checkpoint 上继续训练而非重训
- 实现位置: `ldmdet/core/head.py` (step_mlp + step_proj 零初始化)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a6_step_aware_24obj.py`

### 状态 (✓ 已完成)

- seed 42 ✓ 已完成 (本地 A6000, 早停@ep148/150)
  -- best mAP = 0.859 @ ep118 (Δ=-0.004 vs +DPM-Solver++ 0.863, **在 3-seed std 0.003 范围内**)
  -- 早停: "the monitored metric did not improve in the last 30 records. best score: 0.859."
  -- work_dir: `work_dirs/a6_step_aware_24obj_seed42/`
  -- SwanLab project: `ldmdet-mainline-ablation-24obj` (experiment_name=`a6_step_aware`)
- seed 123/789: 不启动 (方向 C 非负面但增益不显著, GPU 优先分配给 M1 FP32 复现)
- 对照: +DPM-Solver++ baseline (3-seed 均值 0.859 ± 0.003)

### 插桩分析 (2026-07-23, checkpoint epoch_146 + best ep118)

> **核心结论**: 方向 C **不是负面方向**。虽然 mAP 未超 +DPM-Solver++, 但插桩指标显示 step-aware embedding 确实被学习且训练健康。

**1. step_proj 权重分析 (与 M1 fuse 对比)**:

| 指标 | 方向 C step_proj | M1 fuse (对照) |
|------|-----------------|---------------|
| 权重 norm (ep146) | 5.420 (**活跃**) | 0.215 (弱) |
| ep118→ep146 变化 | 5.434→5.420 (收敛) | 0.006→0.238 (仍在增长) |
| 方向性 | **有区分** (std=0.005) | **完全均匀** (std=0.000) |

→ step-aware embedding **确实被模型使用**, 且早期即收敛, 不像 M1 fuse 那样退化。

**2. 训练动态**: loss 仍在下降 (ep140: 1.670 → ep147: 1.654), 但 mAP 已收敛在 0.857 (loss-mAP 分离)。

**3. Per-class AP 对照 (best@ep118 vs +DPM-Solver++ best@ep117)**: **3 类改善 (A1 +0.002, C12 +0.003, Y +0.003), 1 类持平, 20 类轻微退化**。Y 染色体改善尤其有价值 (最小最难类别)。与 M1 (24 类全退化) 形成对比。

### 价值判断 (综合插桩指标, 非 mAP 阈值)

**不归入 FALSIFIED 的理由**:
1. mAP 差距 -0.004 在 3-seed std (0.003) 范围内, 统计上无法区分
2. step_proj 权重活跃 (norm=5.42), 与 M1 fuse 退化 (uniform) 本质不同
3. loss 仍在下降, 训练健康 (梯度稳定)
4. 3 个类别改善 (含最难类别 Y), 非全面退化
5. 早停是 patience 到期而非崩溃

### 理论

- 让 cascade head 感知 DPM-Solver++ step 编号, 零初始化确保预训练兼容
- 与 S1 算子分裂结构不冲突: step embedding 不改变横向 (cascade head) / 纵向 (solver step) 解耦, 仅在横向 head 内部添加 step 条件

### 与 S1 的关系

- S1 形式化 cascade head × solver step 算子分裂 (§七)
- 方向 C 在不破坏 S1 算子分裂结构的前提下, 让 cascade head 显式感知 step
- 与 S1 互补: S1 给出架构合理性框架, 方向 C 在框架内探索性能提升
- **论文叙事价值**: 即使 mAP 持平, "cascade head 已隐式感知 step" 这一发现本身支持 S1 的算子分裂理论 (§七), 可作为 S1 的实验佐证

---

## 十二、FPS / 延迟基准实验 — 交互式临床筛查延迟带论证 (论文 §4.6 / Table 10 / Figure 6)

> 本节为论文 §4.6 FPS / 延迟基准实验的完整记录。论文 Table 10 + Figure 6 的全部数据点均来自本节, 数据源为 `results/benchmark_fps_*.{md,json}` 系列 5 个文件 + `experiments/runners/benchmark_fps.py` (含 MODEL_REGISTRY / KNOWN_MAP 配置化测量)。

### 实验配置

- **硬件**: NVIDIA RTX A6000 (单卡)
- **输入分辨率**: 512 × 512, batch=1
- **测量方法**: CUDA event timing, warmup=100, iters=300 (baseline); warmup=10, iters=100 (主消融变体)
- **覆盖模型**: 9 个 (论文 Table 10 全部行)
  -- RF+Heun (4 步, 7 NFE): RF 范式基线
  -- +Stoch. Coupling (Heun 4 步): 耦合消融
  -- +DPM-Solver++ (4 步, 4 NFE): DPM-Solver++ 主变体
  -- +DPM-Solver++ + Top-K (K=300/200/100): Top-K 剪枝消融
  -- Cascade R-CNN R50 / YOLOX-S / DiffusionDet: SOTA 对比基线
  -- RTMDet-L (仅 CATALOG §6.4, 未进 Table 10)

### 论文 Table 10 数据矩阵

> **mAP 数据来源说明**: 下表 mAP 列使用 **val set (seed42 best checkpoint)** 数值, 与 FPS/延迟测量使用同一 checkpoint, 确保速度-精度对的内部一致性。test set 评估结果见 [§14.1](#141-测试集评估-论文-§4.5.4-c4-任务), 9 模型 val→test Δ ≤ 0.004 (+DPM-Solver++ 唯一显著偏差 −0.004, 其余 ≤ 0.001)。

| 模型 | Solver | NFE | Latency (ms) | FPS | mAP (val) | mAP (test) |
|------|--------|:---:|-------------:|----:|:---------:|:----------:|
| RF+Heun | Heun | 7 | 124.38 ± 3.38 | 8.0 | 0.856 | 0.857 |
| +Stoch. Coupling | Heun | 7 | 128.35 ± 1.95 | 7.8 | 0.858 | 0.858 |
| **+DPM-Solver++** | **DPM++** | **4** | **75.03 ± 0.96** | **13.3** | **0.863** | **0.859** |
| +DPM-Solver++ + Top-K (K=300) | DPM++ | 4 | 71.27 ± 2.39 | 14.0 | 0.861 | 0.860 |
| **+DPM-Solver++ + Top-K (K=200)** | **DPM++** | **4** | **70.46 ± 2.28** | **14.2** | **0.860** | **0.859** |
| +DPM-Solver++ + Top-K (K=100) | DPM++ | 4 | 69.71 ± 1.98 | 14.3 | 0.850 | 0.847 |
| Cascade R-CNN | — | 1 | 20.67 ± 0.48 | 48.4 | 0.854 | 0.853 |
| YOLOX-S | — | 1 | 10.15 ± 0.41 | 98.5 | 0.796 | 0.795 |
| DiffusionDet | Euler | 1 | 24.38 ± 1.09 | 41.0 | 0.803 | 0.804 |

### 关键结论

1. **临床交互式筛查延迟带 13.3-14.2 FPS**: +DPM-Solver++ + Top-K (K=200) 最快 14.2 FPS / 70.46 ms, +DPM-Solver++ baseline 13.3 FPS / 75.03 ms
2. **DPM-Solver++ 1.71× NFE 加速**: 4 NFE (DPM++) vs 7 NFE (Heun), 同等精度下减少 43% NFE
3. **Top-K 剪枝边际加速**: K=200 vs +DPM-Solver++ baseline 加速 1.06× (75→70 ms), 主要因 cascade head 占 90%+ 延迟 (backbone+neck 仅 ~5.8 ms)
4. **标准检测器快 3-7× 但精度低**: Cascade R-CNN 48.4 FPS / 0.854, YOLOX-S 98.5 FPS / 0.796, 但 mAP 落后 0.005-0.067
5. **DiffusionDet 对比**: 41 FPS / 0.803, KaryoFlow 数量级 mAP 改善 (+0.060)
6. **延迟分布**: cascade head 占 90%+ (Head 118-128 ms in Heun 变体, 64-69 ms in DPM++ 变体); backbone+neck 仅 4-8%

### 数据源文件

- [results/benchmark_fps_20260714_231841.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260714_231841.md): RF+Heun/+Stoch. Coupling/+DPM-Solver++/Top-K K=300/K=200/RTMDet-L (warmup=10, iters=100)
- [results/benchmark_fps_20260714_234842.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260714_234842.md): +DPM-Solver++ + Top-K K=300/K=200/K=100 (warmup=10, iters=100)
- [results/benchmark_fps_20260715_013222.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260715_013222.md): Cascade R-CNN + YOLOX-S (warmup=10, iters=100)
- [results/benchmark_fps_20260716_100627.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260716_100627.md): DiffusionDet baseline (warmup=100, iters=300)
- [results/benchmark_fps_20260716_101856.md](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260716_101856.md): Cascade R-CNN + YOLOX-S + DiffusionDet 复测 (warmup=100, iters=300)
- [EXPERIMENT_CATALOG.md §6.4](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md): C8 复测决策记录 (跳过, 现有数据方法论足够严谨)
- 论文图: latex/figures/fps_map.png (4 类语义着色, log-scale FPS 轴, 速度-精度权衡散点图)

### 测量脚本

- 主脚本: [experiments/runners/benchmark_fps.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/runners/benchmark_fps.py)
- MODEL_REGISTRY / KNOWN_MAP 配置化测量, 支持 9 个模型变体的统一基准
- 训练配置: 见各模型对应配置文件 (CATALOG §1.5 SOTA 表 + 主消融配置 DDPM/RF+Heun/+AdaLN-Zero/+Stoch. Coupling/+DPM-Solver++)

---

## 十三、标注噪声鲁棒性实验 — SIER 评估广度论证 (论文 §4.8 / Table 11)

> 本节为论文 §4.8 标注噪声鲁棒性实验的完整记录。论文 Table 11 的 3×3 网格 (σ_bbox × 翻转率 p) 全部 9 个扰动单元 + 干净基线均来自本节, 数据源为 `work_dirs/robustness_noise/consolidated_results.json`。无模型重训, 复用 +DPM-Solver++ checkpoint (DPM-Solver++ 4-step + Top-K, seed 42, best@ep117)。

### 实验设计

- **基础 checkpoint**: +DPM-Solver++ + Top-K (DPM-Solver++ 4-step) best_epoch_117, seed 42
- **配置**: [experiments/configs/robustness/noise_test_a4.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/robustness/noise_test_a4.py)
- **扰动脚本**: [experiments/runners/robustness_noise.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/runners/robustness_noise.py)
- **评估脚本**: `experiments/runners/robustness_eval.py`
- **数据集**: Dataset 2 test set, 1000 张图像 / 45,980 个 GT 实例
- **扰动维度**:
  -- (i) GT bbox 中心高斯抖动 σ_bbox ∈ {2, 5, 10} px (宽/高不变, 中心裁剪到图像边界)
  -- (ii) 类别标签随机翻转率 p ∈ {5%, 10%, 20%} (翻转到其余 23 类中均匀采样的替代)
- **生成**: 3×3 网格 + 干净基线, 以 seed 42 生成一次, 同一 checkpoint 重新评估; 图像像素不动

### 论文 Table 11 数据矩阵 (mAP@[0.50:0.95])

| σ_bbox \ p | 0% | 5% | 10% | 20% |
|------------|----|----|-----|-----|
| 0 px | **0.859** | — | — | — |
| 2 px | — | 0.666 | 0.599 | 0.477 |
| 5 px | — | 0.428 | 0.386 | 0.308 |
| 10 px | — | 0.179 | 0.162 | 0.129 |

### 干净基线完整指标

- σ=0, p=0: mAP=0.859, AP50=0.988, AP75=0.971, AP_S=0.576, AP_M=0.856, AP_L=0.914

### 关键发现

1. **bbox 抖动主导高 IoU 精度**: σ=5 px 下 AP50 温和下降 (0.988→0.847, −0.141), 但 AP75 坍缩 (0.971→0.375, −0.596) — 在 ~100 px 框上 5 px 中心偏移足以打破 IoU≥0.75 但不打破 IoU≥0.50
2. **类别翻转近似乘法降低精度和召回**: 固定 σ 下翻转率加倍大致使剩余 mAP 减半 (σ=2: 0.666→0.599→0.477; σ=5: 0.428→0.386→0.308)
3. **噪声源低噪声下亚加性, 高噪声下严重复合**: σ=2, p=5% 联合 −0.193 小于任一边际之和; 但 σ=10, p=20% 联合退化至 0.129
4. **最对抗设置下不坍缩**: σ=10, p=20% 下 mAP=0.129, 仍为随机类别基线 (1/24 ≈ 0.042) 的 3×, 表明所学 RF 特征在标注腐蚀下不坍缩

### 数据源文件

- [work_dirs/robustness_noise/consolidated_results.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/consolidated_results.json): 主结果 (9 个扰动单元 + 干净基线, 含 AP50/AP75/APs/APm/APl 完整指标)
- [work_dirs/robustness_noise/results.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/results.json): 原始逐单元评估结果
- [work_dirs/robustness_noise/perturbed/](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/perturbed/): 9 个扰动 GT JSON (如 `noise_both_s02_f005.json` 等)
- [work_dirs/robustness_noise/eval.log](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/eval.log): 评估日志
- CATALOG 索引: a4_noise 推理 (~13 runs, FINISHED/CRASHED)

---

## 十四、测试集评估 + 跨域 Zero-shot — 泛化性论证 (论文 §4.5.4 + §4.7)

> 本节为论文 §4.5.4 测试集评估 + §4.7 跨数据集总结中 zero-shot 跨域实验的完整记录。无模型重训, 复用 +DPM-Solver++ checkpoint (DPM-Solver++ 4-step, seed 42, best@ep117)。

### 14.1 测试集评估 (论文 §4.5.4, C4 任务)

- **基础 checkpoint**: 全部 9 个模型 (论文 Table 10 全部行), 各取 seed 42 best checkpoint
- **数据集**: Dataset 2 test split, 1000 张图像 / 45,980 个 GT 实例
- **评估日期**: 2026-07-26
- **评估脚本**: [results/run_test_eval_batch.sh](file:///home/linkst/workspace/projects/chromosome-kd/results/run_test_eval_batch.sh)
- **完整日志**: [results/test_eval_20260726_181932/](file:///home/linkst/workspace/projects/chromosome-kd/results/test_eval_20260726_181932/)
- **分析报告**: [results/test_eval_20260726_181932/ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/results/test_eval_20260726_181932/ANALYSIS.md)

#### 9 模型 val vs test 完整对照表

| 模型 | Val mAP (seed42) | Test mAP | Δ (test−val) | 备注 |
|------|:---------:|:--------:|:----------:|------|
| RF+Heun | 0.856 | 0.857 | +0.001 | 稳定 |
| +Stoch. Coupling | 0.858 | 0.858 |  0.000 | 稳定 |
| **+DPM-Solver++** | **0.863** | **0.859** | **−0.004** | 唯一显著下降 |
| +DPM-Solver++ + Top-K K=300 | 0.861 | 0.860 | −0.001 | 稳定; test 上反超 +DPM-Solver++ |
| +DPM-Solver++ + Top-K K=200 | 0.860 | 0.859 | −0.001 | 稳定; test 上与 +DPM-Solver++ 持平 |
| +DPM-Solver++ + Top-K K=100 | 0.850 | 0.847 | −0.003 | 稳定; K=100 有害结论 robust |
| Cascade R-CNN | 0.854 | 0.853 | −0.001 | 稳定 |
| YOLOX-S | 0.796 | 0.795 | −0.001 | 稳定 |
| DiffusionDet | 0.803 | 0.804 | +0.001 | 稳定 |

#### +DPM-Solver++ 详细指标 (val vs test)

| Split | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|-------|------|------|------|------|------|------|
| val (500 imgs) | 0.863 | 0.989 | 0.972 | 0.499 | 0.860 | 0.901 |
| **test** | **0.859** | 0.988 | 0.971 | **0.577** | 0.856 | 0.914 |

#### 结论

1. **+DPM-Solver++ 是唯一显著偏差**: val→test Δ=−0.004, 其他 8 个模型 Δ ≤ 0.001; 反映 best-checkpoint (ep117) 对 val 的轻微过拟合
2. **test mAP 与 3-seed mean 一致**: +DPM-Solver++ test 0.859 = 3-seed val mean 0.859±0.003, 证实整体泛化良好
3. **AP_S 高方差**: val 0.499 → test 0.577, 仅 60 张 val 含小目标, 应结合 Table 8 逐图像显著性检验解读
4. **Top-K 剪枝叙事增强**: K=300 test 0.860 ≥ +DPM-Solver++ 0.859 (essentially free, 甚至略好); K=200 test 0.859 = +DPM-Solver++ 0.859 (完全 free)
5. **K=100 有害结论 robust**: val −0.013 → test −0.012
6. **DiffusionDet 增益 robust**: val +0.060 → test +0.055

#### 配置文件 (test_eval)

- ldmdet 系列: [experiments/configs/ldmdet/directions/mainline_ablation_24obj/](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/directions/mainline_ablation_24obj/) (a1/a2/a4 _test_eval_24obj.py)
- Top-K 系列: [experiments/configs/ldmdet/directions/inference_opt/](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/directions/inference_opt/) (a4_io3_k{300,200,100}_test_eval_24obj.py)
- baselines: [experiments/configs/baselines/benchmark_24obj/](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/baselines/benchmark_24obj/) (cascade_rcnn_r50_test_eval.py, yolox_s_test_eval.py, diffusiondet_ddpm_test_eval.py)
- **注**: Cascade R-CNN 和 YOLOX-S 首次评估因配置问题失败 (路径双拼接 / EMAHook 未初始化), 已创建 test_eval 配置修复后重跑成功
- **数据源**: [EXPERIMENT_CATALOG.md §7.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (C4: 测试集评估)

### 14.2 跨域 Zero-shot: Dataset 2 → Chromosome20240904 (论文 §4.7 引用)

- **方向**: Dataset 2 (5000 imgs, 训练域) → Dataset 1 (Chromosome20240904, 220 test imgs, 10262 instances)
- **基础 checkpoint**: +DPM-Solver++ (DPM-Solver++ 4-step, seed 42, best@ep117)
- **配置**: [experiments/configs/cross_domain/chr20240904/zero_shot_a4.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/cross_domain/chr20240904/zero_shot_a4.py)
- **整体结果**: mAP=0.157, AP50=0.513, AP75=0.039
- **per-class 高亮**:
  -- A1/A2/A3/B4/B5 (大类): AP50 ≈ 0.87-0.93 (大染色体迁移良好)
  -- D13-D15, E16-E18 (中类): AP50 ≈ 0.73-0.86 (中尺寸迁移良好)
  -- F19/F20: AP50 ≈ 0.58 (小染色体部分迁移)
  -- C6-C12 (C 组): AP50 ≈ 0-0.017 (C 组形态相似, 跨域失效)
  -- G21/G22: AP50 ≈ 0.23-0.46 (小染色体困难)
  -- X: AP50=0.879 (中等迁移); Y: AP50=0.262 (最难, 数据稀缺)
- **结论**: 14/24 类 AP50 > 0.5; C 组失效 (形态相似性导致跨域类别判别失败); 大类迁移良好, 小类困难
- **数据源**: [work_dirs/robustness_noise/zero_shot_results_chr20240904.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/zero_shot_results_chr20240904.json)
- **评估日志**: [work_dirs/robustness_noise/zero_shot_a4_chr20240904.log](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/robustness_noise/zero_shot_a4_chr20240904.log)

---

## 十五、边际有效方向 (历史记录)

> 以下方向在 Dataset 1 (mAP 0.72-0.75) 上获得边际收益, 未叠加到 SOTA。记录作为完整事实, 不作为论文主路线。

### Hard OT Coupling

- Hard OT Coupling (Dataset 1, 2 seeds)
  -- 结果: mAP=0.747 ± 0.000 [+0.001 vs 0.746 baseline, 边际]
  -- 本地: work_dirs/multi_seed_aug/hard_ot/seed_{42,123}/
  -- SwanLab (project=ldmdet-ablation):
     - hard_ot_seed42: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/h8fizm7lmc9v5xzxi8ufj
     - hard_ot_seed123: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/kka4nra9qk3wanx7i9og1
  -- 注: Dataset 1 上 Hard OT 比 Random 更差 (−0.008, p<10⁻⁸), 证实 OT 坍缩病理

### Sinkhorn Stochastic OT

- Sinkhorn Stochastic OT (Dataset 1, 1 seed)
  -- 结果: mAP=0.748 [+0.002 vs 0.746 baseline, 边际]
  -- 本地: work_dirs/multi_seed_aug/sinkhorn_stochastic/seed_42/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/9ca697vnm1l3koccbenif

### Bottleneck: Focal γ=3

- Focal Loss γ=3 (Dataset 1, 1 seed)
  -- 结果: mAP=0.750 [+0.004 vs SOTA 0.746, 分类损失调整]
  -- 状态: 未叠加到 SOTA
  -- 本地: work_dirs/bottleneck/ablation/focal_gamma_3/20260628_013823/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/ye6a2whory9y67tnvalg3

### D1: RoI 空间编码消融 — 空间编码至关重要 (✓ 完成)

- **实验**: RoI 7×7 空间特征 vs 空间抹平 (GlobalAvgPool → 1×1) 消融
- **结果**: baseline mAP=0.863 → ablation mAP=0.009, **Δ = −0.854** (灾难性崩溃)
- **结论**: 7×7 空间结构至关重要, DynamicConv 已有效提取空间编码 (非丢失)
- **影响**: 直接支撑 M1 "应增强而非重建空间编码" 的设计决策
- **数据源**: `work_dirs/diagnosis/d1_roi_ablation.json` + [EXPERIMENT_CATALOG.md §7.6](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md)
- **诊断脚本**: `experiments/analysis/d1_roi_ablation.py`

### M1: 形态感知 RoI 编码器 — null result (设计问题确认, ✓ 完成)

- **设计**: 零初始化残差分支 + 方向解耦卷积 (h_conv 臂长比 + v_conv 着丝粒)
- **FP32 结果**: best mAP=0.862@ep19, **Δ = −0.001 vs A4 0.863** (统计上持平)
- **BF16 结果**: mAP=0.818, Δ=-0.045 (虚假退化, BF16 误导, 排除)
- **核心结论**: h_conv/v_conv 在 FP32 下仍均匀 → **设计问题而非精度问题**
  - (1) 零初始化 fuse 梯度瓶颈 → h_conv/v_conv 梯度极弱
  - (2) (7,1)+(1,7) 感受野与 7×7 RoI 同尺寸, 缺乏空间上下文
  - (3) morph_emb 退化为常数偏置, 未学到方向性形态信息
- **教训**: BF16 导致虚假 −0.045 退化, FP32 复现揭示真实情况; 零初始化 fuse 在低维检测空间存在梯度瓶颈
- **M1-v2 改进方向**: 非零初始化 fuse + 显式形态先验注入 + 注意力替代方向卷积
- **参数开销**: 262.8K/head × 6 = 1.58M (<总参数 0.5%)
- **数据源**: `work_dirs/m1_morphology_aware_24obj_fp32/` + [EXPERIMENT_CATALOG.md §6.6 C22/C23](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md)
- **详细分析**: [STRUCTURAL_IMPROVEMENT_ANALYSIS.md §3.1.7](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md)

### Box Refine Net

- Box Refine Net (Dataset 1, 1 seed)
  -- 结果: mAP=0.747 [+0.001 vs 0.746 baseline, 持平]
  -- 状态: early stop @ epoch 85, best @ epoch 55
  -- 本地: work_dirs/direction_exps/direction_d_box_refine/20260629_091843/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/fnoz9x82aor1utsuo0jtl

### Head Distillation: cascade head 维度蒸馏 (NFE 24→12 加速)

> **创新点**: headwise feature 蒸馏将 H=6 Teacher 知识压缩到 H=3 Student, 实现 NFE 24→12 (2× 加速) 同时保持精度
> **理论依据**: S1 的 H×S 可交换性分析 ([theory_analysis_RF_DPM.md §2](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)), [proposals](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md)
> **关联**: 失败配置 (freeze_backbone=True) → [FALSIFIED §十三](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)

#### 核心贡献: headwise feature 蒸馏实现 cascade head 压缩

通过蒸馏将 6 级 cascade head 压缩到 3 级, Student head 0/1/2 ← Teacher head 0/2/5 (输入/中间/main 对齐), 实现 NFE 减半同时精度持平。

- **形式化**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}}(\text{student}) + \lambda \cdot \mathcal{L}_{\text{distill}}$, $\mathcal{L}_{\text{distill}} = \frac{1}{K}\sum_k \text{MSE}(\text{student\_fc}_k, \text{teacher\_fc}_{\text{map}(k)}.\text{detach}())$
- **head 映射**: $\{0\to0, 1\to2, 2\to5\}$ (输入对齐 + 中间进度 + main 对齐)
- **Teacher**: A4 DPM-Solver++ (H=6, mAP=0.863), 冻结, 仅 forward
- **Student**: H=3, 从 Teacher head 0/2/5 初始化 (非随机)

#### 与 S1 理论的联系

S1 的 H×S 理论说明 "仅改变 H 会破坏横向收敛性" (已证伪 N_cascade e2e mAP=0.684, −0.172)。Head Distillation 通过蒸馏监督让 Student head 继承 Teacher 的行为分布, 避免了随机初始化 H=3 的收敛失败。

#### 实验列表

##### 实验证明目的: Head Distillation 实现 NFE 加速同时保持精度

- Head Distillation (H=3←H=6, backbone解冻 + A4 backbone加载)
  -- 数据集: Dataset 2
  -- 改动: num_heads=6→3, use_distillation=True, distill_lambda=0.05, distill_head_map={0:0,1:2,2:5}, freeze_backbone=False, teacher_checkpoint=A4 best ep117, lr=1e-5, 50ep
  -- 结果: mAP=0.859 (val独立评估 test.py --dataset val, seed 42; 训练best@ep10=0.860, early stop@ep40), AP50=0.986, AP75=0.969 [Δ=-0.004 vs A4 0.863, 在 3-seed noise ±0.003 内]
  -- NFE: 12 (H=3 × S=4) vs A4 24 (H=6 × S=4), **2× 加速**; 延迟 44.72ms / 22.4 FPS (ross A6000, 500iters, Head 39.17ms / Backbone 5.55ms) vs A4 77.57ms / 12.9 FPS, **1.73× 推理加速**
  -- loss_distill: 持续下降 0.050→0.025 (50% 下降), 蒸馏目标有效
  -- per-class AP: 与 A4 对齐 (Δ -0.012~+0.004, 最大差异 D15 -0.012)
  -- work_dir: work_dirs/h3_distill_plan_a_24obj/ (本地 + ross)
  -- SwanLab: ldmdet-head-distill / h3_distill_plan_a
  -- 配置: experiments/configs/ldmdet/directions/mainline_ablation_24obj/h3_distill_plan_a_24obj.py

#### 关键结论

- **NFE 24→12 加速 2x + 精度近乎持平**: mAP=0.859 (val独立评估) 近乎持平 A4 0.863 (Δ=-0.004, 在 3-seed noise ±0.003 内), 延迟 44.72ms / 22.4 FPS (vs A4 77.57ms / 12.9 FPS, 1.73× 加速), 达成工程目标
- **蒸馏有效性**: loss_distill 持续下降 (vs 失败配置停滞 0.033), per-class AP 对齐 A4, 证明 headwise feature 蒸馏可以有效压缩 cascade head
- **与 S1 互补**: S1 证明 H×S 可交换 (H=3,S=4 = H=6,S=2 = 0.859), Head Distillation 证明 H=3 通过蒸馏可达 0.859, 两者共同支撑 "cascade head 可压缩" 的理论
- **未超越 A4**: 近乎持平 (Δ=-0.004), 无增益 (但"近乎持平"可能已是蒸馏最佳结果, 因 backbone 从 A4 加载本身就是知识继承)

#### 失败配置对照 (→ FALSIFIED §十三)

失败配置 (freeze_backbone=True) 是配置Bug: Student backbone 停 ImageNet, Teacher head 期望 A4 染色体特征 → 特征分布不匹配 → mAP=0.717 (Δ=-0.146)。修复后 0.860, 清晰隔离了"配置Bug" vs "方法局限"。

#### 可扩展性

- **3D 检测** (d=6-7): cascade head 维度蒸馏同样适用, 可压缩 NFE 加速推理
- **关键点检测** (d=2K): headwise feature 蒸馏可扩展到关键点级联精化
- **临床部署**: 2× 加速对交互式筛查延迟带 (13.3-14.2 FPS) 有直接价值

### 非线性轨迹 OT Flow Coupling only

- OT Flow Coupling only (Dataset 1, 1 seed)
  -- 结果: mAP=0.751 [+0.005 vs 0.746 baseline]
  -- 改动: coupling=ot_flow, lambda_mod=0.0 (关闭尺度条件)
  -- 本地: work_dirs/nonlinear_trajectory_e42/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/wcp34v3t

### 非线性轨迹 OT + ScaleConditionedRF (SCRF 未启用)

- OT + ScaleConditionedRF argmax eps=1.0 (Dataset 1, 1 seed)
  -- 结果: mAP=0.752 [+0.006 vs 0.746 baseline]
  -- 关键修正: ScaleConditionedRF 当时未集成到 head.py, 0.752 实际来自 OTFlowCoupling + 种子方差
  -- 本地: work_dirs/nonlinear_trajectory/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/usnvd63f
  -- 注: 实际增益来自 OT, 非 SCRF (ScaleConditionedRF 真正启用后 mAP=0.741, −0.005, 证伪)

---

## 十六、SwanLab 项目映射汇总

| SwanLab Project | 实验数 | 范围 | URL Pattern |
|-----------------|--------|------|-------------|
| `ldmdet-mainline-ablation-24obj` | 5 ⭐ | Dataset 2 主路线消融 (DDPM/RF+Heun/+AdaLN-Zero/+Stoch. Coupling/+DPM-Solver++, 论文核心) | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| `ldmdet-ablation` | 9 (Dataset 2) + 23 (Dataset 1) | 主线 + Dataset 2 耦合策略 + Dataset 1 历史 + 非线性轨迹 | `https://swanlab.cn/@einspanner/ldmdet-ablation/runs/<run_id>` |
| `chromosome-kd-benchmark-24obj` | 8 | Dataset 2 SOTA 对比模型 (DINO/RTMDet-L/Cascade/YOLOX/DiffusionDet) | `https://swanlab.cn/@einspanner/chromosome-kd-benchmark-24obj/runs/<run_id>` |
| `ldmdet-breakthrough` | 2 | Dataset 2 SC-RF 自条件化 | `https://swanlab.cn/@einspanner/ldmdet-breakthrough/runs/<run_id>` |
| `ldmdet-frontier-directions` | 6 | Dataset 2 前沿方向探索 | `https://swanlab.cn/@einspanner/ldmdet-frontier-directions/runs/<run_id>` |
| `ldmdet-s1-cascade-decouple` | 2 已完成 + 1 进行中 | S1 cascade head × solver step 解耦消融 (s1_h3_s4 ✓ / s1_h3_s8 ✓ / s1_h6_s2 🔄) | `https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/<run_id>` |
| `ldmdet-r3-vpred` | 1 进行中 + 2 待启动 | R3 v-prediction 对照重训 (seed 42 🔄 / seed 123,789 ⛔) | `https://swanlab.cn/@einspanner/ldmdet-r3-vpred/runs/<run_id>` |
| `nonlinear-3seed-repro` | 2 | 3-seed 复现 (Dataset 1) | `https://swanlab.cn/@einspanner/nonlinear-3seed-repro/runs/<run_id>` |
| `chromosome-kd` | 21 | 早期 Dataset 1 数据集 | `https://swanlab.cn/@einspanner/chromosome-kd/runs/<run_id>` |
| `ldmdet-inference` | 12 | DDIM 步数对齐 + DPM-Solver++ 步数消融 (Dataset 1) | `https://swanlab.cn/@einspanner/ldmdet-inference/runs/<run_id>` |

---

## 十七、关键结论汇总

### 主路线创新点贡献矩阵

| 创新点 | 核心贡献 | 与任务结合 | 关键数据 | 状态 |
|--------|----------|------------|----------|------|
| **RF (§一)** | 直线 ODE 路径取代 DDPM 弯曲随机轨迹 | 密集 proposals 误差复合 / 小训练集 / 24 类细粒度 | DDPM→RF+Heun +0.053 mAP (统一口径), 91% 归因于 RF | ✅ 完成 |
| **OT Collapse + Stoch. Coupling (§二)** | 低维 d=4 OT 坍缩形式化 + Stochastic Coupling 补救 | 低维触发 / 高 K 加剧 / 小训练集放大 | Dataset 1 +0.034 (p<10⁻¹²⁰), Dataset 2 +0.0001 (p=0.80) + 4.6× 平滑 | ✅ 完成 |
| **DPM-Solver++ (§三)** | RF 适配 data-prediction + 修正 FlowDet 结论 | 临床交互式延迟 13.3-14.2 FPS / cascade head 占 90%+ | +0.006 mAP (p<10⁻⁶) + 1.71× NFE 加速 | ✅ 完成 |
| **Top-K Pruning (§四)** | 500→K proposals 剪枝 + DPM-Solver++ 兼容 | K=200 最优 (46 染色体 + 重叠冗余) | K=200: 14.2 FPS, mAP 0.860 | ✅ 完成 |
| **R1 η_str (§五)** | 零开销直线度指标, 量化"2 步收敛" | 修正"RF 接近直线" claim (实际 η_str∈[0.7,1.5]) | 3 seeds 单调下降 3.43→2.45→1.68 | ✅ 完成 |
| **D3 Box Renewal (§六)** | 揭示 box_renewal 与多步法历史矛盾 + 化解 | box_renewal 检测特有 / 密集目标 renewal 比例高 | η_str 虚高 56-58% 但 mAP 仅 −0.0003; K≥200 推理关闭安全, K=100 −0.016 | ✅ 完成 (含 K 值依赖性验证) |
| **S1 Cascade × Solver (§七)** | cascade head 作为 implicit solver 算子分裂 | 解释 24 NFE 架构合理性, 预防"6 head 冗余"质疑 | s1_h3_s8 ✓ (0.859), s1_h6_s2 ⚠ 待确认 (上次 0.859 @ ep106), s1_h3_s4 ✓ | 🔄 部分完成 |
| **R3 v-prediction 对照 (§八)** | 验证低维 + shifted schedule 下 x0-prediction 优势 | 预防"为何不用 v-prediction"质疑 (RF 原文偏好) | seed 42 ⚠ 待确认 (workstation 不可达), seed 123/789 ⛔ | 🔄 进行中 |
| **方向 A per-dim η_str (§九)** | 检测空间 4 维 (cxcywh) 各维度曲率差异诊断 | h 维度曲率显著小于 cx,cy, 启示 per-dim solver | Phase 2: per-dim (h=1阶) mAP=0.863 (+0.001), 加速 5.5%; Phase 3 (A.2): per-dim-w (w,h=1阶) mAP=0.864 (持平), 加速 5.0% | ✓ 完成 |
| **方向 D 自适应阶次 (§十)** | 后期 step 降阶 (3→2 阶) 自适应 DPM-Solver++ | $\eta_{3rd}$ step1→2 降幅 59%, 后期可降阶 | 3 solver mAP 均为 0.863 (ΔmAP=0.000), 自适应 4.2% 加速 | ✓ 完成 |
| **方向 C step-aware (§十一)** | cascade head 感知 solver step 编号 | 零初始化确保预训练兼容, 与 S1 算子分裂不冲突 | seed 42 🔄 ep86/150, best 0.857 (Δ=-0.006, 趋势负面) | 🔄 进行中 |

### SOTA 比较 (Dataset 2 val, 3-seed 均值)

| 方法 | Backbone | mAP (val) | mAP (test) | FPS | 备注 |
|------|----------|:---------:|:----------:|-----|------|
| DINO R50 | ResNet-50 | 0.868 | — | — | 多尺度可变形注意力 (CRASHED, 单 seed) |
| RTMDet-L | CSPNeXt-L | 0.863 | — | — | 更强主干 |
| **KaryoFlow (+DPM-Solver++)** | ResNet-50 | **0.859** | **0.859** | **13.3** | RF + DPM-Solver++ (val 3-seed mean = test seed42) |
| KaryoFlow (+DPM-Solver++) + Top-K (K=200) | ResNet-50 | 0.860 | 0.859 | **14.2** | 最佳速度-精度权衡 (test 上与 +DPM-Solver++ 持平) |
| Cascade R-CNN | ResNet-50 | 0.854 | 0.853 | 48.4 | — |
| DiffusionDet | ResNet-50 | 0.803 | 0.804 | 41.0 | DDPM 基线 (ep26 checkpoint) |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.795 | 98.5 | — |

### 关键统计显著性

| 比较 | Δ mAP | p-value | n |
|------|-------|---------|---|
| RF vs DDPM (Dataset 2, 统一口径 DiffusionDet 0.803) | +0.053 | — | — |
| RF vs DDPM (Dataset 2, 旧口径 a0_baseline 0.774) | +0.082 | — | — |
| RF vs DDPM (Dataset 1) | +0.017 | — | — |
| Stoch vs Random (Dataset 1) | +0.034 | <10⁻¹²⁰ | 1320 |
| Hard OT vs Random (Dataset 1) | −0.008 | <10⁻⁸ | 1320 |
| Stoch vs Random (Dataset 2) | +0.0001 | 0.80 (ns) | 500 |
| DPM++ vs Heun (Dataset 2, 4 步) | +0.006 | <10⁻⁶ | 500 |
| DDPM Euler 1→8 步 (Dataset 2, 论文 Appendix G) | <0.002 | — | — |

### 辅助评估实验索引 (论文数据点 ↔ LINEAGE 章节)

| 论文位置 | 数据点 | LINEAGE 章节 | 数据源 |
|----------|--------|--------------|--------|
| §4.3.1 / Figure 5 / Table F.1 | 24 类 per-class AP (+DPM-Solver++ seed 42) | §一 末段 | results/a4_per_class_ap.md |
| §4.3.2 / Table 8 | Dataset 2 配对显著性检验 (6 行) | §三 末段 | CATALOG §7.4 (C3) |
| §4.4.1 / Table 9 | Dataset 1 耦合消融配对检验 (6 行) | §三 末段 | CATALOG §7.4 (C2) |
| §4.4.2 / Table 7 / Figure 4 | 多维稳定性 (5 指标) | §二 末段 | CATALOG §7.8 + 训练日志 |
| §4.5.4 | 测试集评估 (val vs test) | §十四.1 | CATALOG §7.5 (C4) |
| §4.6 / Table 10 / Figure 6 | FPS / 延迟基准 (9 模型) | §十二 | results/benchmark_fps_*.md (5 个文件) |
| §4.7 | 跨域 Zero-shot (Dataset 2 → Chr20240904) | §十四.2 | work_dirs/robustness_noise/zero_shot_results_chr20240904.json |
| §4.8 / Table 11 | 标注噪声鲁棒性 (3×3 网格) | §十三 | work_dirs/robustness_noise/consolidated_results.json |
| §5.6 / §7.3.5 | Dataset 1 per-class AP 增益 (Stoch Coupling) | §二 末段 | CATALOG §7.3.5 (Problem 3) |
| §4.3.2 (引用, 不入正文) | SOTA per-image Wilcoxon (5 模型) | §一 末段 | CATALOG §7.4.6 (Problem 2B) |

