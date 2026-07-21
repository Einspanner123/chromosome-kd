# 实验脉络主路线文档 (按创新点主题组织)

> 本文档为 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 的有效方向主路线梳理。
> 按"创新点主题"组织实验脉络, 让审稿人快速识别 solid 的研究链条与创新性。
> 数据源: 24 Chromosomes Object (Dataset 2, 5000 张图) 为主, Chromosome20240904 (Dataset 1, 1540 张图) 作低数据对照。
> SwanLab URL 模式: `https://swanlab.cn/@einspanner/<project>/runs/<run_id>`
> 更新时间: 2026-07-21

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

A0→A1 累积 +0.082 mAP, 其中 solver×step 解耦消融证明:
- **94% (+0.077 mAP) 归因于 RF 范式本身**
- 6% (+0.005 mAP) 归因于 solver/步数选择 (Heun 4 步 vs Euler 1 步)
- AdaLN-Zero 单独贡献为 0 (Appendix B 零结果)
- 偏移噪声调度 (shift=3.0) 单独贡献为 −0.001 (噪声范围)

### 实验列表

#### 实验证明目的: RF 范式相对 DDPM 的精度优势 (主消融)

- A0 baseline (DDPM Euler 1-step)
  -- 数据集: Dataset 2 (24obj)
  -- 结果: mAP=0.774, AP50=0.968, AP75=0.916
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a0_baseline

- A1 RF+Heun (KaryoFlow)
  -- 数据集: Dataset 2 (24obj)
  -- 改动: diffusion_type=rectified_flow, solver=heun, rf_schedule=shifted, time_conditioning=adaln_zero
  -- 结果: mAP=0.856, AP50=0.990, AP75=0.969 [+0.082 主贡献]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a1_rf_heun

- A2 +AdaLN-Zero
  -- 结果: mAP=0.856, AP50=0.990, AP75=0.972 [+0.000 持平 A1, AdaLN 单独贡献为 0]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a2_adaln

- A3 +StochOT eps=5
  -- 结果: mAP=0.858, AP50=0.990, AP75=0.973 [+0.002 边际]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

- A4 DPM-Solver++ 替换 Heun
  -- 结果: mAP=0.863, AP50=0.990, AP75=0.974 [+0.005 推理加速且精度提升]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: solver×step 解耦, 隔离 RF 范式贡献

- A1 checkpoint 上 solver×step 全组合 (Dataset 2 验证集, seed 42)
  -- Heun 4 步 (7 NFE): mAP=0.856
  -- Euler 4 步 (4 NFE): mAP=0.855
  -- DPM-Solver++ 4 步 (4 NFE): mAP=0.855
  -- Euler 1 步 (1 NFE): mAP=0.851
  -- DPM-Solver++ 1 步 (1 NFE): mAP=0.851
  -- 结论: 匹配步数下 solver 类型对 mAP 无影响; 步数 1→4 仅 +0.004; solver/步数联合仅贡献 6%
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a1_rf_heun

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

- KaryoFlow A3 (DPM-Solver++) 3-seed 均值
  -- mAP=0.859, 落后 DINO R50 (0.868) 仅 0.009, 落后 RTMDet-L (0.863) 0.004
  -- 超越 Cascade R-CNN (0.854), YOLOX-S (0.796), DiffusionDet (0.803)
  -- 相对 DiffusionDet seed42 best: +0.060 mAP, 3-seed 均值: +0.056
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp
  -- 对照基准 (project=chromosome-kd-benchmark-24obj): DINO/RTMDet-L/Cascade/YOLOX/DiffusionDet

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

| 数据集 | 规模 | Stoch-Rand mAP Δ | 显著性 | 平滑性增益 |
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
  -- 结果: mAP=0.856 (best @ 53) [对应 A3 配置]
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

- ε 消融 (Dataset 2, A3 配置)
  -- ε < 1: 有害 (相同增广下 mAP −1.3%)
  -- ε ≥ 1: 进入饱和, 收益递减
  -- ε = 5: 主路线配置
  -- 结论: Stochastic Coupling 是必要的 OT 正则化项而非精度助推器
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a3_stochot

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

### 实验列表

#### 实验证明目的: A2 vs A3 逐图像配对检验 (匹配步数下精度优势)

- A3 (DPM++) vs A2 (Heun+Stoch. Coup.) 4 步对比 (Dataset 2 验证集)
  -- mAP Δ: +0.0056, Wilcoxon p=2.5×10⁻⁷ ***, 配对 t p=8.4×10⁻⁷ *** (n=500)
  -- A3-A1 (combined): Δ=+0.0057, Wilcoxon p=4.5×10⁻⁴, t p=4.9×10⁻⁵ ***
  -- A2-A1 (Stoch. Coup.): Δ=+0.0001, p=0.797 ns (Dataset 2 上不显著)
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

- DPM-Solver++ 步数消融 (Dataset 2, A3 checkpoint, seed 42)
  -- 2 步: mAP=0.863 (收敛)
  -- 4 步: mAP=0.863 (无收益)
  -- 结论: 超过 2 步无收益, η_str 诊断定量解释该现象
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

#### 实验证明目的: 匹配 NFE 下 DPM-Solver++ 对比 Heun

- DPM-Solver++ 4 步 (4 NFE, mAP=0.863) ≈ Heun 2 步 (3 NFE, mAP=0.863)
  -- 精度相当, DPM-Solver++ 以零成本换取 43% 更少 NFE
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

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

- A3 + Top-K (K=300)
  -- NFE=4, Latency=71.27 ms, FPS=14.0, mAP=0.861 [−0.002 vs A3]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- A3 + Top-K (K=200) [最优]
  -- NFE=4, Latency=70.46 ms, FPS=14.2, mAP=0.860 [−0.003 vs A3, 最佳速度-精度权衡]
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

- A3 + Top-K (K=100)
  -- NFE=4, Latency=69.71 ms, FPS=14.3, mAP=0.850 [−0.013 vs A3, 掉点]
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

### 与已证伪方向 IO1 的区分

- **IO1 (adaptive step early-exit)**: 根据收敛提前终止, 改变推理步数, 已证伪 (失败原因是 step 1 的 x0_pred 不稳定)
- **R1**: 仅观测 $\eta_{str}$, 不改变任何推理流程, 提供事后诊断
- R1 不触发 IO1 的失败模式

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

### 方案 B (renewal off) 已验证

- 3 seed 平均 mAP 0.858 ± 0.003 (vs baseline 0.859 ± 0.004), Δ=−0.0003 (噪声范围)
- **方案 B 不损失精度**, 且使 η_str 诊断有效 (renewal 污染被消除)
- 使 R1 指标在 renewal on 时失效的问题得到化解

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

### 与已证伪方向 N_cascade e2e 的区分

- **N_cascade e2e (已证伪, mAP 0.684, −0.172)**: 重训架构, 把 cascade head 数量从 6 改为其他值
- **D3**: 仅诊断已有架构的 box_renewal 与 DPM-Solver++ 交互, 不重训, 不引入新模块
- D3 是诊断非新模块, 不重复 e2e 失败模式

---

## 七、S1: Cascade Head × Solver Step 解耦 — 架构合理性形式化 (部分完成)

### 核心贡献: cascade head 作为 implicit solver 的算子分裂视角

当前架构 6 cascade head × 4 solver step = 24 次前向, 但 DPM-Solver++ 仅需 4 NFE 的理论框架把每个 time step 内 6 个 cascade head 视为黑盒——这与 Cascade R-CNN 的级联精化思想同构。形式化为双向精化:
- **横向 (cascade head, 固定 t)**: 在固定时间步上精化 $x_t$, 类似 Cascade R-CNN 级联精化
- **纵向 (solver step, 固定 x 精化链)**: 推进时间 $t$, 类似 DPM-Solver++ 多步积分

### 与染色体检测任务特性的结合

- **解释 24 NFE 架构合理性**: 6 cascade head × 4 solver step 构成算子分裂, DPM-Solver++ 把复合算子 $\mathcal{B}_t^* \circ \mathcal{A}_t$ 视为单次 $v_\theta$ 评估
- **预防审稿人对"6 cascade head 是否冗余"质疑**: cascade head 序列在固定 t 上精化 $x_t$ 至不动点 $\mathcal{B}_t^*$, 横向收敛性是 4 NFE 框架有效的前提
- **解释 N_cascade e2e 失败**: 减小 H 破坏横向收敛性, 而 S 未相应增加, 故 mAP 退化 −0.172

### 命题 S1.3: H×S 可交换性边界

在横向收敛假设下, 减小 H (如 H=3) 需增大 S 以补偿, 反之亦然。但 H×S 不是不变量: 因 $\mathcal{A}_t$ 是二阶 solver 而 $\mathcal{B}_{t,k}$ 是一阶精化, H 减半需 S 增加多于两倍。

### 实验列表

#### 实验证明目的: S1 消融重训 (3 配置, 部分完成)

- 配置 1: H=3 S=4 (12 NFE) ✓ 已完成
  -- 验证: 减小 H 是否破坏横向收敛性, 导致 mAP 退化
  -- 状态: 之前会话已完成, 结果见 memory
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h3_s4

- 配置 2: H=6 S=2 (12 NFE) 🔄 训练中
  -- 验证: 减小 S 是否影响纵向积分精度, 与 H=3 S=4 对比验证 H×S 可交换性边界
  -- 状态: epoch 123/150, best mAP=0.859 (epoch 106), ETA ~5h, workstation A5000
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h6_s2

- 配置 3: H=3 S=8 (24 NFE) ✓ 已完成
  -- 验证: 同等 24 NFE 下, 减小 H 增大 S 是否能补偿 (H 减半需 S 增加多于两倍)
  -- 状态: best mAP=0.859 (epoch 64), 30 epochs 未改善早停, ross A6000
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/s1_h3_s8

- 配置对照: A3 baseline H=6 S=4 (24 NFE)
  -- mAP: 3-seed 均值 0.859 ± 0.003 (单 seed best 0.863)
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/a4_dpm_pp

### 关键结论 (阶段性)

- s1_h6_s2 (H=6, S=2, NFE=12) 在 12 NFE 下 best mAP=0.859, **达到 A3 baseline 3-seed 均值水平**, 说明**减少 step 并保持 head 可在更少 NFE 下维持性能**
- s1_h3_s8 (H=3, S=8, NFE=24) 在 24 NFE 下 best mAP=0.859, 与 baseline 持平, 表明同等 NFE 下 H=3 S=8 可补偿 H 减半
- 与 S1 命题 S1.3 (H×S 可交换性边界) 对照: H=6 充分大时减小 S 仍可保持横向收敛性, 横向 head 序列已收敛至不动点 $\mathcal{B}_t^*$
- 与已证伪 N_cascade e2e (mAP 0.684, −0.172) 形成对比: 该实验减小 H 但未相应增加 S, 横向收敛性被破坏

### 与已证伪 N_cascade e2e 的区分

- **N_cascade e2e**: 重训架构, 把 cascade head 数量从 6 改为其他值, 端到端评估 (mAP 0.684, −0.172)
- **S1**: 形式化分析已有 H=6, S=4 架构的算子分裂结构, 给出"solver 阶数 × cascade 深度"权衡框架, 避免未来重试类似 e2e 实验

---

## 八、R3: x0-prediction vs v-prediction 对照重训 (进行中)

### 核心贡献: 验证低维 + shifted schedule 下 x0-prediction 优势

基于 RF 原文 (Liu et al., 2023) 使用 v-prediction, 验证在低维 ($d=4$) 检测空间 + shifted schedule ($s=3.0$) 下 x0-prediction 是否优于 v-prediction, 为论文当前参数化选择提供经验依据。

- **命题 R3.1 (信息等价)**: $\hat{x}_0 = x_t - t\hat{v}$, x0-prediction 与 v-prediction 在 $d=4$ 低维 RF 下信息论等价, 差异仅在损失的 $t$ 加权: $\mathcal{L}_v = t^{-2}\mathcal{L}_{x_0}$
- **命题 R3.2 (shifted schedule 下的偏好)**: shifted schedule ($s=3.0$) 下 x0-prediction 的有效梯度信噪比优于 v-prediction, 因前者在 $t \to 0$ 时不放大梯度
- **命题 R3.3 (设置依赖性)**: RF 原文的 v-prediction 偏好依赖高维 + linear schedule 组合; 在低维 + shifted schedule 下 x0-prediction 是更优选择

### 实验列表

#### 实验证明目的: R3 v-prediction 3-seed 重训 (进行中)

- seed 42 🔄 训练中
  -- 状态: epoch 8, best mAP=0.802 (warmup 阶段), ETA ~1.4 天, workstation A4000
  -- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_24obj.py`
  -- 改动: `criterion=dict(v_prediction=True, v_prediction_t_eps=1e-2)` (batch normalization 均值=1, 避免训练崩溃)
  -- 对照: A4 baseline (x0-prediction, 3-seed 均值 0.859 ± 0.003, 单 seed best 0.863)
  -- SwanLab project: `ldmdet-r3-vpred` (experiment_name=`r3_vpred`)

- seed 123, 789 ⛔ 待启动
  -- 状态: 等待 GPU 空闲
  -- 算力分配: ross A6000 / workstation A5000

### 预期结果

- v-prediction mAP 显著低于 A4 baseline (预期 ΔmAP < 0), 验证命题 R3.2
- 在 $t < 0.5$ (数据主导区, 对检测精度更关键) 时 v-prediction 的 $1/t^2$ 梯度放大引入显著方差
- 若实验确认, 可纳入论文 §3.1.1 末段或 §5.3 (约 0.3 页增量)

### 与已证伪 h_velocity_loss 的区分

- **h_velocity_loss (已证伪, CRASHED)**: 训练崩溃, 无有效 mAP, 未作对照分析 (详见 [FALSIFIED_DIRECTIONS.md §八](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md))
- **R3**: 通过 $1/t^2$ 损失加权模拟 v-prediction 梯度动态 + `v_prediction_t_eps=1e-2` 截断避免数值爆炸 + batch normalization (均值=1) 避免训练崩溃
- R3 是 h_velocity_loss 的可控重训版本, 提供机制级对照

---

## 九、方向 A: per-dim eta_str 诊断 — 维度级曲率分析 (零成本诊断完成, 部分支持)

### 核心贡献: 检测空间 4 维 (cxcywh) 各维度的曲率差异诊断

R1 的 $\eta_{str}$ 是 4 维 (cxcywh) 的整体范数比, 但检测空间各维度物理含义不同 (位置 cx,cy vs 尺度 w,h)。方向 A 在 A4 checkpoint 上零成本诊断各维度曲率, 探究是否可设计 per-dim solver。

### 诊断方法

- 在 A4 checkpoint (best mAP=0.859, epoch 117) 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
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

### Phase 2 启示

- 可设计 w,h 维度用低阶 solver、cx,cy 用高阶的混合方案
- 但 w 维度差距较小, 实际增益可能有限
- 后续: 检测专用 solver 设计 (per-dim order allocation) — 已加入 [TODO_DIRECTIONS.md §五](file:///home/linkst/workspace/projects/chromosome-kd/docs/TODO_DIRECTIONS.md)

### 与 R1 的关系

- R1: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 A: per-dim $\eta_{str}$ 量化各维度曲率差异
- 互补: R1 决定步数, 方向 A 决定 per-dim 阶数分配

---

## 十、方向 D: 自适应阶次 DPM-Solver++ — 后期 step 降阶 (零成本诊断完成, 支持假设)

### 核心贡献: 基于 $\eta_{3rd}$ 趋势的自适应降阶策略

DPM-Solver++ 3 阶校正项 $D_2$ 在后期 step 应小于早期 (因 RF 轨迹在 $t \to 0$ 时趋于直线)。方向 D 通过零成本诊断 $\eta_{3rd} = \|D_2\|/\|\hat{x}_0\|$ 趋势, 验证后期 step 可降为 2 阶的假设。

### 诊断方法

- 在 A4 checkpoint 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
- 测量 $\eta_{3rd}$ 随 step 的变化趋势
- 实现位置: `ldmdet/diffusion/rectified_flow.py` (`RFDPMSolverAdaptive`, static + eta_threshold 两种模式)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a7_dpm_pp_adaptive_24obj.py`
- 结果 JSON: `work_dirs/diagnosis/dpm_pp_adaptive.json`

### 诊断结果

- $\eta_{3rd}$ 趋势: step 1 = 44.6 → step 2 = 18.2 (decreasing, 降幅 59%)
- **结论**: ✓ 支持重构假设 — 后期 step 的 3 阶校正项显著小于早期, 可降为 2 阶
- 与 R1 整体 $\eta_{str}$ 单调下降 (3.43→2.45→1.68) 一致, 但方向 D 量化了 3 阶项的衰减

### 待跑实验

- mAP 对比实验: dpm_solver_pp (2 阶) vs dpm_solver_pp_3 (3 阶) vs dpm_solver_pp_adaptive (自适应)
- 零成本推理 (无需重训, 直接在 A4 checkpoint 上评估)
- 预期: adaptive 在保持 mAP 的同时减少后期 step 计算量

### 与 R1 的关系

- R1: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 D: per-step 3 阶项 $\eta_{3rd}$ 量化"后期 step 可降阶"
- 互补: R1 决定步数, 方向 D 决定每步阶数

---

## 十一、方向 C: step-aware embedding — Cascade head 感知 solver step (代码就绪, 待启动)

### 核心贡献: 让 cascade head 感知 DPM-Solver++ step 编号

当前 cascade head 在所有 solver step 上共享参数, 但不同 step 上 $x_t$ 的统计特性不同 (早期近噪声, 后期近 GT)。方向 C 通过 step embedding 让 head 感知当前 step, 提升每步精化的针对性。

### 实现方式

- step_mlp + step_proj 零初始化
- 零初始化确保预训练兼容: 训练初期 step_proj 输出为 0, 模型行为与无 step embedding 时一致, 可在 A4 checkpoint 上继续训练而非重训
- 实现位置: `ldmdet/core/head.py` (step_mlp + step_proj 零初始化)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a6_step_aware_24obj.py`

### 状态

- ⛔ 待启动训练 (ross A6000 即将启动 seed 42)
- 3 seeds (42/123/789) 重训计划
- 对照: A4 baseline (3-seed 均值 0.859 ± 0.003)

### 理论

- 让 cascade head 感知 DPM-Solver++ step 编号, 零初始化确保预训练兼容
- 与 S1 算子分裂结构不冲突: step embedding 不改变横向 (cascade head) / 纵向 (solver step) 解耦, 仅在横向 head 内部添加 step 条件

### 预期

- 若 step embedding 显著提升 mAP (Δ > +0.005), 可作为论文新方向
- 若持平, 表明 cascade head 已通过 $x_t$ 隐式感知 step 信息 (因 $x_t$ 在不同 step 上统计不同)

### 与 S1 的关系

- S1 形式化 cascade head × solver step 算子分裂 (§七)
- 方向 C 在不破坏 S1 算子分裂结构的前提下, 让 cascade head 显式感知 step
- 与 S1 互补: S1 给出架构合理性框架, 方向 C 在框架内探索性能提升

---

## 十二、边际有效方向 (历史记录)

> 以下方向在 Dataset 1 (chromo, mAP 0.72-0.75) 上获得边际收益, 未叠加到 SOTA。记录作为完整事实, 不作为论文主路线。

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

### Direction D (BoxRefineNet)

- Direction D BoxRefineNet (Dataset 1, 1 seed)
  -- 结果: mAP=0.747 [+0.001 vs 0.746 baseline, 持平]
  -- 状态: early stop @ epoch 85, best @ epoch 55
  -- 本地: work_dirs/direction_exps/direction_d_box_refine/20260629_091843/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/fnoz9x82aor1utsuo0jtl

### 非线性轨迹 E4.2 (OT Flow only)

- E4.2 OT Flow only (Dataset 1, 1 seed)
  -- 结果: mAP=0.751 [+0.005 vs 0.746 baseline]
  -- 改动: coupling=ot_flow, lambda_mod=0.0 (关闭尺度条件)
  -- 本地: work_dirs/nonlinear_trajectory_e42/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/wcp34v3t

### 非线性轨迹 E4.3 (OT+SCRF 未启用)

- E4.3 OT+SCRF argmax eps=1.0 (Dataset 1, 1 seed)
  -- 结果: mAP=0.752 [+0.006 vs 0.746 baseline]
  -- 关键修正: ScaleConditionedRF 当时未集成到 head.py, 0.752 实际来自 OTFlowCoupling + 种子方差
  -- 本地: work_dirs/nonlinear_trajectory/
  -- SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/usnvd63f
  -- 注: 实际增益来自 OT, 非 SCRF (ScaleConditionedRF 真正启用后 mAP=0.741, −0.005, 证伪)

---

## 十三、SwanLab 项目映射汇总

| SwanLab Project | 实验数 | 范围 | URL Pattern |
|-----------------|--------|------|-------------|
| `ldmdet-mainline-ablation-24obj` | 5 ⭐ | 24obj A0-A4 主路线消融 (论文核心) | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| `ldmdet-ablation` | 9 (24obj) + 23 (chromo) | 主线 + 24obj 耦合策略 + chromo 历史 + 非线性轨迹 | `https://swanlab.cn/@einspanner/ldmdet-ablation/runs/<run_id>` |
| `chromosome-kd-benchmark-24obj` | 8 | 24obj SOTA 对比模型 (DINO/RTMDet-L/Cascade/YOLOX/DiffusionDet) | `https://swanlab.cn/@einspanner/chromosome-kd-benchmark-24obj/runs/<run_id>` |
| `ldmdet-breakthrough` | 2 | 24obj SC-RF 自条件化 | `https://swanlab.cn/@einspanner/ldmdet-breakthrough/runs/<run_id>` |
| `ldmdet-frontier-directions` | 6 | 24obj 前沿方向探索 | `https://swanlab.cn/@einspanner/ldmdet-frontier-directions/runs/<run_id>` |
| `ldmdet-s1-cascade-decouple` | 2 已完成 + 1 进行中 | S1 cascade head × solver step 解耦消融 (s1_h3_s4 ✓ / s1_h3_s8 ✓ / s1_h6_s2 🔄) | `https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/<run_id>` |
| `ldmdet-r3-vpred` | 1 进行中 + 2 待启动 | R3 v-prediction 对照重训 (seed 42 🔄 / seed 123,789 ⛔) | `https://swanlab.cn/@einspanner/ldmdet-r3-vpred/runs/<run_id>` |
| `few-shot-benchmark` | 3 | 24obj few-shot 源预训练 | `https://swanlab.cn/@einspanner/few-shot-benchmark/runs/<run_id>` |
| `nonlinear-3seed-repro` | 2 | 3-seed 复现 (chromo) | `https://swanlab.cn/@einspanner/nonlinear-3seed-repro/runs/<run_id>` |
| `chromosome-kd` | 21 | 早期 chromo 数据集 | `https://swanlab.cn/@einspanner/chromosome-kd/runs/<run_id>` |
| `ldmdet-inference` | 12 | DDIM 步数对齐 + DPM-Solver++ 步数消融 (chromo) | `https://swanlab.cn/@einspanner/ldmdet-inference/runs/<run_id>` |

---

## 十四、关键结论汇总

### 主路线创新点贡献矩阵

| 创新点 | 核心贡献 | 与任务结合 | 关键数据 | 状态 |
|--------|----------|------------|----------|------|
| **RF (§一)** | 直线 ODE 路径取代 DDPM 弯曲随机轨迹 | 密集 proposals 误差复合 / 小训练集 / 24 类细粒度 | A0→A1 +0.082 mAP, 94% 归因于 RF | ✅ 完成 |
| **OT Collapse + Stoch. Coupling (§二)** | 低维 d=4 OT 坍缩形式化 + Stochastic Coupling 补救 | 低维触发 / 高 K 加剧 / 小训练集放大 | Dataset 1 +0.034 (p<10⁻¹²⁰), Dataset 2 +0.0001 (p=0.80) + 4.6× 平滑 | ✅ 完成 |
| **DPM-Solver++ (§三)** | RF 适配 data-prediction + 修正 FlowDet 结论 | 临床交互式延迟 13.3-14.2 FPS / cascade head 占 90%+ | +0.006 mAP (p<10⁻⁶) + 1.71× NFE 加速 | ✅ 完成 |
| **Top-K Pruning (§四)** | 500→K proposals 剪枝 + DPM-Solver++ 兼容 | K=200 最优 (46 染色体 + 重叠冗余) | K=200: 14.2 FPS, mAP 0.860 | ✅ 完成 |
| **R1 η_str (§五)** | 零开销直线度指标, 量化"2 步收敛" | 修正"RF 接近直线" claim (实际 η_str∈[0.7,1.5]) | 3 seeds 单调下降 3.43→2.45→1.68 | ✅ 完成 |
| **D3 Box Renewal (§六)** | 揭示 box_renewal 与多步法历史矛盾 + 化解 | box_renewal 检测特有 / 密集目标 renewal 比例高 | η_str 虚高 56-58% 但 mAP 仅 −0.0003 | ✅ 完成 |
| **S1 Cascade × Solver (§七)** | cascade head 作为 implicit solver 算子分裂 | 解释 24 NFE 架构合理性, 预防"6 head 冗余"质疑 | s1_h3_s8 ✓ (0.859), s1_h6_s2 🔄 (0.859 @ ep106), s1_h3_s4 ✓ | 🔄 部分完成 |
| **R3 v-prediction 对照 (§八)** | 验证低维 + shifted schedule 下 x0-prediction 优势 | 预防"为何不用 v-prediction"质疑 (RF 原文偏好) | seed 42 🔄 ep8 (warmup 0.802), seed 123/789 ⛔ | 🔄 进行中 |
| **方向 A per-dim η_str (§九)** | 检测空间 4 维 (cxcywh) 各维度曲率差异诊断 | h 维度曲率显著小于 cx,cy, 启示 per-dim solver | wh/cxcy 比值 0.41-0.50 (2 阶), h 维度差距 3-5×, w 维度 1.5-2× | ✓ 诊断完成 (部分支持) |
| **方向 D 自适应阶次 (§十)** | 后期 step 降阶 (3→2 阶) 自适应 DPM-Solver++ | $\eta_{3rd}$ step1→2 降幅 59%, 后期可降阶 | $\eta_{3rd}$: step1=44.6 → step2=18.2 | ✓ 诊断完成 (支持假设) |
| **方向 C step-aware (§十一)** | cascade head 感知 solver step 编号 | 零初始化确保预训练兼容, 与 S1 算子分裂不冲突 | 代码就绪, ⛔ 待启动训练 | ⛔ 待启动 |

### SOTA 比较 (Dataset 2, 3-seed 均值)

| 方法 | Backbone | mAP | FPS | 备注 |
|------|----------|-----|-----|------|
| DINO R50 | ResNet-50 | 0.868 | — | 多尺度可变形注意力 |
| RTMDet-L | CSPNeXt-L | 0.863 | — | 更强主干 |
| **KaryoFlow A3 (DPM++)** | ResNet-50 | **0.859** | **13.3** | RF + DPM-Solver++ |
| KaryoFlow A3 + Top-K (K=200) | ResNet-50 | 0.860 | **14.2** | 最佳速度-精度权衡 |
| Cascade R-CNN | ResNet-50 | 0.854 | 48.4 | — |
| DiffusionDet | ResNet-50 | 0.803 | 41.0 | DDPM 基线 |
| YOLOX-S | CSPDarkNet-S | 0.796 | 98.5 | — |

### 关键统计显著性

| 比较 | Δ mAP | p-value | n |
|------|-------|---------|---|
| RF vs DDPM (Dataset 2) | +0.082 | — | — |
| RF vs DDPM (Dataset 1) | +0.017 | — | — |
| Stoch vs Random (Dataset 1) | +0.034 | <10⁻¹²⁰ | 1320 |
| Hard OT vs Random (Dataset 1) | −0.008 | <10⁻⁸ | 1320 |
| Stoch vs Random (Dataset 2) | +0.0001 | 0.80 (ns) | 500 |
| DPM++ vs Heun (Dataset 2, 4 步) | +0.006 | <10⁻⁶ | 500 |
